"""Acceptance tests for the VRP core (vrp.hedge.hedged_short_option_pnl).

These encode the Definition of Done in CLAUDE.md section 5: the sign of the mean P&L
must follow the sign of the implied-minus-realized vol spread. Each test states the
finance it is checking, so a failure points at a mechanism, not just a number.

Until ``hedged_short_option_pnl`` is implemented the whole module SKIPS, so the repo
stays green (CLAUDE.md section 3, "always-green"). It arms itself automatically the
moment the core stops raising NotImplementedError.
"""

from __future__ import annotations

import numpy as np
import pytest

from vrp import DEFAULT_CONTRACT, hedged_short_option_pnl, vrp_backtest, vrp_curve

# Enough paths that the Monte-Carlo standard error is far smaller than the effects we
# assert on, but small enough that the suite stays fast.
N_PATHS = 8_000
SIGMA_REALIZED = 0.16
SPREAD = 0.04  # 4 vol points of implied-over-realized: a generous but realistic VRP


def _core_is_implemented() -> bool:
    try:
        hedged_short_option_pnl(
            S0=100.0,
            K=100.0,
            r=0.02,
            sigma_implied=0.20,
            sigma_realized=0.20,
            T=1.0 / 12.0,
            n_steps=5,
            rng=np.random.default_rng(0),
        )
    except NotImplementedError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _core_is_implemented(),
    reason="hedged_short_option_pnl not implemented yet (Max's core) -- see hedge.py",
)


def _std_error(result) -> float:
    """Monte-Carlo standard error of the reported mean P&L."""
    return result.std_pnl / np.sqrt(result.n_paths)


# --------------------------------------------------------------------------------
# The three Definition-of-Done criteria: sign(mean P&L) == sign(implied - realized)
# --------------------------------------------------------------------------------


def test_zero_spread_gives_zero_mean_pnl():
    """sigma_implied == sigma_realized  =>  mean P&L == 0, and *exactly* so.

    This is the sharpest test in the file. The paths are simulated under the
    risk-neutral drift r, so each rebalance contributes delta_i * (S_{i+1} - S_i e^{r dt}),
    a martingale increment with zero conditional mean, and the premium collected is by
    construction the discounted expected payoff. So E[P&L] is exactly zero for ANY
    number of hedges -- discrete hedging adds noise, not bias.

    Consequence: a failure here is a *structural* bug, not a discretization artifact.
    The usual culprit is forgetting to grow cash at the risk-free rate (cash *= exp(r*dt)),
    which breaks the self-financing identity and shows up as a systematic drift.
    """
    res = vrp_backtest(
        sigma_implied=SIGMA_REALIZED,
        sigma_realized=SIGMA_REALIZED,
        n_paths=N_PATHS,
        seed=11,
    )
    assert abs(res.mean_pnl) < 4.0 * _std_error(res), (
        f"zero vol spread should earn nothing in expectation, got {res.mean_pnl:+.4f} "
        f"+/- {_std_error(res):.4f} (1 s.e.)"
    )


def test_positive_spread_makes_money():
    """sigma_implied > sigma_realized  =>  mean P&L > 0: the variance risk premium.

    You sold at a vol the world did not deliver; the gamma-weighted vol spread
    1/2 * sum_t S_t^2 * Gamma_t * (sigma_impl^2 - sigma_real^2) * dt accrues to you.
    """
    res = vrp_backtest(
        sigma_implied=SIGMA_REALIZED + SPREAD,
        sigma_realized=SIGMA_REALIZED,
        n_paths=N_PATHS,
        seed=11,
    )
    assert res.mean_pnl > 3.0 * _std_error(res), (
        f"selling {SPREAD:.0%} vol above realized should earn a premium, "
        f"got {res.mean_pnl:+.4f} +/- {_std_error(res):.4f} (1 s.e.)"
    )


def test_negative_spread_loses_money():
    """sigma_implied < sigma_realized  =>  mean P&L < 0.

    The symmetric case, and the one that matters for honesty: the same machinery that
    prints the premium must bleed when you sell vol too cheap. If only the profitable
    direction works, the P&L is not the vol spread -- it is a sign bug.
    """
    res = vrp_backtest(
        sigma_implied=SIGMA_REALIZED - SPREAD,
        sigma_realized=SIGMA_REALIZED,
        n_paths=N_PATHS,
        seed=11,
    )
    assert res.mean_pnl < -3.0 * _std_error(res), (
        f"selling {SPREAD:.0%} vol below realized should lose money, "
        f"got {res.mean_pnl:+.4f} +/- {_std_error(res):.4f} (1 s.e.)"
    )


# --------------------------------------------------------------------------------
# Shape of the edge and of the distribution
# --------------------------------------------------------------------------------


def test_mean_pnl_increases_with_the_vol_spread():
    """The VRP curve slopes up: more implied-over-realized sold, more edge earned.

    vrp_curve reuses one seed across spreads (common random numbers), so the paths are
    identical and only the vol spread differs -- the comparison is far sharper than the
    per-point standard error would suggest.
    """
    curve = vrp_curve(
        spreads=[-0.04, -0.02, 0.0, 0.02, 0.04],
        sigma_realized=SIGMA_REALIZED,
        n_paths=2_000,
        seed=7,
    )
    means = [mean_pnl for _spread, mean_pnl, _sharpe in curve]
    assert means == sorted(means), f"mean P&L not monotone in the vol spread: {means}"
    assert means[0] < 0.0 < means[-1], "curve should cross zero near a zero vol spread"


def test_short_gamma_distribution_is_left_skewed():
    """Being short gamma, the P&L distribution has its long tail on the LOSS side.

    Per step the hedged P&L is ~ 1/2 * S^2 * Gamma * (sigma_impl^2 dt - (dS/S)^2).
    The realized-variance term (dS/S)^2 is chi-squared(1) -- strongly right-skewed --
    and it enters with a MINUS sign for a short position. So the P&L inherits a left
    skew: many small wins, occasional large losses. This is the "what kills the book"
    property, and it must be visible even at a zero vol spread.
    """
    rng = np.random.default_rng(3)
    cfg = {**DEFAULT_CONTRACT, "n_steps": 21}
    pnls = np.array(
        [
            hedged_short_option_pnl(
                sigma_implied=SIGMA_REALIZED,
                sigma_realized=SIGMA_REALIZED,
                rng=rng,
                **cfg,
            )
            for _ in range(N_PATHS)
        ]
    )
    centered = pnls - pnls.mean()
    skew = float((centered**3).mean() / centered.std() ** 3)
    assert skew < -0.1, f"short-gamma P&L should be left-skewed, got skew {skew:+.3f}"


def test_cvar_is_a_loss_even_when_the_strategy_is_profitable():
    """A positive mean coexists with a negative 5% tail -- the short-vol signature.

    Reporting only the mean would make this strategy look like free money; the whole
    reason the harness reports CVaR is that the tail stays ugly while the mean is good.
    """
    res = vrp_backtest(
        sigma_implied=SIGMA_REALIZED + SPREAD,
        sigma_realized=SIGMA_REALIZED,
        n_paths=N_PATHS,
        seed=11,
    )
    assert res.mean_pnl > 0.0
    assert res.cvar_05 < 0.0, (
        f"expected a losing 5% tail alongside a {res.mean_pnl:+.4f} mean, "
        f"got CVaR {res.cvar_05:+.4f}"
    )


# --------------------------------------------------------------------------------
# Reproducibility (CLAUDE.md section 3: same seed -> same P&L distribution)
# --------------------------------------------------------------------------------


def test_same_seed_reproduces_the_distribution():
    kwargs = dict(
        sigma_implied=SIGMA_REALIZED + SPREAD,
        sigma_realized=SIGMA_REALIZED,
        n_paths=1_000,
    )
    first = vrp_backtest(seed=42, **kwargs)
    second = vrp_backtest(seed=42, **kwargs)
    assert first == second


def test_different_seeds_give_different_draws():
    kwargs = dict(
        sigma_implied=SIGMA_REALIZED + SPREAD,
        sigma_realized=SIGMA_REALIZED,
        n_paths=1_000,
    )
    assert vrp_backtest(seed=1, **kwargs).mean_pnl != vrp_backtest(seed=2, **kwargs).mean_pnl
