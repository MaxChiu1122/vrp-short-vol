"""Transaction costs and the rebalancing-frequency trade-off.

The claim being pinned: hedging risk falls like 1/sqrt(n_steps) but turnover grows
like sqrt(n_steps), so with costs on there is an interior optimal hedge frequency
and it moves toward *less* hedging as costs rise.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vrp import hedge_frequency_sweep, vrp_backtest, vrp_pnl_paths

BASE = dict(sigma_implied=0.20, sigma_realized=0.16, n_paths=4_000, seed=11)


def test_zero_cost_is_exactly_the_old_behaviour():
    """cost_bps defaults to 0 and must not perturb a single path."""
    assert np.array_equal(vrp_pnl_paths(**BASE), vrp_pnl_paths(**BASE, cost_bps=0.0))


def test_costs_reduce_the_edge_monotonically():
    means = [vrp_backtest(**BASE, cost_bps=c).mean_pnl for c in (0.0, 1.0, 5.0, 10.0)]
    assert means == sorted(means, reverse=True), f"cost should only ever hurt: {means}"


def test_costs_do_not_change_the_paths_only_the_pnl():
    """Cost is a deterministic drag: the same seed must draw the same market."""
    free = vrp_pnl_paths(**BASE, cost_bps=0.0)
    charged = vrp_pnl_paths(**BASE, cost_bps=5.0)
    # Every path is worse off by some positive amount, and never by the same amount
    # (turnover is path-dependent).
    drag = free - charged
    assert (drag > 0).all()
    assert drag.std(ddof=1) > 0.0


def test_cost_drag_scales_like_sqrt_of_rebalance_count():
    """The scaling that creates the optimum.

    Per rebalance you trade ~ Gamma*S*sigma*sqrt(dt) shares, so turnover over
    n_steps rebalances grows like sqrt(n_steps). The initial hedge is a FIXED cost
    that does not scale, so it is subtracted before checking the exponent.
    """
    cost_bps = 10.0
    kwargs = dict(sigma_implied=0.20, sigma_realized=0.16, n_paths=6_000, seed=11)

    def drag(n_steps: int) -> float:
        free = vrp_backtest(**kwargs, n_steps=n_steps).mean_pnl
        charged = vrp_backtest(**kwargs, n_steps=n_steps, cost_bps=cost_bps).mean_pnl
        return free - charged

    # Cost of putting the initial hedge on: rate * delta0 * S0, independent of n.
    import mcpricer as mc

    delta0 = mc.black_scholes_delta(mc.OptionType.Call, 100.0, 100.0, 0.02, 0.20, 1 / 12)
    fixed = cost_bps * 1e-4 * delta0 * 100.0

    ratio = (drag(252) - fixed) / (drag(21) - fixed)
    expected = math.sqrt(252 / 21)
    assert ratio == pytest.approx(expected, rel=0.20), (
        f"turnover should scale ~sqrt(n): got {ratio:.2f}, expected ~{expected:.2f}"
    )


def test_fast_backend_refuses_costs_rather_than_ignoring_them():
    with pytest.raises(NotImplementedError, match="does not model transaction costs"):
        vrp_pnl_paths(**BASE, backend="fast", cost_bps=5.0)


# --------------------------------------------------------------------------------
# The sweep
# --------------------------------------------------------------------------------


def test_sweep_covers_the_whole_grid():
    sweep = hedge_frequency_sweep(
        n_steps_grid=(11, 21), cost_bps_grid=(0.0, 5.0), n_paths=1_000, seed=3
    )
    assert len(sweep) == 4
    assert {(n, c) for n, c, _r in sweep} == {(11, 0.0), (11, 5.0), (21, 0.0), (21, 5.0)}


def test_without_costs_more_hedging_is_monotonically_better():
    """Risk is pure discretization error, so refining the hedge only ever helps."""
    sweep = hedge_frequency_sweep(
        n_steps_grid=(11, 21, 42, 84), cost_bps_grid=(0.0,), n_paths=8_000, seed=11
    )
    sharpes = [r.sharpe for _n, _c, r in sweep]
    assert sharpes == sorted(sharpes), f"zero-cost Sharpe should rise with frequency: {sharpes}"


def test_with_costs_the_optimum_is_interior():
    """The headline: at a realistic single-name cost, over-hedging destroys the edge."""
    n_steps_grid = (11, 21, 42, 84, 168, 336)
    sweep = hedge_frequency_sweep(
        n_steps_grid=n_steps_grid, cost_bps_grid=(10.0,), n_paths=8_000, seed=11
    )
    sharpes = [r.sharpe for _n, _c, r in sweep]
    best = int(np.argmax(sharpes))
    assert 0 < best < len(sharpes) - 1, f"expected an interior peak, got {sharpes}"
    # And past the peak it degrades badly enough to lose money outright.
    assert sweep[-1][2].mean_pnl < sweep[best][2].mean_pnl
