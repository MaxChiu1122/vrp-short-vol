"""Position sizing: compounding, leverage, ruin, and no look-ahead."""

from __future__ import annotations

import numpy as np
import pytest

from vrp.roll import monthly_roll
from vrp.sizing import (
    apply_leverage,
    compare_lookahead,
    cvar_budget_leverage,
    fixed_leverage,
    growth_optimal_leverage,
    inverse_implied_leverage,
    leverage_sweep,
    vol_target_leverage,
)

ROLL = monthly_roll(implied_haircut=0.010, cost_bps=1.0)


# --------------------------------------------------------------------------------
# Compounding
# --------------------------------------------------------------------------------


def test_unit_leverage_compounds_the_roll():
    sized = apply_leverage(ROLL, fixed_leverage(ROLL, 1.0))
    assert np.allclose(sized.equity, np.cumprod(1.0 + ROLL.pnl))
    assert sized.cagr > 0.0
    assert not sized.ruined


def test_sharpe_is_invariant_to_leverage():
    """Scale-free by construction -- which is exactly why Sharpe cannot size a book."""
    a = apply_leverage(ROLL, fixed_leverage(ROLL, 1.0))
    b = apply_leverage(ROLL, fixed_leverage(ROLL, 7.0))
    assert a.sharpe_annualized == pytest.approx(b.sharpe_annualized, rel=1e-9)


def test_leverage_scales_risk_and_return_together():
    a = apply_leverage(ROLL, fixed_leverage(ROLL, 1.0))
    b = apply_leverage(ROLL, fixed_leverage(ROLL, 4.0))
    assert b.vol_annualized == pytest.approx(4.0 * a.vol_annualized, rel=1e-9)
    assert b.max_drawdown < a.max_drawdown
    assert b.cagr > a.cagr


def test_ruin_is_absorbing():
    """A wipe-out stays wiped out -- no borrowing back from zero."""
    ruinous = apply_leverage(ROLL, fixed_leverage(ROLL, 40.0))
    assert ruinous.ruined
    assert ruinous.cagr == float("-inf")
    assert ruinous.terminal_multiple == 0.0
    first_zero = int(np.argmax(ruinous.equity <= 0.0))
    assert (ruinous.equity[first_zero:] == 0.0).all()


def test_the_leverage_cliff_exists():
    """Growth rises with leverage until a single month can end the book."""
    sweep = leverage_sweep(ROLL, [1, 5, 10, 20, 40])
    survivors = [(lev, s) for lev, s in sweep if not s.ruined]
    assert len(survivors) >= 3
    cagrs = [s.cagr for _lev, s in survivors]
    assert cagrs == sorted(cagrs), "growth should rise with leverage while solvent"
    assert sweep[-1][1].ruined, "enough leverage must eventually ruin the book"


def test_growth_optimal_leverage_is_high_and_intolerable():
    """Full Kelly sits essentially at the ruin boundary -- the case for a fraction of it."""
    kelly = growth_optimal_leverage(ROLL)
    assert kelly > 5.0
    at_kelly = apply_leverage(ROLL, fixed_leverage(ROLL, kelly))
    assert at_kelly.max_drawdown < -0.60, "if Kelly were comfortable, nobody would warn about it"


# --------------------------------------------------------------------------------
# The rules
# --------------------------------------------------------------------------------


def test_rules_produce_one_leverage_per_trade():
    for lev in (
        vol_target_leverage(ROLL),
        cvar_budget_leverage(ROLL),
        inverse_implied_leverage(ROLL),
    ):
        assert lev.shape == ROLL.pnl.shape
        assert np.isfinite(lev).all()
        assert (lev >= 0).all()


def test_trailing_rules_use_no_future_information():
    """Truncating the history must not change the leverage of the trades that remain.

    This is the look-ahead test: if a rule peeked forward, shortening the sample
    would change earlier decisions.
    """
    import dataclasses

    cut = 300
    short = dataclasses.replace(
        ROLL,
        dates=ROLL.dates[:cut],
        spot=ROLL.spot[:cut],
        implied=ROLL.implied[:cut],
        realized=ROLL.realized[:cut],
        pnl=ROLL.pnl[:cut],
        market_return=ROLL.market_return[:cut],
    )
    for rule in (vol_target_leverage, cvar_budget_leverage, inverse_implied_leverage):
        assert np.allclose(rule(ROLL)[:cut], rule(short)), rule.__name__


def test_inverse_implied_sizes_down_when_vol_is_high():
    lev = inverse_implied_leverage(ROLL)
    high = ROLL.implied > np.median(ROLL.implied)
    assert lev[high].mean() < lev[~high].mean()


def test_leverage_rules_respect_their_cap():
    assert vol_target_leverage(ROLL, cap=3.0).max() <= 3.0
    assert cvar_budget_leverage(ROLL, cap=3.0).max() <= 3.0
    assert inverse_implied_leverage(ROLL, cap=3.0).max() <= 3.0


def test_implied_vol_sizing_beats_trailing_realized_sizing():
    """The finding: forward-looking market information beats backward-looking stats.

    Compared at matched AVERAGE leverage, so this is about the shape of the rule,
    not about one of them simply taking more risk.
    """

    def matched(lev, mean_lev=5.0):
        return apply_leverage(ROLL, lev * (mean_lev / lev.mean()))

    implied = matched(inverse_implied_leverage(ROLL))
    trailing = matched(vol_target_leverage(ROLL, target_vol=0.10))
    assert implied.sharpe_annualized > trailing.sharpe_annualized
    assert implied.max_drawdown > trailing.max_drawdown  # both negative: shallower


def test_trailing_vol_targeting_does_not_beat_doing_nothing():
    """Uncomfortable but real: it levers up into calm and gets caught."""

    def matched(lev, mean_lev=5.0):
        return apply_leverage(ROLL, lev * (mean_lev / lev.mean()))

    flat = matched(fixed_leverage(ROLL, 1.0))
    trailing = matched(vol_target_leverage(ROLL, target_vol=0.10))
    assert trailing.sharpe_annualized < flat.sharpe_annualized


# --------------------------------------------------------------------------------
# Hindsight
# --------------------------------------------------------------------------------


def test_lookahead_sizing_flatters_the_result():
    """Calibrating on the full sample knows in 1990 how bad 2008 will be."""
    out = compare_lookahead(ROLL)
    assert out["lookahead"].sharpe_annualized > out["trailing"].sharpe_annualized
    assert out["lookahead"].max_drawdown > out["trailing"].max_drawdown  # shallower
