"""The historical monthly roll and the equity-beta decomposition.

Offline: everything runs off the vendored data/spx_vix_daily.csv.
"""

from __future__ import annotations

import numpy as np
import pytest

from vrp.marketdata import load_market_data, log_returns
from vrp.roll import monthly_roll, regress_on_market

ROLL = monthly_roll(implied_haircut=0.010, cost_bps=1.0)  # the central, defensible case


# --------------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------------


def test_windows_are_non_overlapping_and_cover_the_history():
    """~438 independent monthly trades, not 9,000 correlated overlapping ones."""
    n_returns = log_returns(load_market_data().spx).size
    assert len(ROLL) == n_returns // 21
    assert len(ROLL) > 400


def test_trades_are_in_chronological_order():
    assert (np.diff(ROLL.dates) > np.timedelta64(0, "D")).all()


def test_pnl_is_a_fraction_of_notional():
    """Not index points: the S&P ran 359 -> 7400, so raw P&L is incomparable."""
    assert np.abs(ROLL.pnl).max() < 0.20  # a month never moves 20% of notional
    assert ROLL.spot.min() < 500 < ROLL.spot.max()  # but the index level really did move


def test_implied_and_realized_are_recorded_per_trade():
    assert (ROLL.implied > 0).all()
    assert (ROLL.realized > 0).all()
    assert 0.10 < ROLL.implied.mean() < 0.30


# --------------------------------------------------------------------------------
# Performance, and the things only a time axis shows
# --------------------------------------------------------------------------------


def test_the_roll_made_money_over_36_years():
    assert ROLL.mean_pnl > 0.0
    assert ROLL.hit_rate > 0.6
    assert ROLL.sharpe_annualized > 1.0


def test_equity_curve_is_the_running_sum():
    assert np.allclose(ROLL.equity_curve, np.cumsum(ROLL.pnl))
    assert ROLL.equity_curve[-1] == pytest.approx(ROLL.pnl.sum())


def test_drawdown_is_negative_and_at_least_as_deep_as_the_worst_trade():
    assert ROLL.max_drawdown < 0.0
    # A drawdown contains at least one losing trade, so it cannot be shallower.
    assert ROLL.max_drawdown <= ROLL.worst_trade


def test_the_worst_trades_are_the_crises():
    worst = ROLL.dates[np.argsort(ROLL.pnl)[:3]].astype("datetime64[Y]").astype(int) + 1970
    assert {2008, 2020} & set(worst.tolist()), worst


def test_losses_cluster_in_time():
    """Positive serial correlation: the i.i.d. simulations understate a bad run."""
    assert ROLL.serial_correlation > 0.0
    assert ROLL.worst_streak >= 2
    assert ROLL.longest_drawdown_trades > ROLL.worst_streak


# --------------------------------------------------------------------------------
# Sensitivity: does the edge survive the VIX bias and costs?
# --------------------------------------------------------------------------------


def test_haircutting_the_implied_vol_reduces_the_edge_monotonically():
    means = [monthly_roll(implied_haircut=h).mean_pnl for h in (0.0, 0.005, 0.010, 0.015)]
    assert means == sorted(means, reverse=True), means


def test_costs_reduce_the_edge():
    assert monthly_roll(cost_bps=5.0).mean_pnl < monthly_roll(cost_bps=0.0).mean_pnl


def test_the_edge_survives_a_pessimistic_case():
    """1.5 vol points off VIX plus 5bp of cost -- still profitable."""
    pessimistic = monthly_roll(implied_haircut=0.015, cost_bps=5.0)
    assert pessimistic.mean_pnl > 0.0
    assert pessimistic.sharpe_annualized > 0.75


# --------------------------------------------------------------------------------
# Alpha or beta?
# --------------------------------------------------------------------------------


def test_the_strategy_is_mostly_not_equity_beta():
    """If this were the index in disguise, R-squared would be high. It isn't."""
    reg = regress_on_market(ROLL)
    assert reg.n == len(ROLL)
    assert reg.r_squared < 0.5
    assert abs(reg.beta) < 0.25


def test_alpha_is_positive_after_removing_market_exposure():
    reg = regress_on_market(ROLL)
    assert reg.alpha > 0.0
    assert reg.alpha_annualized > 0.01  # more than 1%/yr


def test_the_beta_is_asymmetric_and_bites_on_the_downside():
    """The short-vol signature: market-neutral on the way up, not on the way down."""
    reg = regress_on_market(ROLL)
    assert reg.beta_down > reg.beta_up
    assert reg.beta_down > 0.0


def test_regression_recovers_a_planted_relationship():
    """Sanity-check the OLS itself on data with a known answer."""
    import dataclasses

    x = np.linspace(-0.1, 0.1, 200)
    synthetic = dataclasses.replace(ROLL, market_return=x, pnl=0.002 + 0.5 * x)
    reg = regress_on_market(synthetic)
    assert reg.alpha == pytest.approx(0.002, abs=1e-9)
    assert reg.beta == pytest.approx(0.5, abs=1e-9)
    assert reg.r_squared == pytest.approx(1.0, abs=1e-9)


def test_hedge_core_rejects_having_neither_rng_nor_returns():
    from vrp import hedged_short_option_pnl

    with pytest.raises(ValueError, match="supply either an rng"):
        hedged_short_option_pnl(
            S0=100.0, K=100.0, r=0.02, sigma_implied=0.2, sigma_realized=0.16,
            T=1 / 12, n_steps=21, rng=None,
        )
