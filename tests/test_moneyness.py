"""Moneyness and skew: where along the strike ladder the edge actually lives."""

from __future__ import annotations

import numpy as np
import pytest

import mcpricer as mc
from vrp import (
    TYPICAL_EQUITY_SKEW,
    implied_vol_at_strike,
    moneyness_sweep,
    strike_for_delta,
)

CONTRACT = dict(S0=100.0, r=0.02, T=1.0 / 12.0)


@pytest.mark.parametrize("target_delta", [0.9, 0.75, 0.5, 0.25, 0.1])
def test_strike_solver_round_trips(target_delta):
    """The solved strike must reprice to the delta we asked for."""
    K = strike_for_delta(target_delta=target_delta, sigma=0.20, **CONTRACT)
    back = mc.black_scholes_delta(mc.OptionType.Call, 100.0, K, 0.02, 0.20, 1.0 / 12.0)
    assert back == pytest.approx(target_delta, abs=1e-9)


def test_lower_delta_means_higher_strike():
    strikes = [
        strike_for_delta(target_delta=d, sigma=0.20, **CONTRACT) for d in (0.75, 0.5, 0.25, 0.1)
    ]
    assert strikes == sorted(strikes), f"delta down should mean strike up: {strikes}"


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_strike_solver_rejects_impossible_deltas(bad):
    with pytest.raises(ValueError, match="target_delta must be in"):
        strike_for_delta(target_delta=bad, sigma=0.20, **CONTRACT)


def test_flat_surface_quotes_one_vol_everywhere():
    for K in (90.0, 100.0, 110.0):
        assert implied_vol_at_strike(K=K, S0=100.0, sigma_atm=0.20, skew_slope=0.0) == 0.20


def test_equity_skew_prices_upside_calls_cheaper():
    """Negative slope: above the money the surface quotes a lower vol."""
    low = implied_vol_at_strike(K=95.0, S0=100.0, sigma_atm=0.20, skew_slope=TYPICAL_EQUITY_SKEW)
    atm = implied_vol_at_strike(K=100.0, S0=100.0, sigma_atm=0.20, skew_slope=TYPICAL_EQUITY_SKEW)
    high = implied_vol_at_strike(K=110.0, S0=100.0, sigma_atm=0.20, skew_slope=TYPICAL_EQUITY_SKEW)
    assert low > atm > high
    assert atm == pytest.approx(0.20)


# --------------------------------------------------------------------------------
# The sweep
# --------------------------------------------------------------------------------

DELTAS = (0.75, 0.50, 0.30, 0.10)
SWEEP = dict(sigma_atm=0.20, sigma_realized=0.16, n_paths=6_000, seed=11)


def test_flat_surface_edge_peaks_near_the_money():
    """Dollar gamma and vega both peak ATM, so the harvested spread must too."""
    sweep = moneyness_sweep(target_deltas=DELTAS, skew_slope=0.0, **SWEEP)
    means = [r.mean_pnl for _d, _K, _s, r in sweep]
    best = int(np.argmax(means))
    assert 0 < best < len(means) - 1, f"expected an interior (near-ATM) peak: {means}"


def test_far_otm_earns_much_less_than_atm():
    sweep = moneyness_sweep(target_deltas=(0.50, 0.10), skew_slope=0.0, **SWEEP)
    atm, otm = (r.mean_pnl for _d, _K, _s, r in sweep)
    assert otm < 0.5 * atm, f"10-delta should harvest far less than ATM: {otm:.4f} vs {atm:.4f}"


def test_skew_takes_the_edge_out_of_upside_calls():
    """The headline: selling OTM calls on a skewed surface collects a smaller spread."""
    flat = moneyness_sweep(target_deltas=DELTAS, skew_slope=0.0, **SWEEP)
    skewed = moneyness_sweep(target_deltas=DELTAS, skew_slope=TYPICAL_EQUITY_SKEW, **SWEEP)

    # At the money the two surfaces agree, so the results should be close.
    atm_flat = next(r for d, _K, _s, r in flat if d == 0.50)
    atm_skew = next(r for d, _K, _s, r in skewed if d == 0.50)
    assert atm_skew.mean_pnl == pytest.approx(atm_flat.mean_pnl, rel=0.10)

    # Out of the money the skew has taken the premium away.
    otm_flat = next(r for d, _K, _s, r in flat if d == 0.10)
    otm_skew = next(r for d, _K, _s, r in skewed if d == 0.10)
    assert otm_skew.mean_pnl < 0.5 * otm_flat.mean_pnl


def test_skew_moves_the_best_risk_adjusted_strike_in_the_money():
    """With equity skew the Sharpe-maximizing strike sits below the flat-surface one."""
    flat = moneyness_sweep(target_deltas=DELTAS, skew_slope=0.0, **SWEEP)
    skewed = moneyness_sweep(target_deltas=DELTAS, skew_slope=TYPICAL_EQUITY_SKEW, **SWEEP)

    best_flat = max(flat, key=lambda row: row[3].sharpe)[0]
    best_skew = max(skewed, key=lambda row: row[3].sharpe)[0]
    assert best_skew >= best_flat, (
        f"skew should push the best strike in-the-money (higher delta): "
        f"flat {best_flat}, skewed {best_skew}"
    )


def test_sweep_reports_the_strike_and_vol_it_used():
    sweep = moneyness_sweep(target_deltas=(0.25,), skew_slope=TYPICAL_EQUITY_SKEW, **SWEEP)
    delta, K, sigma_implied, _res = sweep[0]
    assert delta == 0.25
    assert K > 100.0  # a 25-delta call is out of the money
    assert sigma_implied < 0.20  # and the skew quotes it below the ATM vol
