"""Tests for the distribution statistics (independent of the VRP core).

These run green immediately -- they exercise summarize_pnl on synthetic P&L arrays,
so the harness/stats are verified before the hedge core is written.
"""

import numpy as np

from vrp import summarize_pnl


def test_basic_moments():
    pnls = np.array([1.0, -1.0, 2.0, -2.0, 0.5])
    r = summarize_pnl(pnls)
    assert r.n_paths == 5
    assert r.mean_pnl == 0.1
    assert r.win_rate == 3 / 5  # three positive


def test_cvar_is_mean_of_worst_tail():
    # 100 P&Ls: worst 5 are -10..-6; CVaR5% = mean of the 5 worst = -8.
    pnls = list(range(-10, 90))  # -10, -9, ..., 89
    r = summarize_pnl(pnls)
    assert r.cvar_05 == float(np.mean([-10, -9, -8, -7, -6]))


def test_sharpe_sign_follows_mean():
    winners = np.array([1.0, 1.2, 0.8, 1.1, 0.9])
    assert summarize_pnl(winners).sharpe > 0
    losers = -winners
    assert summarize_pnl(losers).sharpe < 0


def test_zero_variance_sharpe_is_nan():
    r = summarize_pnl(np.ones(10))
    assert np.isnan(r.sharpe)
