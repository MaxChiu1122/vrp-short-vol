"""Tests for the Monte-Carlo orchestration.

``vrp_pnl_paths`` returns the raw distribution and ``vrp_backtest`` summarizes it;
these pin that the split is lossless and that both halves stay reproducible.
"""

from __future__ import annotations

import numpy as np

from vrp import summarize_pnl, vrp_backtest, vrp_pnl_paths

CONTRACT = dict(sigma_implied=0.20, sigma_realized=0.16, n_paths=500, seed=5)


def test_pnl_paths_shape_and_dtype():
    pnls = vrp_pnl_paths(**CONTRACT)
    assert pnls.shape == (500,)
    assert pnls.dtype == np.float64
    assert np.isfinite(pnls).all()


def test_backtest_is_exactly_the_summary_of_the_paths():
    """The split must be lossless -- no resampling, no second RNG draw."""
    assert vrp_backtest(**CONTRACT) == summarize_pnl(vrp_pnl_paths(**CONTRACT))


def test_pnl_paths_are_reproducible():
    assert np.array_equal(vrp_pnl_paths(**CONTRACT), vrp_pnl_paths(**CONTRACT))


def test_contract_overrides_reach_the_core():
    """A different strike must change the distribution -- kwargs aren't swallowed."""
    atm = vrp_pnl_paths(**CONTRACT)
    otm = vrp_pnl_paths(**CONTRACT, K=110.0)
    assert not np.array_equal(atm, otm)
