"""Tests for the Monte-Carlo orchestration.

``vrp_pnl_paths`` returns the raw distribution and ``vrp_backtest`` summarizes it;
these pin that the split is lossless and that both halves stay reproducible.
"""

from __future__ import annotations

import numpy as np
import pytest

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


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="backend must be one of"):
        vrp_pnl_paths(**CONTRACT, backend="cuda")


# --------------------------------------------------------------------------------
# The C++ fast path. Same experiment, different engine -- so it must agree in
# DISTRIBUTION (it draws from a different generator, so never elementwise).
# --------------------------------------------------------------------------------


def test_fast_backend_shape_and_finiteness():
    pnls = vrp_pnl_paths(**CONTRACT, backend="fast")
    assert pnls.shape == (500,)
    assert np.isfinite(pnls).all()


def test_fast_backend_is_reproducible():
    assert np.array_equal(
        vrp_pnl_paths(**CONTRACT, backend="fast"), vrp_pnl_paths(**CONTRACT, backend="fast")
    )


def test_fast_backend_is_thread_count_independent():
    """Fixed chunking + per-chunk RNG streams: the array must not depend on threads."""
    one = vrp_pnl_paths(**CONTRACT, backend="fast", n_threads=1)
    many = vrp_pnl_paths(**CONTRACT, backend="fast", n_threads=8)
    assert np.array_equal(one, many)


def test_backends_agree_in_distribution():
    """The two engines must be the same experiment, judged statistically.

    Different generators means different paths, so we compare the sample means
    against their combined Monte-Carlo standard error rather than elementwise.
    """
    kwargs = dict(sigma_implied=0.20, sigma_realized=0.16, n_paths=40_000, seed=3)
    slow = vrp_pnl_paths(**kwargs, backend="python")
    fast = vrp_pnl_paths(**kwargs, backend="fast")

    se = np.sqrt(slow.var(ddof=1) / slow.size + fast.var(ddof=1) / fast.size)
    assert abs(slow.mean() - fast.mean()) < 4.0 * se, (
        f"means disagree: python {slow.mean():+.4f} vs fast {fast.mean():+.4f} "
        f"({abs(slow.mean() - fast.mean()) / se:.1f} s.e. apart)"
    )
    # The tail is the point of the exercise, so check it too.
    assert abs(np.percentile(slow, 5) - np.percentile(fast, 5)) < 0.05
