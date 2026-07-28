"""Tests for the Heston-realized stress.

The claim being pinned: holding E[variance] fixed and turning on vol-of-vol leaves
the mean edge broadly intact while the left tail degrades badly. That asymmetry is
the reason this repo reports CVaR next to the mean rather than the mean alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from vrp import HestonParams, heston_pnl_paths, summarize_pnl, vrp_pnl_paths

SIGMA_IMPLIED = 0.20
SIGMA_REALIZED = 0.16
N_PATHS = 40_000
SEED = 11


def test_matched_params_preserve_expected_variance():
    h = HestonParams.matched_to(SIGMA_REALIZED, xi=0.5)
    assert h.v0 == pytest.approx(SIGMA_REALIZED**2)
    assert h.theta == pytest.approx(SIGMA_REALIZED**2)
    assert h.long_run_vol == pytest.approx(SIGMA_REALIZED)


def test_feller_flag_matches_its_definition():
    calm = HestonParams.matched_to(SIGMA_REALIZED, kappa=2.0, xi=0.1)
    rough = HestonParams.matched_to(SIGMA_REALIZED, kappa=2.0, xi=0.5)
    assert calm.feller_satisfied  # 2*2*0.0256 = 0.1024 >= 0.01
    assert not rough.feller_satisfied  # 0.1024 < 0.25 -- the regime full truncation handles


def test_zero_vol_of_vol_reproduces_the_gbm_backtest():
    """xi=0 with v0=theta freezes the variance, so Heston collapses to GBM.

    A distributional match, not elementwise: the Heston step consumes two normals
    per step, so the two runs walk different points of the stream.
    """
    gbm = vrp_pnl_paths(
        sigma_implied=SIGMA_IMPLIED,
        sigma_realized=SIGMA_REALIZED,
        n_paths=N_PATHS,
        seed=SEED,
        backend="fast",
    )
    frozen = heston_pnl_paths(
        sigma_implied=SIGMA_IMPLIED,
        heston=HestonParams.matched_to(SIGMA_REALIZED, xi=0.0),
        n_paths=N_PATHS,
        seed=SEED,
    )
    se = np.sqrt(gbm.var(ddof=1) / gbm.size + frozen.var(ddof=1) / frozen.size)
    assert abs(gbm.mean() - frozen.mean()) < 4.0 * se


def test_vol_of_vol_fattens_the_tail_while_the_mean_survives():
    """The headline stress result."""

    def run(xi: float):
        return summarize_pnl(
            heston_pnl_paths(
                sigma_implied=SIGMA_IMPLIED,
                heston=HestonParams.matched_to(SIGMA_REALIZED, xi=xi),
                n_paths=N_PATHS,
                seed=SEED,
            )
        )

    calm = run(0.0)
    rough = run(0.8)

    assert rough.mean_pnl > 0.0, "the strategy should still be profitable on average"
    assert rough.mean_pnl > 0.8 * calm.mean_pnl, "the mean should not collapse"
    assert rough.cvar_05 < 2.0 * calm.cvar_05, "the tail should be far worse (both negative)"
    assert rough.sharpe < calm.sharpe, "risk-adjusted return must degrade"


def test_stress_is_reproducible_and_thread_count_independent():
    kwargs = dict(
        sigma_implied=SIGMA_IMPLIED,
        heston=HestonParams.matched_to(SIGMA_REALIZED, xi=0.5),
        n_paths=2_000,
        seed=SEED,
    )
    assert np.array_equal(heston_pnl_paths(**kwargs), heston_pnl_paths(**kwargs))
    assert np.array_equal(
        heston_pnl_paths(**kwargs, n_threads=1), heston_pnl_paths(**kwargs, n_threads=8)
    )
