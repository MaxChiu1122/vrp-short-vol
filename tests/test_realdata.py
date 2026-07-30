"""Real market data: the measured premium, and real return shapes in the hedge.

These run entirely off the vendored ``data/spx_vix_daily.csv`` -- no network, so
the suite stays deterministic and offline.
"""

from __future__ import annotations

import numpy as np
import pytest

from vrp import bootstrap_pnl_paths, load_market_data, measure_vrp, summarize_pnl, vrp_pnl_paths
from vrp.bootstrap import (
    bootstrap_log_returns,
    historical_daily_returns,
    historical_kurtosis,
    historical_vol_clustering,
)
from vrp.marketdata import forward_realized_vol, log_returns, realized_vol

# --------------------------------------------------------------------------------
# The data itself
# --------------------------------------------------------------------------------


def test_vendored_data_loads_and_is_sane():
    d = load_market_data()
    assert len(d) > 8_000  # ~36 years of trading days
    assert (d.spx > 0).all()
    assert (d.vix > 0).all()
    assert (np.diff(d.dates) > np.timedelta64(0, "D")).all(), "dates must be strictly increasing"
    assert d.vix.max() < 200.0  # VIX is in points, not decimals


def test_realized_vol_of_the_sp500_is_plausible():
    """Full-sample SPX vol should land in the high teens."""
    vol = realized_vol(log_returns(load_market_data().spx))
    assert 0.12 < vol < 0.25, vol


def test_forward_realized_vol_looks_forward_only():
    """Element i must depend on returns[i:i+w] -- never on anything before i."""
    r = np.zeros(50)
    r[30] = 0.05  # one big move at index 30
    fwd = forward_realized_vol(r, window=10)
    assert fwd[29] > 0  # windows [29:39] and [21:31] contain it
    assert fwd[21] > 0
    assert fwd[20] == 0.0  # window [20:30] stops just short of it
    assert np.isnan(fwd[-5:]).all()  # incomplete futures are NaN, not fabricated


# --------------------------------------------------------------------------------
# The measured premium
# --------------------------------------------------------------------------------


def test_the_variance_risk_premium_is_positive_on_average():
    """The whole thesis, measured rather than assumed."""
    m = measure_vrp()
    assert m.dates.size > 8_000
    assert 2.0 < m.mean_spread_vol_points < 7.0, m.mean_spread_vol_points
    assert m.fraction_positive > 0.75


def test_the_premium_inverts_during_crises():
    """It is not free money: the spread goes sharply negative in a vol spike."""
    m = measure_vrp()
    assert m.spread.min() * 100 < -30.0  # at least one catastrophic month
    worst = m.dates[int(np.argmin(m.spread))]
    assert np.datetime64("2020-01-01") <= worst <= np.datetime64("2020-12-31"), worst


def test_implied_and_realized_are_both_decimals():
    m = measure_vrp()
    assert 0.05 < m.implied.mean() < 0.40
    assert 0.05 < m.realized.mean() < 0.40


# --------------------------------------------------------------------------------
# Return shape: fat tails and clustering
# --------------------------------------------------------------------------------


def test_real_returns_are_fat_tailed_and_clustered():
    """The two facts a Gaussian misses, as numbers rather than assertions."""
    assert historical_kurtosis() > 5.0  # Gaussian is 0
    assert historical_vol_clustering() > 0.15  # IID is 0


def test_bootstrap_matches_the_requested_vol_level():
    """Vol is matched so the comparison against GBM differs in SHAPE only."""
    rng = np.random.default_rng(0)
    dt = 1 / 252
    sample = bootstrap_log_returns(
        historical_daily_returns(), n_paths=4_000, n_steps=21, rng=rng, target_vol=0.16, r=0.02, dt=dt
    )
    realized = np.std(sample) * np.sqrt(1 / dt)
    assert realized == pytest.approx(0.16, rel=0.05)


def test_bootstrap_preserves_fat_tails_after_scaling():
    """Standardizing on POPULATION moments must not flatten the tails."""
    rng = np.random.default_rng(0)
    sample = bootstrap_log_returns(
        historical_daily_returns(),
        n_paths=4_000,
        n_steps=21,
        rng=rng,
        target_vol=0.16,
        r=0.02,
        dt=1 / 252,
    ).ravel()
    z = (sample - sample.mean()) / sample.std(ddof=1)
    assert float(np.mean(z**4) - 3.0) > 3.0, "rescaling should not have Gaussianized the sample"


def test_block_bootstrap_retains_clustering_and_iid_does_not():
    rng = np.random.default_rng(0)
    hist = historical_daily_returns()

    def clustering(block: int) -> float:
        s = bootstrap_log_returns(hist, n_paths=3_000, n_steps=42, rng=rng, block=block)
        a = np.abs(s)
        # Mean lag-1 autocorrelation of |return| within each path.
        centred = a - a.mean(axis=1, keepdims=True)
        num = (centred[:, :-1] * centred[:, 1:]).mean()
        return float(num / a.var())

    assert clustering(21) > clustering(1)


def test_bootstrap_rejects_impossible_blocks():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="block must be >= 1"):
        bootstrap_log_returns(historical_daily_returns(), n_paths=2, n_steps=5, rng=rng, block=0)
    with pytest.raises(ValueError, match="exceeds"):
        bootstrap_log_returns(
            historical_daily_returns(), n_paths=2, n_steps=5, rng=rng, block=10**7
        )


# --------------------------------------------------------------------------------
# Real returns through the hedge
# --------------------------------------------------------------------------------

RUN = dict(sigma_implied=0.20, sigma_realized=0.16, n_paths=8_000, seed=11)


def test_real_return_shapes_fatten_the_tail_at_the_same_vol():
    """The headline: matched vol level, far worse tail."""
    gaussian = summarize_pnl(vrp_pnl_paths(**RUN))
    real = summarize_pnl(bootstrap_pnl_paths(**RUN, block=1))

    assert real.mean_pnl > 0.0, "the premium should still be harvested on average"
    assert real.cvar_05 < 3.0 * gaussian.cvar_05, (
        f"real returns should be far more dangerous: {real.cvar_05:.3f} vs {gaussian.cvar_05:.3f}"
    )
    assert real.sharpe < gaussian.sharpe


def test_clustering_makes_the_tail_worse_still():
    iid = summarize_pnl(bootstrap_pnl_paths(**RUN, block=1))
    clustered = summarize_pnl(bootstrap_pnl_paths(**RUN, block=5))
    assert clustered.cvar_05 < iid.cvar_05


def test_bootstrap_backtest_is_reproducible():
    assert np.array_equal(bootstrap_pnl_paths(**RUN, block=5), bootstrap_pnl_paths(**RUN, block=5))


def test_supplied_returns_must_match_the_step_count():
    from vrp import hedged_short_option_pnl

    with pytest.raises(ValueError, match="expected n_steps"):
        hedged_short_option_pnl(
            S0=100.0,
            K=100.0,
            r=0.02,
            sigma_implied=0.2,
            sigma_realized=0.16,
            T=1 / 12,
            n_steps=21,
            rng=np.random.default_rng(0),
            log_returns=np.zeros(5),
        )
