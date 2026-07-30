"""Drive the backtest with *real* return shapes instead of Gaussian draws.

The GBM backtest assumes returns are independent and normal. They are neither:
equity returns are fat-tailed, and their volatility clusters -- quiet weeks follow
quiet weeks, and violent days arrive in clumps. Both properties matter enormously
to a short-gamma book, and neither exists in a Gaussian.

This resamples the actual S&P 500 daily return history:

  * ``block=1`` -- **IID bootstrap.** Keeps the fat tails, destroys the clustering
    (each day is drawn independently).
  * ``block=k`` -- **moving-block bootstrap.** Draws runs of ``k`` consecutive real
    days, so volatility clustering survives inside each block.

Comparing GBM -> IID -> block isolates the two effects one at a time, in the same
spirit as the Heston stress: hold the vol *level* fixed and vary only the shape.

That vol matching is what makes the comparison honest. Sampled returns are
standardized against the **historical population** moments and rescaled to the
target vol -- never per path, which would force every path to the same realized
vol and destroy exactly the tail behaviour being studied.
"""

from __future__ import annotations

import numpy as np

from .marketdata import TRADING_DAYS, load_market_data, log_returns


def historical_daily_returns(*, since: str | None = None) -> np.ndarray:
    """S&P 500 daily log returns from the vendored history.

    ``since`` optionally trims the start (e.g. ``"2000-01-01"``), which is how you
    ask whether a result depends on including the 1990s.
    """
    data = load_market_data()
    spx, dates = data.spx, data.dates
    if since is not None:
        keep = dates >= np.datetime64(since)
        spx = spx[keep]
    return log_returns(spx)


def bootstrap_log_returns(
    historical: np.ndarray,
    *,
    n_paths: int,
    n_steps: int,
    rng,
    block: int = 1,
    target_vol: float | None = None,
    r: float = 0.0,
    dt: float | None = None,
) -> np.ndarray:
    """Resample ``historical`` into an ``(n_paths, n_steps)`` array of log returns.

    Parameters
    ----------
    block:
        Length of each contiguous run drawn from the history. ``1`` is an IID
        bootstrap (fat tails, no clustering); larger blocks retain volatility
        clustering within the run.
    target_vol, r, dt:
        If ``target_vol`` is given, the sample is standardized on the historical
        population moments and rescaled so its *expected* annualized vol equals
        ``target_vol``, with the per-step mean set to the risk-neutral drift
        ``(r - target_vol**2 / 2) * dt``. Re-centring on ``r`` matters: it is what
        makes the comparison against the GBM baseline differ in shape only, and it
        keeps the zero-spread mean-P&L identity intact. Without it the historical
        equity drift leaks into the hedge result.
    """
    hist = np.asarray(historical, dtype=float)
    if block < 1:
        raise ValueError(f"block must be >= 1, got {block}")
    if block > hist.size:
        raise ValueError(f"block {block} exceeds the {hist.size}-day history")

    n_blocks = -(-n_steps // block)  # ceiling division
    starts = rng.integers(0, hist.size - block + 1, size=(n_paths, n_blocks))
    # starts[..., None] + arange(block) indexes each block's consecutive days.
    idx = starts[:, :, None] + np.arange(block)
    sample = hist[idx].reshape(n_paths, n_blocks * block)[:, :n_steps]

    if target_vol is None:
        return sample
    if dt is None:
        raise ValueError("dt is required when target_vol is given")

    # Standardize on POPULATION moments -- never per path.
    hist_mean = hist.mean()
    hist_std = hist.std(ddof=1)
    if hist_std == 0.0:
        raise ValueError("historical returns have zero dispersion")

    step_std = target_vol * np.sqrt(dt)
    step_mean = (r - 0.5 * target_vol**2) * dt
    return step_mean + step_std * (sample - hist_mean) / hist_std


def annualized_vol_of(returns: np.ndarray, *, dt: float) -> float:
    """Annualized vol implied by a sample of per-step log returns."""
    steps_per_year = 1.0 / dt
    return float(np.std(np.asarray(returns, dtype=float)) * np.sqrt(steps_per_year))


def historical_kurtosis(*, since: str | None = None) -> float:
    """Excess kurtosis of the daily return history (0 for a Gaussian).

    Reported so the fat-tail claim is a number rather than an assertion.
    """
    r = historical_daily_returns(since=since)
    z = (r - r.mean()) / r.std(ddof=1)
    return float(np.mean(z**4) - 3.0)


def historical_vol_clustering(*, since: str | None = None, lag: int = 1) -> float:
    """Autocorrelation of |daily return| at ``lag`` -- the clustering signature.

    Returns themselves are near-uncorrelated; their *magnitudes* are strongly
    autocorrelated, which is what a block bootstrap preserves and an IID one does
    not. ``TRADING_DAYS`` is unused here; the statistic is scale-free.
    """
    r = np.abs(historical_daily_returns(since=since))
    a, b = r[:-lag], r[lag:]
    return float(np.corrcoef(a, b)[0, 1])


__all__ = [
    "historical_daily_returns",
    "bootstrap_log_returns",
    "annualized_vol_of",
    "historical_kurtosis",
    "historical_vol_clustering",
    "TRADING_DAYS",
]
