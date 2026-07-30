"""Real S&P 500 and VIX history, and the variance risk premium *measured* from it.

Everything else in this repo takes ``sigma_realized`` as an input, which is the
right way to study the mechanics but means the premium is assumed rather than
shown. This module closes that gap: VIX is the market's 30-day implied vol, and
the realized vol of the *following* month is what actually happened, so their
difference is the premium, observed rather than posited.

The data is vendored in ``data/spx_vix_daily.csv`` (see
``scripts/fetch_market_data.py``) so results are reproducible and the tests never
touch the network. Loading is numpy-only -- pandas is a fetch-script dependency,
not a package one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent.parent / "data" / "spx_vix_daily.csv"

TRADING_DAYS = 252  # per year; the annualization convention used throughout
MONTH_DAYS = 21  # trading days in a month -- VIX's 30 calendar days


@dataclass(frozen=True)
class MarketData:
    """Daily closes. ``vix`` is in vol points (17.24 means 17.24% annualized)."""

    dates: np.ndarray  # datetime64[D]
    spx: np.ndarray
    vix: np.ndarray

    def __len__(self) -> int:
        return self.dates.size

    def __repr__(self) -> str:
        return (
            f"MarketData({len(self)} days, {self.dates[0]} -> {self.dates[-1]}, "
            f"VIX {self.vix.min():.1f}-{self.vix.max():.1f})"
        )


def load_market_data(path: Path | str = DATA) -> MarketData:
    """Load the vendored daily SPX/VIX history."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python scripts/fetch_market_data.py` to vendor it"
        )
    raw = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return MarketData(
        dates=raw["date"].astype("datetime64[D]"),
        spx=raw["spx"].astype(float),
        vix=raw["vix"].astype(float),
    )


def log_returns(prices: np.ndarray) -> np.ndarray:
    """Close-to-close log returns; length ``len(prices) - 1``."""
    return np.diff(np.log(np.asarray(prices, dtype=float)))


def realized_vol(returns: np.ndarray, *, annualize: bool = True) -> float:
    """Annualized close-to-close realized volatility of a return sample.

    Uses the *uncentred* second moment, the market convention for realized
    variance: over a month the mean return is negligible next to its dispersion,
    and subtracting an estimated mean adds noise without removing bias.
    """
    r = np.asarray(returns, dtype=float)
    vol = np.sqrt(np.mean(r**2))
    return float(vol * np.sqrt(TRADING_DAYS)) if annualize else float(vol)


def forward_realized_vol(returns: np.ndarray, *, window: int = MONTH_DAYS) -> np.ndarray:
    """Realized vol over the ``window`` days FOLLOWING each point.

    Element ``i`` looks at ``returns[i : i + window]`` -- forward-looking on
    purpose, since that is the vol the option seller at time ``i`` is exposed to.
    The last ``window`` entries are NaN: their future has not happened yet. Nothing
    here peeks backwards into information the seller would not have had.
    """
    r = np.asarray(returns, dtype=float)
    out = np.full(r.size, np.nan)
    if r.size < window:
        return out
    # Rolling mean of squared returns via a cumulative sum.
    sq = r**2
    csum = np.concatenate([[0.0], np.cumsum(sq)])
    windowed = (csum[window:] - csum[:-window]) / window
    out[: windowed.size] = np.sqrt(windowed) * np.sqrt(TRADING_DAYS)
    return out


@dataclass(frozen=True)
class VrpMeasurement:
    """The measured variance risk premium: implied now vs realized next month."""

    dates: np.ndarray
    implied: np.ndarray  # VIX on the day, as a decimal (0.1724)
    realized: np.ndarray  # realized vol over the following month, decimal
    spread: np.ndarray  # implied - realized, in decimal vol

    @property
    def mean_spread_vol_points(self) -> float:
        return float(np.mean(self.spread) * 100.0)

    @property
    def median_spread_vol_points(self) -> float:
        return float(np.median(self.spread) * 100.0)

    @property
    def fraction_positive(self) -> float:
        return float(np.mean(self.spread > 0.0))

    def __repr__(self) -> str:
        return (
            f"VrpMeasurement(n={self.dates.size}, mean={self.mean_spread_vol_points:+.2f}vp, "
            f"median={self.median_spread_vol_points:+.2f}vp, "
            f"positive={self.fraction_positive:.1%})"
        )


def measure_vrp(data: MarketData | None = None, *, window: int = MONTH_DAYS) -> VrpMeasurement:
    """Measure implied-minus-subsequent-realized vol, day by day.

    VIX(t) is a 30-calendar-day implied vol, so it is compared against the realized
    vol of the following ``window`` trading days -- the same horizon, forward from
    the same date. A positive average is the variance risk premium; the episodes
    where it goes sharply negative are the ones that end short-vol funds.
    """
    if data is None:
        data = load_market_data()

    rets = log_returns(data.spx)
    fwd = forward_realized_vol(rets, window=window)

    # rets[i] spans dates[i] -> dates[i+1], so the seller quoting on dates[i] sees
    # VIX[i] and then experiences fwd[i]. Align on that, and drop the tail whose
    # future is incomplete.
    implied = data.vix[:-1] / 100.0
    dates = data.dates[:-1]
    ok = ~np.isnan(fwd)
    return VrpMeasurement(
        dates=dates[ok], implied=implied[ok], realized=fwd[ok], spread=implied[ok] - fwd[ok]
    )
