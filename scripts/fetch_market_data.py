"""Download the daily S&P 500 and VIX history and vendor it into ``data/``.

Usage:  python scripts/fetch_market_data.py     (needs: pip install yfinance)

Run this by hand, rarely. The CSV it writes is **committed to the repo** so that
every result stays reproducible offline and the test suite never touches the
network -- a backtest whose inputs change under you is not a backtest.

Why these two series:
  * **^GSPC** gives realized volatility, computed from actual close-to-close returns
    rather than assumed.
  * **^VIX** is the market's 30-day implied volatility on the same index, so
    VIX(t) vs the realized vol over the *following* month is a direct, observable
    measurement of the variance risk premium this repo is built on.

VIX history begins 1990, which sets the start date.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "spx_vix_daily.csv"
START = "1990-01-01"


def main() -> None:
    import yfinance as yf

    raw = yf.download(["^GSPC", "^VIX"], start=START, auto_adjust=False, progress=False)
    close = raw["Close"].rename(columns={"^GSPC": "spx", "^VIX": "vix"})

    before = len(close)
    close = close.dropna()
    print(f"downloaded {before} rows, {before - len(close)} dropped for missing values")

    close.index.name = "date"
    OUT.parent.mkdir(exist_ok=True)
    close.round(4).to_csv(OUT, float_format="%.4f")

    print(f"wrote {OUT.relative_to(OUT.parent.parent)}: {len(close)} rows")
    print(f"  {close.index[0].date()} -> {close.index[-1].date()}")
    print(f"  size {OUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
