"""A historical monthly roll: what this strategy would actually have done since 1990.

Everything else in the repo reports the distribution of ONE trade, drawn many times
independently. That is the right way to study a mechanism, but it has no time axis
-- and a systematic strategy lives on one. Losses cluster, drawdowns compound, and
"the worst 5% of paths" says nothing about whether those paths arrive consecutively.

So this rolls the trade forward through actual history: on each roll date sell a
one-month at-the-money call at the VIX quoted that day, delta-hedge daily against
the S&P's actual closes, settle at expiry, repeat. Windows are **non-overlapping**,
so the ~438 trades since 1990 are genuinely independent observations rather than
the 9,000 correlated ones an overlapping window would manufacture.

P&L is reported as a **fraction of notional**, not in index points: the S&P went
from 359 to 7,400 across the sample, so raw P&L is not comparable across decades.
Each trade is run at ``S0 = 100`` on the real return path, making the result
directly a percentage.

Two honest caveats, both stated in the README:

  * **VIX is not an ATM implied vol.** It is a variance-swap-style measure of the
    whole surface, so it prints roughly a vol point above the at-the-money option
    this strategy actually sells. The roll therefore flatters the premium slightly.
    Replacing it needs an option-chain history.
  * **The risk-free rate is held constant.** Rates ran from 8% to 0% over the
    sample; the hedge's drift and cash accrual use one number. Over a one-month
    horizon the effect is small, and it cancels entirely in continuous hedging.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hedge import hedged_short_option_pnl
from .marketdata import MONTH_DAYS, MarketData, load_market_data, log_returns, realized_vol

MONTHS_PER_YEAR = 12.0


@dataclass(frozen=True)
class RollResult:
    """Per-trade record of a historical monthly roll, plus the usual risk metrics.

    All P&L figures are fractions of notional (0.004 = 40 bp of the index level).
    """

    dates: np.ndarray  # roll date of each trade
    spot: np.ndarray  # index level sold against
    implied: np.ndarray  # VIX on the roll date, decimal
    realized: np.ndarray  # what the following month actually realized, decimal
    pnl: np.ndarray  # per-trade P&L, fraction of notional
    market_return: np.ndarray  # S&P log return over the same window

    def __len__(self) -> int:
        return self.pnl.size

    # -- aggregate performance ---------------------------------------------------

    @property
    def equity_curve(self) -> np.ndarray:
        """Cumulative P&L at constant notional -- the sum, not a compounded return.

        Constant notional is the honest choice for a strategy with no defined
        capital base: compounding would silently assume a reinvestment rule that
        this repo never specifies (see the sizing gap in the README).
        """
        return np.cumsum(self.pnl)

    @property
    def mean_pnl(self) -> float:
        return float(self.pnl.mean())

    @property
    def hit_rate(self) -> float:
        return float((self.pnl > 0).mean())

    @property
    def sharpe_annualized(self) -> float:
        """Per-trade Sharpe scaled by sqrt(12) -- trades are monthly and independent."""
        sd = self.pnl.std(ddof=1)
        if sd == 0.0:
            return float("nan")
        return float(self.pnl.mean() / sd * np.sqrt(MONTHS_PER_YEAR))

    @property
    def cvar_05(self) -> float:
        """Mean of the worst 5% of trades -- the same statistic the simulations report."""
        k = max(1, int(round(0.05 * self.pnl.size)))
        return float(np.sort(self.pnl)[:k].mean())

    @property
    def worst_trade(self) -> float:
        return float(self.pnl.min())

    # -- the things only a time axis can show ------------------------------------

    @property
    def max_drawdown(self) -> float:
        """Deepest peak-to-trough fall of the equity curve (negative)."""
        curve = self.equity_curve
        return float((curve - np.maximum.accumulate(curve)).min())

    @property
    def longest_drawdown_trades(self) -> int:
        """Most consecutive trades spent below a previous equity peak."""
        curve = self.equity_curve
        peak = np.maximum.accumulate(curve)
        underwater = curve < peak
        longest = run = 0
        for flag in underwater:
            run = run + 1 if flag else 0
            longest = max(longest, run)
        return int(longest)

    @property
    def worst_streak(self) -> int:
        """Most consecutive losing trades -- do the losses arrive together?"""
        longest = run = 0
        for x in self.pnl:
            run = run + 1 if x < 0 else 0
            longest = max(longest, run)
        return int(longest)

    @property
    def serial_correlation(self) -> float:
        """Lag-1 autocorrelation of trade P&L.

        Independent draws would give ~0. Anything meaningfully positive means the
        i.i.d. picture the simulations paint understates how bad a *run* can get.
        """
        a, b = self.pnl[:-1], self.pnl[1:]
        if a.size < 2:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    def __repr__(self) -> str:
        return (
            f"RollResult(n={len(self)}, mean={self.mean_pnl * 100:+.3f}%, "
            f"ann.Sharpe={self.sharpe_annualized:.2f}, maxDD={self.max_drawdown * 100:.2f}%, "
            f"hit={self.hit_rate:.1%}, worst={self.worst_trade * 100:+.2f}%)"
        )


def monthly_roll(
    data: MarketData | None = None,
    *,
    window: int = MONTH_DAYS,
    r: float = 0.02,
    cost_bps: float = 0.0,
    implied_haircut: float = 0.0,
    notional: float = 100.0,
) -> RollResult:
    """Sell a 1-month ATM call at VIX each period and delta-hedge it to expiry.

    Parameters
    ----------
    window:
        Trading days per trade. Windows are consecutive and non-overlapping.
    r:
        Constant risk-free rate for the deltas and the cash leg (see module docstring).
    cost_bps:
        Proportional hedging cost, in basis points of notional traded.
    implied_haircut:
        Vol points (as a decimal) subtracted from VIX before selling, e.g. ``0.01``
        to sell one point below the VIX print. VIX overstates the ATM implied vol
        this strategy actually sells, so this is the knob for testing how much of
        the result survives that bias.
    """
    if data is None:
        data = load_market_data()

    rets = log_returns(data.spx)
    n_trades = rets.size // window
    if n_trades == 0:
        raise ValueError(f"history is shorter than one {window}-day window")

    T = window / 252.0  # the option's maturity, in years
    dates, spots, implieds, realizeds, pnls, mkt = [], [], [], [], [], []

    for i in range(n_trades):
        start = i * window
        path = rets[start : start + window]

        sigma_implied = data.vix[start] / 100.0 - implied_haircut
        if sigma_implied <= 0.0:
            continue  # a haircut deeper than the quote: no trade to do

        pnl = hedged_short_option_pnl(
            S0=notional,
            K=notional,  # at the money
            r=r,
            sigma_implied=sigma_implied,
            sigma_realized=sigma_implied,  # unused: log_returns drives the path
            T=T,
            n_steps=window,
            rng=None,  # not consumed when log_returns is supplied
            cost_bps=cost_bps,
            log_returns=path,
        )

        dates.append(data.dates[start])
        spots.append(data.spx[start])
        implieds.append(sigma_implied)
        realizeds.append(realized_vol(path))
        pnls.append(pnl / notional)  # as a fraction of notional
        mkt.append(float(path.sum()))  # log return over the window

    return RollResult(
        dates=np.array(dates),
        spot=np.array(spots, dtype=float),
        implied=np.array(implieds, dtype=float),
        realized=np.array(realizeds, dtype=float),
        pnl=np.array(pnls, dtype=float),
        market_return=np.array(mkt, dtype=float),
    )


# --------------------------------------------------------------------------------
# Is the edge alpha, or equity beta wearing a hat?
# --------------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketRegression:
    """OLS of strategy P&L on the market return over the same window.

    ``alpha`` is per trade, in the same units as the P&L (fraction of notional).
    ``beta_down`` / ``beta_up`` split the sample on the sign of the market return:
    a short-vol book is expected to be far more market-sensitive when the market
    falls than when it rises, and a single beta averages that asymmetry away.
    """

    alpha: float
    beta: float
    r_squared: float
    correlation: float
    beta_down: float
    beta_up: float
    n: int

    @property
    def alpha_annualized(self) -> float:
        return self.alpha * MONTHS_PER_YEAR

    def __repr__(self) -> str:
        return (
            f"MarketRegression(alpha={self.alpha * 100:+.3f}%/trade "
            f"({self.alpha_annualized * 100:+.2f}%/yr), beta={self.beta:+.3f}, "
            f"R2={self.r_squared:.3f}, down/up beta={self.beta_down:+.3f}/{self.beta_up:+.3f})"
        )


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least squares fit of ``y = alpha + beta * x``."""
    design = np.column_stack([np.ones_like(x), x])
    alpha, beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(alpha), float(beta)


def regress_on_market(roll: RollResult) -> MarketRegression:
    """Decompose the roll's P&L into market exposure and the residual.

    The question this answers is the one a short-vol pitch always gets: is the
    premium compensation for a risk nobody else wants, or is it equity beta
    repackaged? A high R-squared would mean the latter.
    """
    x, y = roll.market_return, roll.pnl
    alpha, beta = _ols(x, y)
    corr = float(np.corrcoef(x, y)[0, 1])

    down = x < 0
    beta_down = _ols(x[down], y[down])[1] if down.sum() > 2 else float("nan")
    beta_up = _ols(x[~down], y[~down])[1] if (~down).sum() > 2 else float("nan")

    return MarketRegression(
        alpha=alpha,
        beta=beta,
        r_squared=corr**2,
        correlation=corr,
        beta_down=beta_down,
        beta_up=beta_up,
        n=x.size,
    )
