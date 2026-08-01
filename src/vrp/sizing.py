"""Position sizing: turning a risk metric into a decision.

This repo computes CVaR everywhere and then never uses it. That is the gap the
historical roll exposed: the strategy earns ~4%/yr on full notional, which nobody
runs unlevered, and the moment you lever it the tail stops being a statistic and
starts being the thing that ends the fund. Short-vol blow-ups -- LJM, the 2018 XIV
unwind -- were sizing failures, not signal failures.

So: given capital and a risk budget, how much notional do you hold?

**Everything here is compounded on capital**, not summed at constant notional like
``vrp.roll``. That difference is the point. Summed P&L cannot go bankrupt; compounded
equity can, and a levered short-vol book is precisely the thing that does.

**Every rule uses trailing information only.** A leverage calibrated on the whole
sample is look-ahead -- it knows how bad 2008 was going to be before it happens --
and ``compare_lookahead`` measures exactly how much that hindsight is worth, so the
honest number is never quietly replaced by the flattering one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .roll import MONTHS_PER_YEAR, RollResult


@dataclass(frozen=True)
class SizedResult:
    """A levered, compounded track record.

    ``returns`` are on *capital* (leverage x the per-notional P&L), and ``equity``
    starts at 1.0. If equity ever reaches zero the book is ruined and everything
    after is zero -- there is no borrowing back from bankruptcy.
    """

    dates: np.ndarray
    leverage: np.ndarray
    returns: np.ndarray
    equity: np.ndarray

    @property
    def ruined(self) -> bool:
        return bool(self.equity[-1] <= 0.0)

    @property
    def terminal_multiple(self) -> float:
        return float(self.equity[-1])

    @property
    def cagr(self) -> float:
        """Compound annual growth rate. Negative-infinite if the book was ruined."""
        if self.ruined:
            return float("-inf")
        years = self.returns.size / MONTHS_PER_YEAR
        return float(self.equity[-1] ** (1.0 / years) - 1.0)

    @property
    def vol_annualized(self) -> float:
        return float(np.std(self.returns, ddof=1) * np.sqrt(MONTHS_PER_YEAR))

    @property
    def sharpe_annualized(self) -> float:
        sd = np.std(self.returns, ddof=1)
        if sd == 0.0:
            return float("nan")
        return float(np.mean(self.returns) / sd * np.sqrt(MONTHS_PER_YEAR))

    @property
    def max_drawdown(self) -> float:
        """Deepest proportional fall in equity (negative), e.g. -0.35 for -35%."""
        peak = np.maximum.accumulate(self.equity)
        with np.errstate(divide="ignore", invalid="ignore"):
            dd = np.where(peak > 0, self.equity / peak - 1.0, -1.0)
        return float(np.min(dd))

    @property
    def worst_month(self) -> float:
        return float(self.returns.min())

    def __repr__(self) -> str:
        if self.ruined:
            return f"SizedResult(RUINED after {int(np.argmax(self.equity <= 0)) + 1} trades)"
        return (
            f"SizedResult(CAGR={self.cagr * 100:+.2f}%, vol={self.vol_annualized * 100:.1f}%, "
            f"Sharpe={self.sharpe_annualized:.2f}, maxDD={self.max_drawdown * 100:.1f}%, "
            f"x{self.terminal_multiple:.1f})"
        )


def apply_leverage(roll: RollResult, leverage) -> SizedResult:
    """Compound ``roll``'s per-notional P&L at the given (possibly varying) leverage."""
    lev = np.broadcast_to(np.asarray(leverage, dtype=float), roll.pnl.shape).copy()
    returns = lev * roll.pnl

    equity = np.empty(returns.size)
    value = 1.0
    for i, ret in enumerate(returns):
        value = value * (1.0 + ret) if value > 0.0 else 0.0
        equity[i] = max(value, 0.0)
        value = equity[i]
    return SizedResult(dates=roll.dates, leverage=lev, returns=returns, equity=equity)


# --------------------------------------------------------------------------------
# Sizing rules. Each returns a leverage series; all are trailing-only.
# --------------------------------------------------------------------------------


def fixed_leverage(roll: RollResult, level: float) -> np.ndarray:
    """Constant leverage -- the naive baseline."""
    return np.full(roll.pnl.size, float(level))


def vol_target_leverage(
    roll: RollResult,
    *,
    target_vol: float = 0.10,
    lookback: int = 24,
    warmup: float = 1.0,
    cap: float = 20.0,
) -> np.ndarray:
    """Scale so the *trailing* realized vol of strategy returns hits ``target_vol``.

    Uses only trades strictly before the one being sized. The first ``lookback``
    trades have no history and take ``warmup``; ``cap`` prevents the rule from
    demanding absurd leverage after an unusually quiet stretch.
    """
    pnl = roll.pnl
    lev = np.full(pnl.size, float(warmup))
    for i in range(lookback, pnl.size):
        trailing = np.std(pnl[i - lookback : i], ddof=1) * np.sqrt(MONTHS_PER_YEAR)
        lev[i] = min(target_vol / trailing, cap) if trailing > 0 else warmup
    return lev


def cvar_budget_leverage(
    roll: RollResult,
    *,
    budget: float = 0.10,
    lookback: int = 60,
    alpha: float = 0.05,
    warmup: float = 1.0,
    cap: float = 20.0,
) -> np.ndarray:
    """Scale so the *trailing* expected shortfall equals ``budget`` of capital.

    The rule this repo has been building toward: size so that the average of the
    worst ``alpha`` of recent trades costs a fixed fraction of capital. Unlike a
    vol target it responds to the shape of the left tail rather than to dispersion,
    which is the distinction that matters for a short-gamma book.
    """
    pnl = roll.pnl
    lev = np.full(pnl.size, float(warmup))
    k = max(1, int(round(alpha * lookback)))
    for i in range(lookback, pnl.size):
        window = np.sort(pnl[i - lookback : i])[:k]
        shortfall = -float(window.mean())
        lev[i] = min(budget / shortfall, cap) if shortfall > 0 else cap
    return lev


def inverse_implied_leverage(
    roll: RollResult, *, reference_vol: float = 0.20, cap: float = 20.0
) -> np.ndarray:
    """Scale inversely with the implied vol quoted at trade time.

    Legitimate rather than look-ahead: VIX on the roll date is known when the trade
    is put on. The intuition is that risk scales with vol, so constant risk means
    less notional when vol is high -- and it needs no warm-up period at all.
    """
    return np.minimum(reference_vol / roll.implied, cap)


# --------------------------------------------------------------------------------
# How much leverage is too much?
# --------------------------------------------------------------------------------


def leverage_sweep(roll: RollResult, levels) -> list[tuple[float, SizedResult]]:
    """Fixed leverage across a grid -- where growth peaks, and where ruin begins."""
    return [(float(x), apply_leverage(roll, fixed_leverage(roll, x))) for x in levels]


def growth_optimal_leverage(roll: RollResult, *, cap: float = 50.0, points: int = 2000) -> float:
    """The leverage maximizing expected log growth (Kelly), on the FULL sample.

    In-sample by construction, so it is a diagnostic, not a rule you could have
    followed. It is worth computing precisely because the answer is usually absurd:
    full Kelly on a strategy with a thin, fat left tail sits at a leverage whose
    drawdowns no investor would tolerate, which is the standard argument for
    running a small fraction of it.
    """
    grid = np.linspace(0.01, cap, points)
    best_growth, best = -np.inf, 0.0
    for lev in grid:
        r = 1.0 + lev * roll.pnl
        if np.any(r <= 0):
            break  # beyond here a single trade wipes the book out
        growth = float(np.mean(np.log(r)))
        if growth > best_growth:
            best_growth, best = growth, float(lev)
    return best


def compare_lookahead(roll: RollResult, *, target_vol: float = 0.10, lookback: int = 24) -> dict:
    """What is hindsight worth? Full-sample calibration vs the trailing rule.

    Sizing off the whole sample's volatility is look-ahead: in 1990 it already knows
    what 2008 held. This runs both and returns them side by side so the cheat is
    quantified rather than silently enjoyed.
    """
    full_sample_vol = float(np.std(roll.pnl, ddof=1) * np.sqrt(MONTHS_PER_YEAR))
    cheating = apply_leverage(roll, fixed_leverage(roll, target_vol / full_sample_vol))
    honest = apply_leverage(
        roll, vol_target_leverage(roll, target_vol=target_vol, lookback=lookback)
    )
    return {"lookahead": cheating, "trailing": honest}
