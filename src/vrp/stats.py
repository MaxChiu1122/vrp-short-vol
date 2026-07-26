"""Distribution statistics for a strategy's per-path P&L.

The value of running the backtest over many Monte-Carlo paths (rather than one
historical path) is that we get the full P&L *distribution* -- so we can report not
just the mean edge but the tail, which is what a short-gamma book actually dies on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VrpResult:
    """Summary of a short-vol backtest's P&L distribution over n_paths scenarios."""

    n_paths: int
    mean_pnl: float  # average edge per trade -- the harvested variance risk premium
    std_pnl: float  # dispersion of the P&L
    sharpe: float  # mean / std of per-trade P&L (dimensionless; annualize externally)
    cvar_05: float  # mean of the worst 5% of P&Ls (the short-gamma tail); loss is negative
    win_rate: float  # fraction of paths that made money

    def __repr__(self) -> str:  # compact, notebook-friendly
        return (
            f"VrpResult(n={self.n_paths}, mean={self.mean_pnl:+.4f}, std={self.std_pnl:.4f}, "
            f"sharpe={self.sharpe:.3f}, CVaR5%={self.cvar_05:+.4f}, win={self.win_rate:.1%})"
        )


def summarize_pnl(pnls) -> VrpResult:
    """Reduce an array of per-path P&Ls to a VrpResult.

    CVaR (expected shortfall) at 5% is the mean of the worst 5% of outcomes -- a
    coherent tail measure, and the right lens for a short-vol book whose losses are
    fat-tailed and left-skewed.
    """
    a = np.asarray(pnls, dtype=float)
    n = a.size
    if n == 0:
        raise ValueError("summarize_pnl: empty P&L array")

    mean = float(a.mean())
    std = float(a.std(ddof=1)) if n > 1 else 0.0
    sharpe = mean / std if std > 0.0 else float("nan")

    k = max(1, int(round(0.05 * n)))  # size of the 5% tail
    cvar_05 = float(np.sort(a)[:k].mean())  # mean of the k worst P&Ls
    win_rate = float((a > 0.0).mean())

    return VrpResult(
        n_paths=n,
        mean_pnl=mean,
        std_pnl=std,
        sharpe=sharpe,
        cvar_05=cvar_05,
        win_rate=win_rate,
    )
