"""Plot helpers (optional; needs the ``plot`` extra: pip install -e ".[plot]").

Kept separate so the core package has no hard matplotlib dependency.
"""

from __future__ import annotations

import numpy as np


def plot_pnl_distribution(pnls, *, title="Short-vol P&L distribution", ax=None):
    """Histogram of per-path P&L, with the mean and 5% CVaR marked."""
    import matplotlib.pyplot as plt

    a = np.asarray(pnls, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.hist(a, bins=80, color="#4C78A8", alpha=0.85)
    mean = a.mean()
    k = max(1, int(round(0.05 * a.size)))
    cvar = np.sort(a)[:k].mean()
    ax.axvline(mean, color="#111", lw=1.5, label=f"mean {mean:+.3f}")
    ax.axvline(cvar, color="#E4572E", lw=1.5, ls="--", label=f"CVaR 5% {cvar:+.3f}")
    ax.axvline(0.0, color="#888", lw=0.8)
    ax.set_xlabel("P&L per trade")
    ax.set_ylabel("paths")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_vrp_curve(curve, *, ax=None):
    """Mean P&L vs implied-minus-realized vol spread (output of vrp.vrp_curve)."""
    import matplotlib.pyplot as plt

    spreads = [c[0] for c in curve]
    means = [c[1] for c in curve]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.plot([s * 100 for s in spreads], means, "-o", color="#4C78A8")
    ax.axhline(0.0, color="#888", lw=0.8)
    ax.set_xlabel("implied − realized vol (vol points)")
    ax.set_ylabel("mean P&L per trade")
    ax.set_title("Variance risk premium vs vol spread")
    return ax
