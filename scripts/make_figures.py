"""Regenerate the README figures from a fresh backtest.

Usage:  python scripts/make_figures.py

Writes light and dark variants of both figures into ``docs/`` (where .gitignore
un-ignores PNGs). Every number in them comes from a seeded run, so re-running
reproduces the images exactly.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt  # noqa: E402

from vrp import DEFAULT_CONTRACT, summarize_pnl, vrp_curve, vrp_pnl_paths  # noqa: E402
from vrp.plot import THEMES, plot_pnl_distribution, plot_vrp_curve  # noqa: E402

# The realistic contract: 1-month ATM call, 21 daily hedges, sold 4 vol points
# above what the world goes on to realize.
SIGMA_REALIZED = 0.16
SIGMA_IMPLIED = 0.20
N_PATHS = 50_000
SEED = 11

FIGURES = Path(__file__).resolve().parent.parent / "docs"


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]

    print(f"contract: {DEFAULT_CONTRACT}")
    print(f"selling {SIGMA_IMPLIED:.0%} vol into {SIGMA_REALIZED:.0%} realized, {N_PATHS:,} paths")

    pnls = vrp_pnl_paths(
        sigma_implied=SIGMA_IMPLIED,
        sigma_realized=SIGMA_REALIZED,
        n_paths=N_PATHS,
        seed=SEED,
    )
    print(f"  {summarize_pnl(pnls)}")

    curve = vrp_curve(
        spreads=[-0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        sigma_realized=SIGMA_REALIZED,
        n_paths=20_000,
        seed=SEED,
    )
    print("  spread(vol pts)  mean P&L   Sharpe")
    for spread, mean_pnl, sharpe in curve:
        print(f"  {spread * 100:+6.1f}        {mean_pnl:+8.4f}  {sharpe:+7.3f}")

    for theme in ("light", "dark"):
        surface = THEMES[theme]["surface"]

        ax = plot_pnl_distribution(
            pnls,
            title=f"Short-vol P&L: sell {SIGMA_IMPLIED:.0%} vol, realize {SIGMA_REALIZED:.0%}",
            theme=theme,
        )
        out = FIGURES / f"pnl_distribution_{theme}.png"
        ax.figure.savefig(out, dpi=200, bbox_inches="tight", facecolor=surface)
        plt.close(ax.figure)
        print(f"  wrote {out.relative_to(FIGURES.parent)}")

        axes = plot_vrp_curve(curve, theme=theme)
        out = FIGURES / f"vrp_curve_{theme}.png"
        axes[0].figure.savefig(out, dpi=200, bbox_inches="tight", facecolor=surface)
        plt.close(axes[0].figure)
        print(f"  wrote {out.relative_to(FIGURES.parent)}")


if __name__ == "__main__":
    main()
