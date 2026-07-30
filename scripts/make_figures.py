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

from vrp import (  # noqa: E402
    DEFAULT_CONTRACT,
    HestonParams,
    hedge_frequency_sweep,
    heston_pnl_paths,
    summarize_pnl,
    vrp_curve,
    vrp_pnl_paths,
)
from vrp.plot import (  # noqa: E402
    THEMES,
    plot_frequency_sweep,
    plot_pnl_distribution,
    plot_stress_curve,
    plot_vrp_curve,
)

# Rebalancing-frequency sweep. T = 1/12 year is ~21 trading days, so n_steps=21 is
# once a day; the grid spans roughly twice-weekly to ~24x a day.
STEPS_PER_DAY = 21.0
N_STEPS_GRID = (4, 7, 11, 21, 42, 63, 126, 252, 504)
COST_BPS_GRID = (0.0, 1.0, 5.0, 10.0)

# Vol-of-vol sweep for the Heston stress; expected variance is held fixed at
# SIGMA_REALIZED**2 throughout, so only the violence of the vol moves.
XI_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8)

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

    # The Heston stress: same expected variance, rising vol-of-vol.
    stress = []
    print("\n  vol-of-vol   mean P&L    CVaR 5%    Sharpe   win     Feller")
    for xi in XI_GRID:
        heston = HestonParams.matched_to(SIGMA_REALIZED, xi=xi)
        pnls = heston_pnl_paths(
            sigma_implied=SIGMA_IMPLIED, heston=heston, n_paths=N_PATHS, seed=SEED
        )
        res = summarize_pnl(pnls)
        stress.append((xi, res.mean_pnl, res.cvar_05))
        print(
            f"  {xi:>9.1f}   {res.mean_pnl:+8.4f}   {res.cvar_05:+8.4f}   "
            f"{res.sharpe:6.3f}   {res.win_rate:5.1%}   "
            f"{'ok' if heston.feller_satisfied else 'VIOLATED'}"
        )

    # Rebalancing frequency vs transaction cost.
    print("\n  rebal/day   " + "".join(f"{c:g}bp Sharpe   " for c in COST_BPS_GRID))
    sweep = hedge_frequency_sweep(
        n_steps_grid=N_STEPS_GRID,
        cost_bps_grid=COST_BPS_GRID,
        sigma_implied=SIGMA_IMPLIED,
        sigma_realized=SIGMA_REALIZED,
        n_paths=20_000,
        seed=SEED,
    )
    for n_steps in N_STEPS_GRID:
        row = f"  {n_steps / STEPS_PER_DAY:>9.2f}   "
        for cost in COST_BPS_GRID:
            res = next(r for n, c, r in sweep if n == n_steps and c == cost)
            row += f"{res.sharpe:>11.3f}   "
        print(row)

    for theme in ("light", "dark"):
        surface = THEMES[theme]["surface"]

        axes = plot_frequency_sweep(sweep, steps_per_day=STEPS_PER_DAY, theme=theme)
        out = FIGURES / f"hedge_frequency_{theme}.png"
        axes[0].figure.savefig(out, dpi=200, bbox_inches="tight", facecolor=surface)
        plt.close(axes[0].figure)
        print(f"  wrote {out.relative_to(FIGURES.parent)}")

        axes = plot_stress_curve(stress, theme=theme)
        out = FIGURES / f"heston_stress_{theme}.png"
        axes[0].figure.savefig(out, dpi=200, bbox_inches="tight", facecolor=surface)
        plt.close(axes[0].figure)
        print(f"  wrote {out.relative_to(FIGURES.parent)}")

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
