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
from vrp import bootstrap_pnl_paths, measure_vrp  # noqa: E402
from vrp.roll import monthly_roll, regress_on_market  # noqa: E402
from vrp.moneyness import TYPICAL_EQUITY_SKEW, moneyness_sweep  # noqa: E402
from vrp.plot import (  # noqa: E402
    THEMES,
    plot_frequency_sweep,
    plot_moneyness_sweep,
    plot_pnl_distribution,
    plot_market_regression,
    plot_return_shape_comparison,
    plot_roll_equity,
    plot_stress_curve,
    plot_vrp_curve,
    plot_vrp_history,
)

# Strike ladder, quoted by delta the way a desk would.
TARGET_DELTAS = (0.75, 0.60, 0.50, 0.40, 0.30, 0.25, 0.15, 0.10)

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

    gbm_pnls = vrp_pnl_paths(
        sigma_implied=SIGMA_IMPLIED,
        sigma_realized=SIGMA_REALIZED,
        n_paths=N_PATHS,
        seed=SEED,
    )
    print(f"  {summarize_pnl(gbm_pnls)}")

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
        stress_pnls = heston_pnl_paths(
            sigma_implied=SIGMA_IMPLIED, heston=heston, n_paths=N_PATHS, seed=SEED
        )
        res = summarize_pnl(stress_pnls)
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

    # Strike ladder, flat surface vs a realistic equity skew.
    money_flat = moneyness_sweep(
        target_deltas=TARGET_DELTAS,
        sigma_atm=SIGMA_IMPLIED,
        sigma_realized=SIGMA_REALIZED,
        skew_slope=0.0,
        n_paths=20_000,
        seed=SEED,
    )
    money_skew = moneyness_sweep(
        target_deltas=TARGET_DELTAS,
        sigma_atm=SIGMA_IMPLIED,
        sigma_realized=SIGMA_REALIZED,
        skew_slope=TYPICAL_EQUITY_SKEW,
        n_paths=20_000,
        seed=SEED,
    )
    print("\n  delta      K    flat: s_impl mean Sharpe    skewed: s_impl mean Sharpe")
    for (d, K, s_f, r_f), (_d, _K, s_s, r_s) in zip(money_flat, money_skew):
        print(
            f"  {d:>5.2f} {K:>7.2f}      {s_f:>6.3f} {r_f.mean_pnl:>+7.4f} {r_f.sharpe:>6.3f}"
            f"        {s_s:>6.3f} {r_s.mean_pnl:>+7.4f} {r_s.sharpe:>6.3f}"
        )

    # Real data: the measured premium, and real return shapes through the hedge.
    measurement = measure_vrp()
    print(f"\n  measured VRP: {measurement}")

    real_series = [("GBM (Gaussian)", gbm_pnls)]
    for block, label in ((1, "real returns, IID"), (5, "real returns, block=5")):
        real_series.append((
            label,
            bootstrap_pnl_paths(
                sigma_implied=SIGMA_IMPLIED,
                sigma_realized=SIGMA_REALIZED,
                n_paths=N_PATHS,
                seed=SEED,
                block=block,
            ),
        ))
    for label, p in real_series:
        print(f"  {label:<24} {summarize_pnl(p)}")

    # The historical monthly roll. 1 vol point off VIX (it prints above the ATM
    # implied this sells) and 1bp of hedging cost -- the central, defensible case.
    roll = monthly_roll(implied_haircut=0.010, cost_bps=1.0)
    reg = regress_on_market(roll)
    print(f"\n  monthly roll: {roll}")
    print(f"  regression  : {reg}")

    for theme in ("light", "dark"):
        surface = THEMES[theme]["surface"]

        axes = plot_roll_equity(roll, theme=theme)
        out = FIGURES / f"roll_equity_{theme}.png"
        axes[0].figure.savefig(out, dpi=200, bbox_inches="tight", facecolor=surface)
        plt.close(axes[0].figure)
        print(f"  wrote {out.relative_to(FIGURES.parent)}")

        ax = plot_market_regression(roll, reg, theme=theme)
        out = FIGURES / f"roll_beta_{theme}.png"
        ax.figure.savefig(out, dpi=200, bbox_inches="tight", facecolor=surface)
        plt.close(ax.figure)
        print(f"  wrote {out.relative_to(FIGURES.parent)}")

        ax = plot_vrp_history(measurement, theme=theme)
        out = FIGURES / f"vrp_history_{theme}.png"
        ax.figure.savefig(out, dpi=200, bbox_inches="tight", facecolor=surface)
        plt.close(ax.figure)
        print(f"  wrote {out.relative_to(FIGURES.parent)}")

        ax = plot_return_shape_comparison(real_series, theme=theme)
        out = FIGURES / f"return_shape_{theme}.png"
        ax.figure.savefig(out, dpi=200, bbox_inches="tight", facecolor=surface)
        plt.close(ax.figure)
        print(f"  wrote {out.relative_to(FIGURES.parent)}")

        axes = plot_moneyness_sweep(money_flat, money_skew, theme=theme)
        out = FIGURES / f"moneyness_{theme}.png"
        axes[0].figure.savefig(out, dpi=200, bbox_inches="tight", facecolor=surface)
        plt.close(axes[0].figure)
        print(f"  wrote {out.relative_to(FIGURES.parent)}")

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
            gbm_pnls,
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
