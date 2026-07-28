"""Plot helpers (optional; needs the ``plot`` extra: pip install -e ".[plot]").

Kept separate so the core package has no hard matplotlib dependency.

Both figures render in a light or dark theme from the same code path -- the dark
steps are chosen for the dark surface, not flipped from the light ones. The two
colours in play (series blue, status red) clear the colour-vision, chroma,
lightness and contrast checks in both modes.
"""

from __future__ import annotations

import numpy as np

# Chart chrome, ink and the two data colours, per mode. The red is a *status*
# colour (the tail is a risk state), never a second data series.
THEMES = {
    "light": dict(
        surface="#fcfcfb",
        ink="#0b0b0b",
        ink_secondary="#52514e",
        muted="#898781",
        grid="#e1e0d9",
        axis="#c3c2b7",
        series="#2a78d6",
        critical="#d03b3b",
    ),
    "dark": dict(
        surface="#1a1a19",
        ink="#ffffff",
        ink_secondary="#c3c2b7",
        muted="#898781",
        grid="#2c2c2a",
        axis="#383835",
        series="#3987e5",
        critical="#d03b3b",
    ),
}


def _style_axes(ax, t: dict) -> None:
    """Recessive grid and axes; ink-coloured text. Applied to every panel."""
    ax.set_facecolor(t["surface"])
    ax.grid(True, color=t["grid"], lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=t["muted"], labelsize=9, length=0)
    ax.xaxis.label.set_color(t["ink_secondary"])
    ax.yaxis.label.set_color(t["ink_secondary"])


def plot_pnl_distribution(
    pnls,
    *,
    title: str = "Short-vol P&L distribution",
    theme: str = "light",
    ax=None,
):
    """Histogram of per-path P&L, with the worst 5% shaded as the CVaR region.

    CVaR is the *mean of a tail*, so the tail is drawn as an area rather than only
    a line -- the shaded bars are the outcomes being averaged. The point of the
    figure is that the mass sits to the right of zero while the shaded tail runs
    far to the left: many small wins, occasional real damage.
    """
    import matplotlib.pyplot as plt

    t = THEMES[theme]
    a = np.asarray(pnls, dtype=float)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        fig.patch.set_facecolor(t["surface"])

    mean = float(a.mean())
    k = max(1, int(round(0.05 * a.size)))
    tail = np.sort(a)[:k]
    cvar = float(tail.mean())
    var_05 = float(tail[-1])  # the 5th-percentile cutoff: where the tail region ends

    counts, edges = np.histogram(a, bins=90)
    widths = np.diff(edges)
    # A 2px surface gap between bars keeps adjacent fills legible.
    colors = [t["critical"] if edges[i + 1] <= var_05 else t["series"] for i in range(len(counts))]
    ax.bar(
        edges[:-1],
        counts,
        width=widths * 0.88,
        align="edge",
        color=colors,
        linewidth=0,
        zorder=2,
    )

    ax.axvline(0.0, color=t["axis"], lw=1.0, zorder=3)
    ax.axvline(mean, color=t["ink"], lw=2.0, zorder=4)
    ax.axvline(cvar, color=t["critical"], lw=2.0, ls=(0, (4, 2)), zorder=4)

    # Direct labels rather than a legend box for the two reference lines; the
    # shaded region gets the one legend entry it needs.
    top = ax.get_ylim()[1]
    ax.annotate(
        f"mean {mean:+.3f}",
        xy=(mean, top * 0.97),
        xytext=(6, 0),
        textcoords="offset points",
        color=t["ink"],
        fontsize=9.5,
        fontweight="bold",
        va="top",
    )
    ax.annotate(
        f"CVaR 5%  {cvar:+.3f}",
        xy=(cvar, top * 0.80),
        xytext=(-6, 0),
        textcoords="offset points",
        color=t["critical"],
        fontsize=9.5,
        fontweight="bold",
        ha="right",
        va="top",
    )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=t["series"]),
        plt.Rectangle((0, 0), 1, 1, color=t["critical"]),
    ]
    leg = ax.legend(
        handles,
        ["P&L per trade", "worst 5% (the CVaR region)"],
        loc="upper left",
        frameon=False,
        fontsize=9,
    )
    for text in leg.get_texts():
        text.set_color(t["ink_secondary"])

    ax.set_xlabel("P&L per trade (currency, same units as S)")
    ax.set_ylabel("paths")
    ax.set_title(title, color=t["ink"], fontsize=12, fontweight="bold", loc="left", pad=12)
    _style_axes(ax, t)
    return ax


def plot_vrp_curve(curve, *, theme: str = "light", axes=None):
    """Mean P&L and Sharpe vs the implied-minus-realized vol spread.

    Two stacked panels sharing one x axis rather than two y scales on one panel --
    mean P&L and Sharpe have unrelated units, and a twin axis would let the reader
    infer a crossing point that means nothing.
    """
    import matplotlib.pyplot as plt

    t = THEMES[theme]
    spreads = np.array([c[0] for c in curve]) * 100.0  # vol points
    means = np.array([c[1] for c in curve])
    sharpes = np.array([c[2] for c in curve])

    if axes is None:
        fig, axes = plt.subplots(
            2, 1, figsize=(7.5, 5.6), sharex=True, gridspec_kw=dict(height_ratios=[1.35, 1])
        )
        fig.patch.set_facecolor(t["surface"])
    ax_mean, ax_sharpe = axes

    for ax, y, label in ((ax_mean, means, "mean P&L per trade"), (ax_sharpe, sharpes, "Sharpe (per trade)")):
        ax.axhline(0.0, color=t["axis"], lw=1.0, zorder=1)
        ax.plot(spreads, y, "-", color=t["series"], lw=2.0, zorder=3)
        ax.plot(spreads, y, "o", color=t["series"], ms=5.5, zorder=4)
        ax.set_ylabel(label)
        _style_axes(ax, t)

    # The region where you sold vol too cheap -- losing by construction.
    for ax in (ax_mean, ax_sharpe):
        ax.axvspan(spreads.min(), 0.0, color=t["critical"], alpha=0.07, zorder=0)
    ax_mean.annotate(
        "sold below realized\n→ loses by construction",
        xy=(spreads.min(), means.max() * 0.55),
        xytext=(6, 0),
        textcoords="offset points",
        color=t["critical"],
        fontsize=9,
        va="center",
    )

    ax_sharpe.set_xlabel("implied − realized vol (vol points)")
    ax_mean.set_title(
        "Variance risk premium vs the vol spread",
        color=t["ink"],
        fontsize=12,
        fontweight="bold",
        loc="left",
        pad=12,
    )
    return axes
