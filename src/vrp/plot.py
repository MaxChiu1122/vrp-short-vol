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
        contrast="#eb6834",  # second categorical slot, for a two-series comparison
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
        contrast="#d95926",  # second categorical slot, stepped for the dark surface
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


# Cost is an ORDERED quantity, so the cost levels take a one-hue sequential ramp
# (light -> dark as cost rises), not categorical hues. Both directions validate as
# ordinal ramps against their own surface.
COST_RAMP = {
    "light": ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"],
    "dark": ["#184f95", "#2a78d6", "#5598e7", "#86b6ef"],
}


def plot_frequency_sweep(sweep, *, steps_per_day: float, theme: str = "light", axes=None):
    """Sharpe and mean P&L against rebalancing frequency, one line per cost level.

    ``sweep`` is the ``(n_steps, cost_bps, VrpResult)`` list from
    ``vrp.hedge_frequency_sweep``. ``steps_per_day`` converts n_steps into
    rebalances per day so the x axis means something to a trader.

    The shape is the point. At zero cost Sharpe rises without bound -- hedging error
    is the only risk and it falls like 1/sqrt(n). Add cost and turnover (~sqrt(n))
    starts winning: each curve peaks and then falls, and the peak moves LEFT as cost
    rises. Past that peak you are paying away more spread than you remove in gamma.
    """
    import matplotlib.pyplot as plt

    t = THEMES[theme]
    ramp = COST_RAMP[theme]

    costs = sorted({c for _n, c, _r in sweep})
    if len(costs) > len(ramp):
        raise ValueError(f"the cost ramp has {len(ramp)} steps, got {len(costs)} cost levels")

    if axes is None:
        fig, axes = plt.subplots(
            2, 1, figsize=(7.5, 6.2), sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1])
        )
        fig.patch.set_facecolor(t["surface"])
    ax_sharpe, ax_mean = axes

    for color, cost in zip(ramp, costs):
        rows = sorted((n, r) for n, c, r in sweep if c == cost)
        x = np.array([n / steps_per_day for n, _r in rows])
        sharpes = np.array([r.sharpe for _n, r in rows])
        means = np.array([r.mean_pnl for _n, r in rows])

        label = "no cost" if cost == 0 else f"{cost:g} bp"
        ax_sharpe.plot(x, sharpes, "-o", color=color, lw=2.0, ms=5.0, label=label, zorder=3)
        ax_mean.plot(x, means, "-o", color=color, lw=2.0, ms=5.0, zorder=3)

        # Mark the Sharpe-maximizing frequency when the peak is interior.
        best = int(np.argmax(sharpes))
        if 0 < best < len(x) - 1:
            ax_sharpe.plot(
                x[best], sharpes[best], "o", ms=13, mfc="none", mec=t["critical"], mew=2.0, zorder=5
            )
            ax_sharpe.annotate(
                f"peak {x[best]:.0f}×/day",
                xy=(x[best], sharpes[best]),
                xytext=(0, 14),
                textcoords="offset points",
                color=t["critical"],
                fontsize=9,
                fontweight="bold",
                ha="center",
            )

    for ax in (ax_sharpe, ax_mean):
        ax.axhline(0.0, color=t["axis"], lw=1.0, zorder=1)
        ax.set_xscale("log")
        _style_axes(ax, t)

    ax_sharpe.set_ylabel("Sharpe (per trade)")
    ax_mean.set_ylabel("mean P&L per trade")
    ax_mean.set_xlabel("rebalances per day  (log scale)")
    leg = ax_sharpe.legend(
        title="transaction cost", loc="upper left", frameon=False, fontsize=9, ncol=2
    )
    leg.get_title().set_color(t["ink_secondary"])
    leg.get_title().set_fontsize(9)
    for text in leg.get_texts():
        text.set_color(t["ink_secondary"])
    ax_sharpe.set_title(
        "Hedging more is only free if trading is",
        color=t["ink"],
        fontsize=12,
        fontweight="bold",
        loc="left",
        pad=12,
    )
    return axes


def plot_moneyness_sweep(flat, skewed, *, theme: str = "light", axes=None):
    """Mean P&L and Sharpe across target delta, flat surface vs a skewed one.

    Both arguments are ``(delta, K, sigma_implied, VrpResult)`` lists from
    ``vrp.moneyness_sweep``. The x axis runs in-the-money (left) to out-of-the-money
    (right), the way a strike ladder is usually read.

    On a flat surface the edge simply peaks at the money, because dollar gamma
    ``½S²Γ`` and vega both do. Quote a realistic equity skew -- calls above the
    money priced *below* the ATM vol -- and the peak slides in-the-money: out there
    you are selling a vol spread that the surface has already taken away from you.
    """
    import matplotlib.pyplot as plt

    t = THEMES[theme]
    series = (
        ("flat surface", flat, t["series"]),
        ("with equity skew", skewed, t["contrast"]),
    )

    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.0), sharex=True)
        fig.patch.set_facecolor(t["surface"])
    ax_mean, ax_sharpe = axes

    for label, rows, color in series:
        x = np.array([d for d, _K, _s, _r in rows])
        means = np.array([r.mean_pnl for _d, _K, _s, r in rows])
        sharpes = np.array([r.sharpe for _d, _K, _s, r in rows])
        ax_mean.plot(x, means, "-o", color=color, lw=2.0, ms=5.0, label=label, zorder=3)
        ax_sharpe.plot(x, sharpes, "-o", color=color, lw=2.0, ms=5.0, zorder=3)

        best = int(np.argmax(sharpes))
        ax_sharpe.plot(
            x[best], sharpes[best], "o", ms=13, mfc="none", mec=color, mew=2.0, zorder=5
        )

    for ax in (ax_mean, ax_sharpe):
        ax.axhline(0.0, color=t["axis"], lw=1.0, zorder=1)
        _style_axes(ax, t)
    # The axes are shared, so invert ONCE -- inverting both would cancel out.
    ax_mean.invert_xaxis()  # ITM on the left, OTM on the right

    ax_mean.set_ylabel("mean P&L per trade")
    ax_sharpe.set_ylabel("Sharpe (per trade)")
    ax_sharpe.set_xlabel("target call delta   ←  in the money      out of the money  →")
    leg = ax_mean.legend(loc="lower left", frameon=False, fontsize=9)
    for text in leg.get_texts():
        text.set_color(t["ink_secondary"])
    ax_mean.set_title(
        "Skew decides where the edge lives",
        color=t["ink"],
        fontsize=12,
        fontweight="bold",
        loc="left",
        pad=12,
    )
    return axes


def plot_stress_curve(stress, *, theme: str = "light", axes=None):
    """Mean P&L and 5% CVaR against vol-of-vol -- the "what kills you" figure.

    ``stress`` is a sequence of (xi, mean_pnl, cvar_05) from a Heston sweep with the
    *expected* variance held fixed, so the only thing changing is how violently the
    realized vol moves. The two panels tell opposite stories on purpose: the mean
    barely reacts (and even drifts up), while the tail falls off a cliff. If you
    monitor only the top panel you never see the risk arriving.
    """
    import matplotlib.pyplot as plt

    t = THEMES[theme]
    xis = np.array([s[0] for s in stress])
    means = np.array([s[1] for s in stress])
    cvars = np.array([s[2] for s in stress])

    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(7.5, 5.6), sharex=True)
        fig.patch.set_facecolor(t["surface"])
    ax_mean, ax_cvar = axes

    ax_mean.axhline(0.0, color=t["axis"], lw=1.0, zorder=1)
    ax_mean.plot(xis, means, "-", color=t["series"], lw=2.0, zorder=3)
    ax_mean.plot(xis, means, "o", color=t["series"], ms=5.5, zorder=4)
    ax_mean.set_ylabel("mean P&L per trade")
    ax_mean.set_ylim(0.0, max(means) * 1.35)
    ax_mean.annotate(
        "the mean barely moves —\nit never warns you",
        xy=(xis[len(xis) // 2], means[len(means) // 2]),
        xytext=(0, 22),
        textcoords="offset points",
        color=t["ink_secondary"],
        fontsize=9,
        ha="center",
    )

    ax_cvar.axhline(0.0, color=t["axis"], lw=1.0, zorder=1)
    ax_cvar.plot(xis, cvars, "-", color=t["critical"], lw=2.0, zorder=3)
    ax_cvar.plot(xis, cvars, "o", color=t["critical"], ms=5.5, zorder=4)
    ax_cvar.set_ylabel("CVaR 5% (the tail)")
    ax_cvar.annotate(
        f"CVaR {cvars[-1] / cvars[0]:.1f}× worse\nby ξ = {xis[-1]:g}",
        xy=(0.97, 0.72),
        xycoords="axes fraction",
        color=t["critical"],
        fontsize=9.5,
        fontweight="bold",
        ha="right",
        va="top",
    )

    for ax in (ax_mean, ax_cvar):
        _style_axes(ax, t)
    ax_cvar.set_xlabel("Heston vol-of-vol  ξ   (expected variance held fixed)")
    ax_mean.set_title(
        "Stochastic realized vol: the mean survives, the tail does not",
        color=t["ink"],
        fontsize=12,
        fontweight="bold",
        loc="left",
        pad=12,
    )
    return axes


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
