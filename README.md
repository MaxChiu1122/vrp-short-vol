# vrp-short-vol

A **systematic short-volatility strategy**, backtested as a Monte-Carlo *distribution*
rather than a single historical path. Built on the
[`mcpricer`](https://github.com/MaxChiu1122/Monte-Carlo_option_pricer) C++ pricing/Greeks
engine — this repo is the strategy/research layer that consumes it.

## The idea

Sell a European call and delta-hedge it to expiry. You **sell and hedge at the implied
vol** (the premium you collect and every delta you trade use `sigma_implied`), but the
underlying **realizes** a different vol, `sigma_realized`. A delta-hedged short call's
P&L is the *gamma-weighted vol spread*:

```
P&L  ≈  ½ · Σ_t  S_t² · Γ_t · (σ_implied² − σ_realized²) · dt
```

When `σ_implied > σ_realized` — the empirical norm, since implied vol carries a risk
premium — the strategy earns the **variance risk premium (VRP)**. Because the book is
**short gamma**, a realized-vol spike (a crash) produces a sharp loss: the P&L is
left-skewed, and the tail is what actually matters.

## Why a *distributional* backtest

A single historical path tells you what happened once. Because `mcpricer` is fast and
reproducible, we instead simulate the strategy over **many thousands of market paths**
and report the full P&L distribution:

- **mean P&L** — the harvested VRP (the edge),
- **Sharpe** — edge per unit of P&L dispersion,
- **CVaR (5%)** — the mean of the worst 5% of outcomes: the short-gamma tail,
- and the **VRP curve** — mean P&L as a function of the (implied − realized) vol spread.

A *stress* mode simulates realized vol under **Heston** (stochastic vol / vol-of-vol,
via `mcpricer`) to show the left tail fatten even when the mean stays positive — the
"what kills you" story.

## Results

The contract: a 1-month ATM European call (`S = K = 100`, `r = 2%`), delta-hedged over 21
daily steps, **sold at 20% implied into a world that realizes 16%** — a 4-vol-point spread.
50,000 paths, seed 11.

| metric | value |
|---|---|
| mean P&L per trade | **+0.4626** |
| Sharpe (per trade) | **1.23** |
| win rate | **90.8%** |
| CVaR (5%) | **−0.2610** |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/pnl_distribution_dark.png">
  <img alt="Histogram of per-path P&L: the bulk of the mass sits right of zero with mean +0.463, and a long left tail runs to about -2. The worst 5% of paths are shaded, averaging -0.261." src="docs/pnl_distribution_light.png">
</picture>

That shape *is* the strategy: it wins 91% of the time and still carries a losing 5% tail.
The shaded bars are the outcomes CVaR averages over — the tail is drawn as an area because
CVaR is the mean of one, not a threshold. Many small wins, occasional real damage, which is
what being **short gamma** buys you.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/vrp_curve_dark.png">
  <img alt="Two stacked panels sharing an x axis of implied-minus-realized vol in vol points. Mean P&L rises linearly from -0.46 at -4 points through zero to +0.70 at +6 points; Sharpe rises from -1.16 to +1.72 with mild concavity. The negative-spread region is shaded as losing by construction." src="docs/vrp_curve_light.png">
</picture>

The edge is linear in the vol spread and crosses zero exactly where implied meets realized —
sell vol below what the world delivers and the same machinery bleeds. Mean P&L and Sharpe sit
in separate panels rather than on twin axes, since their units are unrelated.

**A check on the core worth more than the plots.** The curve's slope is `0.1156` per vol
point; Black-Scholes vega on this contract is `0.1150`. They agree to 0.5%, because for small
spreads `mean P&L ≈ vega · (σ_implied − σ_realized)`. That pins the P&L engine's *magnitude*
against a closed-form Greek — evidence independent of the test suite, which only checks signs
and distribution shape.

Both figures regenerate from a seeded run:

```bash
python scripts/make_figures.py
```

> Note on scope: `sigma_realized` here is a **model input, not a measurement**. These results
> establish the *mechanics* of harvesting the premium and the shape of its risk — not evidence
> that the premium exists in real markets. (It does: S&P implied has run ~2–4 vol points above
> subsequent realized for decades. This repo assumes that, it doesn't demonstrate it.)

## Install (local co-development)

`mcpricer` is a sibling repo; install it editable first, then this package:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ../mcpricer        # the engine (builds the C++ extension)
.venv/bin/pip install numpy pytest
.venv/bin/pip install -e . --no-deps        # this package
.venv/bin/pytest -q
```

Regenerating the figures additionally needs matplotlib — `pip install -e ".[plot]"`.

(An external clone can instead resolve `mcpricer` from its git URL — see `pyproject.toml`.)

## Layout

```
src/vrp/
  hedge.py       # hedged_short_option_pnl — the VRP core (one path's P&L)   [Max]
  backtest.py    # vrp_pnl_paths / vrp_backtest / vrp_curve — MC harness     [harness]
  stats.py       # summarize_pnl — mean / Sharpe / CVaR / win-rate           [harness]
  plot.py        # P&L histogram + VRP curve (optional: .[plot])             [harness]
scripts/         # make_figures.py — regenerates docs/*.png from a seeded run
docs/            # the README figures (light + dark)
tests/           # pytest
notebooks/       # analysis write-ups
```

## Status / roadmap

- [x] Harness + distribution stats (mean, Sharpe, CVaR) — tested.
- [x] **`hedged_short_option_pnl`** — the two-vol hedged short-call P&L (the VRP core).
- [x] VRP curve + P&L-distribution plots on a realistic contract.
- [ ] Heston-realized stress: show the tail fatten (crash risk) at a positive mean.
- [ ] Speed: move the per-path core into the `mcpricer` C++ engine (implied ≠ realized
      generalization of its delta-hedge) for large-path distributional runs.
