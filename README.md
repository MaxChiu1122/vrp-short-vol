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

## The stress: what actually kills the book

Constant realized vol is a convenient fiction. Real vol clusters, spikes, and spikes *while the
market falls*. So the stress keeps the **hedger unchanged** — still quoting and hedging off flat
Black-Scholes at 20%, because a quote is all a desk observes — and replaces the underlying's
dynamics with **Heston**:

```
dS_t = r S_t dt + √v_t S_t dW¹
dv_t = κ(θ − v_t) dt + ξ √v_t dW²,     corr(dW¹, dW²) = ρ
```

The calibration is the point: `v0 = θ = σ_realized²`, so the **expected variance is identical to
the GBM run**. Only the *violence* of the vol changes. Anything that moves is attributable to
vol-of-vol `ξ` and the leverage correlation `ρ = −0.7`, not to selling a different amount of vol.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/heston_stress_dark.png">
  <img alt="Two stacked panels against Heston vol-of-vol from 0 to 0.8. Mean P&L per trade is nearly flat, drifting from +0.46 to +0.58. CVaR 5% falls steadily from -0.26 to -1.03, four times worse." src="docs/heston_stress_light.png">
</picture>

| ξ (vol-of-vol) | mean P&L | CVaR 5% | Sharpe | win rate | Feller |
|---|---|---|---|---|---|
| 0.0 | +0.4615 | −0.2582 | 1.222 | 90.7% | ok |
| 0.2 | +0.4682 | −0.3403 | 1.158 | 89.4% | ok |
| 0.4 | +0.4894 | −0.5337 | 1.028 | 86.4% | violated |
| 0.6 | +0.5272 | −0.7741 | 0.916 | 83.5% | violated |
| 0.8 | +0.5804 | −1.0306 | 0.844 | 81.3% | violated |

**The mean never warns you.** It doesn't fall — it *rises*, from +0.46 to +0.58. Meanwhile CVaR
degrades **4×** and Sharpe drops from 1.22 to 0.84. A desk monitoring average P&L would see a
strategy getting better right up until the tail arrived. That asymmetry is the entire argument for
reporting expected shortfall next to the mean, and it is why this repo's summary carries both.

Two details worth noting. At ξ ≥ 0.4 the **Feller condition** `2κθ ≥ ξ²` is violated, so the
variance can reach zero — exactly the regime the engine's full-truncation Euler scheme exists to
handle, and normal for calibrated parameters. And at ξ = 0 the stress reproduces the GBM backtest
(+0.4615 vs +0.4626), which is what validates the Heston path against the one we already trust.

## Performance

The per-path core lives in `mcpricer`'s multithreaded C++ engine; the Python implementation stays
as the readable reference and the default. Both are the same experiment.

```bash
python scripts/benchmark_backends.py
```

| paths | python | fast (C++) | speedup | paths/s |
|---|---|---|---|---|
| 10,000 | 0.139 s | 0.0019 s | **72×** | 5.2M |
| 50,000 | 0.701 s | 0.0094 s | **75×** | 5.3M |
| 200,000 | 2.813 s | 0.0373 s | **75×** | 5.4M |

At 200k paths the two agree distributionally — mean +0.4618 vs +0.4603 (1.29 combined s.e.),
Sharpe 1.230 vs 1.227, CVaR 5% −0.2548 vs −0.2549. They can't agree path-for-path: the backends
draw from different generators (NumPy's PCG64 vs the engine's counter-based streams), so the tests
compare them statistically. The C++ arrays are reproducible and **independent of thread count** —
fixed chunking with a per-chunk RNG stream, asserted element-wise.

```python
from vrp import vrp_backtest
vrp_backtest(sigma_implied=0.20, sigma_realized=0.16, n_paths=1_000_000, backend="fast")
```

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
  stress.py      # HestonParams / heston_pnl_paths — the stochastic-vol stress
  stats.py       # summarize_pnl — mean / Sharpe / CVaR / win-rate           [harness]
  plot.py        # P&L histogram, VRP curve, stress curve (optional: .[plot])
scripts/         # make_figures.py, benchmark_backends.py
docs/            # the README figures (light + dark)
tests/           # pytest
notebooks/       # analysis write-ups
```

## Status / roadmap

- [x] Harness + distribution stats (mean, Sharpe, CVaR) — tested.
- [x] **`hedged_short_option_pnl`** — the two-vol hedged short-call P&L (the VRP core).
- [x] VRP curve + P&L-distribution plots on a realistic contract.
- [x] Heston-realized stress: the tail fattens 4× while the mean holds.
- [x] Speed: the per-path core moved into the `mcpricer` C++ engine (implied ≠ realized
      generalization of its delta-hedge) — **75×**, 5.4M paths/s.

Next: replace the assumed `sigma_realized` with a **block bootstrap of real returns**, and use
VIX-vs-realized to show the premium is a measured fact rather than a parameter.
