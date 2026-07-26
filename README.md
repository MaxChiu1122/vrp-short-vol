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

## Install (local co-development)

`mcpricer` is a sibling repo; install it editable first, then this package:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ../mcpricer        # the engine (builds the C++ extension)
.venv/bin/pip install numpy pytest
.venv/bin/pip install -e . --no-deps        # this package
.venv/bin/pytest -q
```

(An external clone can instead resolve `mcpricer` from its git URL — see `pyproject.toml`.)

## Layout

```
src/vrp/
  hedge.py       # hedged_short_option_pnl — the VRP core (one path's P&L)   [Max]
  backtest.py    # vrp_backtest / vrp_curve — Monte-Carlo orchestration      [harness]
  stats.py       # summarize_pnl — mean / Sharpe / CVaR / win-rate           [harness]
  plot.py        # P&L histogram + VRP curve (optional: .[plot])             [harness]
tests/           # pytest
notebooks/       # analysis write-ups
```

## Status / roadmap

- [x] Harness + distribution stats (mean, Sharpe, CVaR) — tested.
- [ ] **`hedged_short_option_pnl`** — the two-vol hedged short-call P&L (the VRP core).
- [ ] VRP curve + P&L-distribution plots on a realistic contract.
- [ ] Heston-realized stress: show the tail fatten (crash risk) at a positive mean.
- [ ] Speed: move the per-path core into the `mcpricer` C++ engine (implied ≠ realized
      generalization of its delta-hedge) for large-path distributional runs.
