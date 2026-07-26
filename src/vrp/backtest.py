"""Monte-Carlo distributional backtest of the short-vol / VRP strategy.

Orchestration only: run the per-path VRP core (vrp.hedge.hedged_short_option_pnl)
over many independent scenarios and summarize the P&L distribution, and sweep the
implied-vs-realized vol spread to trace the VRP curve.
"""

from __future__ import annotations

import numpy as np

from .hedge import hedged_short_option_pnl
from .stats import VrpResult, summarize_pnl

# A conventional 1-month ATM contract, quoted at 20% implied vol.
DEFAULT_CONTRACT = dict(S0=100.0, K=100.0, r=0.02, T=1.0 / 12.0, n_steps=21)


def vrp_backtest(
    *,
    sigma_implied: float,
    sigma_realized: float,
    n_paths: int = 50_000,
    seed: int = 0,
    **contract,
) -> VrpResult:
    """Backtest the delta-hedged short call over ``n_paths`` simulated markets.

    Sells at ``sigma_implied``; the underlying realizes ``sigma_realized``. Extra
    keyword args override DEFAULT_CONTRACT (S0, K, r, T, n_steps).
    """
    cfg = {**DEFAULT_CONTRACT, **contract}
    rng = np.random.default_rng(seed)
    pnls = np.empty(n_paths, dtype=float)
    for i in range(n_paths):
        pnls[i] = hedged_short_option_pnl(
            sigma_implied=sigma_implied, sigma_realized=sigma_realized, rng=rng, **cfg
        )
    return summarize_pnl(pnls)


def vrp_curve(
    *,
    spreads,
    sigma_realized: float = 0.16,
    n_paths: int = 50_000,
    seed: int = 0,
    **contract,
):
    """Trace mean strategy P&L vs the implied-minus-realized vol spread.

    For each ``spread`` in ``spreads``, sell at ``sigma_realized + spread`` while the
    world realizes ``sigma_realized``. Returns a list of (spread, mean_pnl, sharpe).
    Expectation: mean P&L rises ~monotonically with the spread (the harvested VRP).
    """
    out = []
    for spread in spreads:
        res = vrp_backtest(
            sigma_implied=sigma_realized + spread,
            sigma_realized=sigma_realized,
            n_paths=n_paths,
            seed=seed,
            **contract,
        )
        out.append((float(spread), res.mean_pnl, res.sharpe))
    return out
