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

BACKENDS = ("python", "fast")


def vrp_pnl_paths(
    *,
    sigma_implied: float,
    sigma_realized: float,
    n_paths: int = 50_000,
    seed: int = 0,
    backend: str = "python",
    n_threads: int = 0,
    cost_bps: float = 0.0,
    **contract,
) -> np.ndarray:
    """Per-path P&L of the delta-hedged short call over ``n_paths`` simulated markets.

    Sells at ``sigma_implied``; the underlying realizes ``sigma_realized``. Extra
    keyword args override DEFAULT_CONTRACT (S0, K, r, T, n_steps).

    Returns the raw distribution rather than a summary -- plots need the array, and
    summarizing is a separate concern (see ``vrp_backtest``).

    Parameters
    ----------
    backend:
        ``"python"`` runs ``vrp.hedge.hedged_short_option_pnl`` once per path -- the
        readable reference. ``"fast"`` calls the same experiment in mcpricer's
        multithreaded C++ core, ~2 orders of magnitude quicker.
    n_threads:
        Only used by the ``"fast"`` backend; 0 means all cores. The result is
        independent of this (fixed chunking + per-chunk RNG streams).
    cost_bps:
        Proportional transaction cost in basis points of notional traded, charged
        on the initial hedge and every rebalance. ``"python"`` backend only -- the
        C++ core does not model costs yet, so asking for both raises rather than
        silently returning a gross number.

    Both backends simulate the same model and agree in distribution, but NOT
    path-for-path: they draw from different generators (NumPy's PCG64 vs the
    engine's counter-based stream), so compare them statistically, never elementwise.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    cfg = {**DEFAULT_CONTRACT, **contract}

    if backend == "fast":
        if cost_bps:
            raise NotImplementedError(
                "the C++ backend does not model transaction costs; "
                'use backend="python" for cost_bps != 0'
            )
        import mcpricer as mc

        if not hasattr(mc, "delta_hedge_pnl_two_vol_paths"):
            raise RuntimeError(
                "the installed mcpricer has no delta_hedge_pnl_two_vol_paths; "
                "reinstall the engine (pip install -e ../mcpricer) to get the two-vol core"
            )
        return mc.delta_hedge_pnl_two_vol_paths(
            S0=cfg["S0"],
            K=cfg["K"],
            r=cfg["r"],
            sigma_implied=sigma_implied,
            sigma_realized=sigma_realized,
            T=cfg["T"],
            n_steps=cfg["n_steps"],
            n_scenarios=n_paths,
            seed=seed,
            n_threads=n_threads,
        )

    rng = np.random.default_rng(seed)
    pnls = np.empty(n_paths, dtype=float)
    for i in range(n_paths):
        pnls[i] = hedged_short_option_pnl(
            sigma_implied=sigma_implied,
            sigma_realized=sigma_realized,
            rng=rng,
            cost_bps=cost_bps,
            **cfg,
        )
    return pnls


def vrp_backtest(
    *,
    sigma_implied: float,
    sigma_realized: float,
    n_paths: int = 50_000,
    seed: int = 0,
    backend: str = "python",
    cost_bps: float = 0.0,
    **contract,
) -> VrpResult:
    """Backtest the delta-hedged short call and summarize the P&L distribution.

    Thin wrapper: ``summarize_pnl(vrp_pnl_paths(...))``. Same arguments.
    """
    return summarize_pnl(
        vrp_pnl_paths(
            sigma_implied=sigma_implied,
            sigma_realized=sigma_realized,
            n_paths=n_paths,
            seed=seed,
            backend=backend,
            cost_bps=cost_bps,
            **contract,
        )
    )


def vrp_curve(
    *,
    spreads,
    sigma_realized: float = 0.16,
    n_paths: int = 50_000,
    seed: int = 0,
    backend: str = "python",
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
            backend=backend,
            **contract,
        )
        out.append((float(spread), res.mean_pnl, res.sharpe))
    return out


def bootstrap_pnl_paths(
    *,
    sigma_implied: float,
    sigma_realized: float,
    n_paths: int = 20_000,
    seed: int = 0,
    block: int = 1,
    cost_bps: float = 0.0,
    since: str | None = None,
    **contract,
) -> np.ndarray:
    """Per-path P&L with the market path resampled from REAL S&P 500 returns.

    Same hedge, same option, same implied vol -- only the return-generating process
    changes, from Gaussian to a bootstrap of actual daily history. ``sigma_realized``
    is still honoured as the *level*: the resampled returns are rescaled to it, so
    this differs from ``vrp_pnl_paths`` in the SHAPE of the returns (fat tails, and
    with ``block > 1`` volatility clustering) and nothing else.

    ``block=1`` is an IID bootstrap; ``block=5`` keeps a week of consecutive real
    days together. ``since`` trims the history (e.g. ``"2000-01-01"``).
    """
    from .bootstrap import bootstrap_log_returns, historical_daily_returns

    cfg = {**DEFAULT_CONTRACT, **contract}
    dt = cfg["T"] / cfg["n_steps"]
    rng = np.random.default_rng(seed)

    paths = bootstrap_log_returns(
        historical_daily_returns(since=since),
        n_paths=n_paths,
        n_steps=cfg["n_steps"],
        rng=rng,
        block=block,
        target_vol=sigma_realized,
        r=cfg["r"],
        dt=dt,
    )

    pnls = np.empty(n_paths, dtype=float)
    for i in range(n_paths):
        pnls[i] = hedged_short_option_pnl(
            sigma_implied=sigma_implied,
            sigma_realized=sigma_realized,
            rng=rng,
            cost_bps=cost_bps,
            log_returns=paths[i],
            **cfg,
        )
    return pnls


def hedge_frequency_sweep(
    *,
    n_steps_grid,
    cost_bps_grid=(0.0,),
    sigma_implied: float = 0.20,
    sigma_realized: float = 0.16,
    n_paths: int = 20_000,
    seed: int = 0,
    **contract,
):
    """Sweep rebalancing frequency against transaction cost.

    Returns a list of ``(n_steps, cost_bps, VrpResult)``.

    This is the trade-off that decides how a real vol book is run, and the two
    sides scale differently:

      * **Risk falls like 1/sqrt(n_steps).** With continuous hedging a positive vol
        spread wins on *every* path; all of the dispersion (and the whole left tail)
        is discretization error, and it shrinks as you rebalance more.
      * **Cost grows like sqrt(n_steps).** Each rebalance trades ~ Gamma*S*sigma*sqrt(dt)
        shares, so n_steps rebalances turn over ~ sqrt(n_steps) of notional.

    At zero cost more hedging is monotonically better. At any positive cost the two
    curves cross and Sharpe peaks at a finite frequency -- hedging *more* past that
    point pays away more in spread than it removes in gamma risk.
    """
    out = []
    for n_steps in n_steps_grid:
        for cost_bps in cost_bps_grid:
            res = vrp_backtest(
                sigma_implied=sigma_implied,
                sigma_realized=sigma_realized,
                n_paths=n_paths,
                seed=seed,
                cost_bps=float(cost_bps),
                n_steps=int(n_steps),
                **contract,
            )
            out.append((int(n_steps), float(cost_bps), res))
    return out
