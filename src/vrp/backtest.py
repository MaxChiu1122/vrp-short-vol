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

    Both backends simulate the same model and agree in distribution, but NOT
    path-for-path: they draw from different generators (NumPy's PCG64 vs the
    engine's counter-based stream), so compare them statistically, never elementwise.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    cfg = {**DEFAULT_CONTRACT, **contract}

    if backend == "fast":
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
            sigma_implied=sigma_implied, sigma_realized=sigma_realized, rng=rng, **cfg
        )
    return pnls


def vrp_backtest(
    *,
    sigma_implied: float,
    sigma_realized: float,
    n_paths: int = 50_000,
    seed: int = 0,
    backend: str = "python",
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
