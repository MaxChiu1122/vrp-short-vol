"""Heston-realized stress: what happens when realized vol is itself stochastic.

The GBM backtest asks "what if the world realizes a *constant* vol below implied?"
That is the clean way to see the variance risk premium, but it is not how a
short-vol book dies. Real realized vol clusters, spikes, and spikes *while the
market falls*.

So the stress holds the hedger fixed -- still quoting and hedging off flat
Black-Scholes at ``sigma_implied``, because that is all a desk observes -- and
replaces the underlying's dynamics with Heston:

    dS_t = r S_t dt + sqrt(v_t) S_t dW1
    dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW2,   corr(dW1, dW2) = rho

Calibrate ``v0 = theta = sigma_realized**2`` and the *expected* variance matches
the GBM run exactly, so the mean edge is broadly preserved. What changes is the
tail: vol-of-vol (``xi``) plus a negative ``rho`` manufactures the crash paths that
short gamma cannot survive. The result to report is CVaR moving while the mean
does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backtest import DEFAULT_CONTRACT


@dataclass(frozen=True)
class HestonParams:
    """Heston parameters. ``v0`` and ``theta`` are VARIANCES (vol squared)."""

    v0: float  # initial variance
    kappa: float  # mean-reversion speed of the variance
    theta: float  # long-run variance
    xi: float  # vol-of-variance ("vol of vol")
    rho: float  # spot/variance correlation; negative = the leverage effect

    @classmethod
    def matched_to(
        cls,
        sigma_realized: float,
        *,
        kappa: float = 2.0,
        xi: float = 0.5,
        rho: float = -0.7,
    ) -> HestonParams:
        """Heston with the same *expected* variance as a GBM run at ``sigma_realized``.

        Setting v0 = theta = sigma_realized**2 starts the variance at its long-run
        mean and keeps it there in expectation, so this is the honest comparison:
        only the vol-of-vol and the correlation differ from the GBM case, and any
        change in the P&L distribution is attributable to them.
        """
        variance = sigma_realized**2
        return cls(v0=variance, kappa=kappa, theta=variance, xi=xi, rho=rho)

    @property
    def feller_satisfied(self) -> bool:
        """2 kappa theta >= xi^2 -- whether the variance can reach zero.

        Violated is normal for calibrated parameters and is precisely the regime the
        full-truncation scheme exists to handle; it is not an error, but it is worth
        knowing which side of the line a stress sits on.
        """
        return 2.0 * self.kappa * self.theta >= self.xi**2

    @property
    def long_run_vol(self) -> float:
        """sqrt(theta) -- the vol the variance mean-reverts to, for reporting."""
        return float(np.sqrt(self.theta))


def heston_pnl_paths(
    *,
    sigma_implied: float,
    heston: HestonParams,
    n_paths: int = 50_000,
    seed: int = 0,
    n_threads: int = 0,
    **contract,
) -> np.ndarray:
    """Per-path P&L of the short call hedged at flat ``sigma_implied`` under Heston.

    Runs in mcpricer's C++ core (there is no Python reference for this one -- the
    GBM backend in ``vrp.backtest`` is the readable version of the mechanics, and
    this swaps only the path generator). Extra keyword args override
    DEFAULT_CONTRACT (S0, K, r, T, n_steps).
    """
    import mcpricer as mc

    if not hasattr(mc, "delta_hedge_pnl_heston_paths"):
        raise RuntimeError(
            "the installed mcpricer has no delta_hedge_pnl_heston_paths; "
            "reinstall the engine (pip install -e ../mcpricer) to get the stress core"
        )
    cfg = {**DEFAULT_CONTRACT, **contract}
    return mc.delta_hedge_pnl_heston_paths(
        S0=cfg["S0"],
        K=cfg["K"],
        r=cfg["r"],
        sigma_implied=sigma_implied,
        v0=heston.v0,
        kappa=heston.kappa,
        theta=heston.theta,
        xi=heston.xi,
        rho=heston.rho,
        T=cfg["T"],
        n_steps=cfg["n_steps"],
        n_scenarios=n_paths,
        seed=seed,
        n_threads=n_threads,
    )
