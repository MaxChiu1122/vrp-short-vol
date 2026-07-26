"""The VRP core: one scenario's P&L of a delta-hedged short option.

This is the piece that makes the strategy a *strategy* rather than a repricing.
It is deliberately the one function YOU (Max) write and defend -- the harness,
stats, and plots around it are wiring.
"""

from __future__ import annotations

import mcpricer as mc


def hedged_short_option_pnl(
    *,
    S0: float,
    K: float,
    r: float,
    sigma_implied: float,
    sigma_realized: float,
    T: float,
    n_steps: int,
    rng,  # numpy.random.Generator
) -> float:
    """P&L of SELLING one European call and delta-hedging it to expiry.

    The whole point: you SELL and HEDGE at the IMPLIED vol (the premium you collect
    and every delta you trade use ``sigma_implied``), but the underlying REALIZES
    ``sigma_realized``. The two need not be equal -- and their gap is where the money
    (or the loss) comes from.

    TODO(max): write the self-financing hedge for one path. Mirror the pure-Python
    reference ``delta_hedge_pnl_python`` in mcpricer's tools/delta_hedge_backtest.py,
    but SPLIT THE VOL:

      * premium (t=0), initial delta, and every rebalance delta   -> use sigma_implied
        via  mc.black_scholes_price / mc.black_scholes_delta(type=mc.OptionType.Call, ...)
        with the *remaining* maturity tau at each step.
      * the GBM path step  S *= exp((r - 1/2 sigma_realized^2) dt + sigma_realized*sqrt(dt)*Z)
        -> use sigma_realized, with Z = rng.standard_normal().

    Rebalance on steps 1..n_steps-1 (skip the final one -- it is a no-op at expiry and
    calling black_scholes_delta at tau=0 divides by zero). Return the expiry P&L:
        cash + delta * S_T - max(S_T - K, 0).

    The mechanism you are demonstrating: a delta-hedged short call earns
        ~ 1/2 * sum_t S_t^2 * Gamma_t * (sigma_implied^2 - sigma_realized^2) * dt
    -- the gamma-weighted vol spread. sigma_implied > sigma_realized  =>  positive
    expected P&L (the variance risk premium); being short gamma, a realized-vol spike
    is the tail risk.
    """
    raise NotImplementedError(
        "Max writes the two-vol hedged short-call P&L for one path (see docstring)."
    )
