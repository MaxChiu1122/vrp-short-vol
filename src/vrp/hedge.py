"""The VRP core: one scenario's P&L of a delta-hedged short option.

This is the piece that makes the strategy a *strategy* rather than a repricing.
"""

from __future__ import annotations

import math

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
    cost_bps: float = 0.0,
) -> float:
    """P&L of SELLING one European call and delta-hedging it to expiry.

    You SELL and HEDGE at ``sigma_implied`` (the premium collected and every delta
    traded use it), but the underlying REALIZES ``sigma_realized``. The gap is the
    P&L: a delta-hedged short call earns the gamma-weighted vol spread

        ~ 1/2 * sum_t S_t^2 * Gamma_t * (sigma_implied^2 - sigma_realized^2) * dt

    so sigma_implied > sigma_realized gives a positive expected P&L (the variance
    risk premium). Being short gamma, the distribution is left-skewed.

    ``cost_bps`` charges a proportional transaction cost, in basis points of the
    notional traded, on every share bought or sold: ``cost * |d_delta| * S``. It is
    the half-spread plus fees you actually pay to move the hedge. Costs are charged
    on the INITIAL hedge and on every rebalance; the expiry unwind is assumed
    costless (the position settles at expiry rather than being traded out).

    Note the scaling, which is why hedging more is not simply better. Each
    rebalance trades ~ Gamma * |dS| ~ Gamma * S * sigma * sqrt(dt) shares, so total
    turnover over n_steps rebalances grows like sqrt(n_steps) while the
    discrete-hedging risk falls like 1/sqrt(n_steps). Cost and risk move in
    opposite directions and there is an interior optimum.

    Rates and vols are annualized; ``T`` is in years. The path is simulated under
    the risk-neutral drift ``r``, which makes E[P&L] exactly zero when the two vols
    agree and ``cost_bps`` is zero, for any ``n_steps``. Units: currency, same as
    ``S0``.
    """
    dt = T / n_steps
    cost_rate = cost_bps * 1e-4

    # t=0: collect the premium, then buy the initial hedge out of that cash.
    # Both are quoted at sigma_implied -- it is all you can observe.
    cash = mc.black_scholes_price(mc.OptionType.Call, S0, K, r, sigma_implied, T)
    delta = mc.black_scholes_delta(mc.OptionType.Call, S0, K, r, sigma_implied, T)
    cash -= delta * S0
    cash -= cost_rate * abs(delta) * S0  # you pay to put the hedge on

    S = S0
    for i in range(1, n_steps + 1):
        # The world moves at sigma_realized; we held the previous delta through it.
        Z = rng.standard_normal()
        S *= math.exp(
            (r - 0.5 * sigma_realized**2) * dt + sigma_realized * math.sqrt(dt) * Z
        )
        cash *= math.exp(r * dt)  # self-financing: the cash leg earns the risk-free rate

        # Rebalance at the *remaining* maturity, funded entirely from cash. Skipped at
        # i == n_steps: tau = 0 divides by zero in d1, and we liquidate immediately anyway.
        if i < n_steps:
            tau = T - i * dt
            new_delta = mc.black_scholes_delta(mc.OptionType.Call, S, K, r, sigma_implied, tau)
            traded = new_delta - delta
            cash -= traded * S
            cash -= cost_rate * abs(traded) * S  # cost is on the notional traded
            delta = new_delta

    # Liquidate: cash + the shares we hold, less what we owe the option buyer.
    return cash + delta * S - max(S - K, 0.0)