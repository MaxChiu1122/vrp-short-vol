"""Moneyness: what changes when you sell an OTM call instead of an ATM one.

Desks quote options by **delta**, not by strike -- a "25-delta call" is a fixed
point on the surface, while a fixed strike drifts in and out of moneyness as spot
and vol move. So this sweeps target delta and solves for the strike.

Two things move as you go out of the money, in opposite directions:

  * **You harvest less.** The P&L is the vol spread weighted by dollar gamma
    ``½ S² Γ``, and both gamma and vega peak at the money. A 25-delta call collects
    a smaller premium and scalps less of the spread, so the mean edge falls.
  * **You get hit less often.** An OTM call usually expires untouched, so the win
    rate rises. What you are buying with that comfort is a rarer, relatively larger
    loss -- the same short-gamma bargain, struck further out.

**The flat-vol caveat, which is the real content here.** The rest of this repo
quotes one ``sigma_implied`` for every strike. Real equity index surfaces are
skewed: calls above the money trade at *lower* implied vol than the at-the-money
option. Selling a 25-delta call at the ATM vol therefore overstates the premium you
would actually collect. ``skew_slope`` models that as a linear function of
log-moneyness so the sweep can be run both ways -- and the gap between them is the
answer to "what does skew do to the edge?".
"""

from __future__ import annotations

import math
from statistics import NormalDist

from .backtest import DEFAULT_CONTRACT, vrp_backtest

_NORMAL = NormalDist()

# Illustrative equity-index call-side skew: implied vol falls as the strike rises.
# Roughly -2 vol points at a 25-delta call on a 1-month contract.
TYPICAL_EQUITY_SKEW = -0.4


def strike_for_delta(
    *,
    target_delta: float,
    S0: float,
    r: float,
    sigma: float,
    T: float,
) -> float:
    """The call strike whose Black-Scholes delta equals ``target_delta``.

    Inverts ``delta = N(d1)``: with ``d1 = N^-1(delta)``,

        K = S0 * exp(-d1 * sigma * sqrt(T) + (r + sigma^2 / 2) * T)

    ``target_delta`` is in (0, 1); below ~0.5 is out of the money for a call.
    ``sigma`` is the vol the strike is quoted against (the implied vol).
    """
    if not 0.0 < target_delta < 1.0:
        raise ValueError(f"target_delta must be in (0, 1), got {target_delta}")
    d1 = _NORMAL.inv_cdf(target_delta)
    return S0 * math.exp(-d1 * sigma * math.sqrt(T) + (r + 0.5 * sigma**2) * T)


def implied_vol_at_strike(
    *,
    K: float,
    S0: float,
    sigma_atm: float,
    skew_slope: float = 0.0,
) -> float:
    """A linear skew in log-moneyness: ``sigma(K) = sigma_atm + slope * ln(K/S0)``.

    ``skew_slope = 0`` is the flat surface the rest of the repo assumes. Negative
    values reproduce the equity-index shape, where calls above the money are quoted
    *cheaper* than the at-the-money option -- so a short-call strategy collects less
    premium the further out it sells.
    """
    return sigma_atm + skew_slope * math.log(K / S0)


def moneyness_sweep(
    *,
    target_deltas,
    sigma_atm: float = 0.20,
    sigma_realized: float = 0.16,
    skew_slope: float = 0.0,
    n_paths: int = 20_000,
    seed: int = 0,
    cost_bps: float = 0.0,
    **contract,
):
    """Sweep the strike by target delta. Returns ``(delta, K, sigma_implied, VrpResult)``.

    The strike is solved at ``sigma_atm`` (you pick the strike off the ATM quote),
    then the option is sold at ``implied_vol_at_strike`` -- identical to ``sigma_atm``
    on a flat surface, lower for an OTM call once ``skew_slope`` is negative.
    """
    cfg = {**DEFAULT_CONTRACT, **contract}
    out = []
    for target_delta in target_deltas:
        K = strike_for_delta(
            target_delta=float(target_delta),
            S0=cfg["S0"],
            r=cfg["r"],
            sigma=sigma_atm,
            T=cfg["T"],
        )
        sigma_implied = implied_vol_at_strike(
            K=K, S0=cfg["S0"], sigma_atm=sigma_atm, skew_slope=skew_slope
        )
        res = vrp_backtest(
            sigma_implied=sigma_implied,
            sigma_realized=sigma_realized,
            n_paths=n_paths,
            seed=seed,
            cost_bps=cost_bps,
            **{**cfg, "K": K},
        )
        out.append((float(target_delta), K, sigma_implied, res))
    return out
