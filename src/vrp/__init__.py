"""vrp -- short-volatility / variance-risk-premium distributional backtest.

Built on the ``mcpricer`` C++ pricing/Greeks engine. The workflow: sell a European
call, delta-hedge it at *implied* vol while the underlying *realizes* a different
vol, and Monte-Carlo the strategy over many market paths to get the full P&L
distribution -- mean edge (the VRP), Sharpe, and the short-gamma tail (CVaR).
"""

from .backtest import (
    BACKENDS,
    DEFAULT_CONTRACT,
    bootstrap_pnl_paths,
    hedge_frequency_sweep,
    vrp_backtest,
    vrp_curve,
    vrp_pnl_paths,
)
from .hedge import hedged_short_option_pnl
from .stats import VrpResult, summarize_pnl
from .marketdata import MarketData, load_market_data, measure_vrp
from .moneyness import (
    TYPICAL_EQUITY_SKEW,
    implied_vol_at_strike,
    moneyness_sweep,
    strike_for_delta,
)
from .stress import HestonParams, heston_pnl_paths

__all__ = [
    "hedged_short_option_pnl",
    "summarize_pnl",
    "VrpResult",
    "vrp_backtest",
    "vrp_curve",
    "vrp_pnl_paths",
    "hedge_frequency_sweep",
    "bootstrap_pnl_paths",
    "moneyness_sweep",
    "strike_for_delta",
    "implied_vol_at_strike",
    "TYPICAL_EQUITY_SKEW",
    "load_market_data",
    "measure_vrp",
    "MarketData",
    "HestonParams",
    "heston_pnl_paths",
    "BACKENDS",
    "DEFAULT_CONTRACT",
]
