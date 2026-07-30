"""vrp -- short-volatility / variance-risk-premium distributional backtest.

Built on the ``mcpricer`` C++ pricing/Greeks engine. The workflow: sell a European
call, delta-hedge it at *implied* vol while the underlying *realizes* a different
vol, and Monte-Carlo the strategy over many market paths to get the full P&L
distribution -- mean edge (the VRP), Sharpe, and the short-gamma tail (CVaR).
"""

from .backtest import (
    BACKENDS,
    DEFAULT_CONTRACT,
    hedge_frequency_sweep,
    vrp_backtest,
    vrp_curve,
    vrp_pnl_paths,
)
from .hedge import hedged_short_option_pnl
from .stats import VrpResult, summarize_pnl
from .stress import HestonParams, heston_pnl_paths

__all__ = [
    "hedged_short_option_pnl",
    "summarize_pnl",
    "VrpResult",
    "vrp_backtest",
    "vrp_curve",
    "vrp_pnl_paths",
    "hedge_frequency_sweep",
    "HestonParams",
    "heston_pnl_paths",
    "BACKENDS",
    "DEFAULT_CONTRACT",
]
