"""Statistical Arbitrage Pairs Trading"""

from .pairs_finder import find_cointegrated_pairs, engle_granger_test, calculate_hurst_exponent
from .trading_strategy import PairsTradingStrategy
from .backtest import run_pairs_backtest, print_results

__all__ = [
    'find_cointegrated_pairs',
    'engle_granger_test', 
    'calculate_hurst_exponent',
    'PairsTradingStrategy',
    'run_pairs_backtest',
    'print_results'
]
