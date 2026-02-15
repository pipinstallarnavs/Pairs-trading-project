"""Statistical Arbitrage Pairs Trading"""

from .pairs_finder import find_cointegrated_pairs, PairStats
from .trading_strategy import PairsTradingStrategy, Trade
from .backtest import RollingWindowBacktest
from .data_loader import download_stock_data, get_sp500_tickers

__all__ = [
    'find_cointegrated_pairs',
    'PairStats',
    'PairsTradingStrategy',
    'Trade',
    'RollingWindowBacktest',
    'download_stock_data',
    'get_sp500_tickers'
]