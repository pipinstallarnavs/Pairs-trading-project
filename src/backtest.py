"""
Backtesting Framework for Pairs Trading

Generates synthetic stock data and calculates performance metrics
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List
from src.pairs_finder import find_cointegrated_pairs, PairStats
from src.trading_strategy import PairsTradingStrategy, Trade


@dataclass
class BacktestResults:
    """Results from backtest"""
    total_return: float
    cagr: float
    sharpe_ratio: float
    max_drawdown: float
    num_trades: int
    win_rate: float
    avg_trade_pnl: float
    trades: List[Trade]
    pnl_series: np.ndarray


def generate_cointegrated_pairs(
    n_days: int = 252,
    n_stocks: int = 10,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic stock prices with some cointegrated pairs
    
    Creates pairs that are genuinely cointegrated for testing
    
    Args:
        n_days: Number of trading days
        n_days: Number of stocks to generate
        seed: Random seed
        
    Returns:
        Dict of {ticker: price_series}
    """
    np.random.seed(seed)
    prices = {}
    
    # Generate base stocks (half the universe)
    for i in range(n_stocks // 2):
        ticker = f"STOCK_{chr(65+i)}"
        
        # Random walk with drift
        returns = np.random.normal(0.0005, 0.02, n_days)
        prices[ticker] = 100 * np.exp(np.cumsum(returns))
    
    # Generate cointegrated pairs (other half)
    base_tickers = list(prices.keys())
    for i, base_ticker in enumerate(base_tickers):
        pair_ticker = f"STOCK_{chr(65+len(base_tickers)+i)}"
        
        base_price = prices[base_ticker]
        
        # Create cointegrated pair with VERY mean-reverting noise
        beta = 1.0  # Use 1:1 ratio for simplicity
        
        # Ornstein-Uhlenbeck process (strong mean reversion)
        noise = np.zeros(n_days)
        noise[0] = 0
        mean_reversion_speed = 0.3  # Higher = faster mean reversion
        volatility = 1.0
        
        for t in range(1, n_days):
            # OU process: dx = -theta*(x - mu)*dt + sigma*dW
            dt = 1
            dW = np.random.normal(0, np.sqrt(dt))
            noise[t] = noise[t-1] - mean_reversion_speed * noise[t-1] * dt + volatility * dW
        
        prices[pair_ticker] = beta * base_price + noise + 10
    
    return prices


def calculate_metrics(
    trades: List[Trade],
    pnl_series: np.ndarray,
    initial_capital: float = 10000,
    risk_free_rate: float = 0.02
) -> BacktestResults:
    """
    Calculate performance metrics
    
    Args:
        trades: List of executed trades
        pnl_series: P&L time series
        initial_capital: Starting capital
        risk_free_rate: Annual risk-free rate
        
    Returns:
        BacktestResults with all metrics
    """
    if len(trades) == 0:
        return BacktestResults(
            total_return=0,
            cagr=0,
            sharpe_ratio=0,
            max_drawdown=0,
            num_trades=0,
            win_rate=0,
            avg_trade_pnl=0,
            trades=[],
            pnl_series=pnl_series
        )
    
    # Total return
    total_pnl = pnl_series[-1]
    total_return = total_pnl / initial_capital
    
    # CAGR (assuming 252 trading days per year)
    n_years = len(pnl_series) / 252
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    # Sharpe Ratio
    returns = np.diff(pnl_series) / initial_capital
    sharpe_ratio = (np.mean(returns) - risk_free_rate/252) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    
    # Max Drawdown
    cummax = np.maximum.accumulate(pnl_series)
    drawdown = pnl_series - cummax
    max_drawdown = np.min(drawdown) / initial_capital
    
    # Trade statistics
    winning_trades = [t for t in trades if t.pnl > 0]
    win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0
    avg_trade_pnl = np.mean([t.pnl for t in trades]) if trades else 0
    
    return BacktestResults(
        total_return=total_return,
        cagr=cagr,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        num_trades=len(trades),
        win_rate=win_rate,
        avg_trade_pnl=avg_trade_pnl,
        trades=trades,
        pnl_series=pnl_series
    )


def run_pairs_backtest(
    n_days: int = 252,
    n_stocks: int = 10,
    entry_threshold: float = 2.0,
    lookback: int = 60,
    seed: int = 42,
    verbose: bool = True
) -> BacktestResults:
    """
    Run complete pairs trading backtest
    
    Steps:
    1. Generate stock data
    2. Find cointegrated pairs
    3. Trade the best pair
    4. Calculate metrics
    
    Args:
        n_days: Number of trading days
        n_stocks: Number of stocks in universe
        entry_threshold: Z-score entry level
        lookback: Rolling window for statistics
        seed: Random seed
        verbose: Print progress
        
    Returns:
        BacktestResults
    """
    if verbose:
        print("="*60)
        print("PAIRS TRADING BACKTEST")
        print("="*60)
        print(f"Universe: {n_stocks} stocks")
        print(f"Period: {n_days} days")
        print(f"Entry threshold: {entry_threshold}")
        print(f"Lookback: {lookback}")
        print()
    
    # Generate data
    if verbose:
        print("Generating stock data...")
    prices = generate_cointegrated_pairs(n_days, n_stocks, seed)
    
    # Find pairs
    if verbose:
        print("Finding cointegrated pairs...")
    pairs = find_cointegrated_pairs(prices, p_threshold=0.20, hurst_threshold=0.55)
    
    if len(pairs) == 0:
        if verbose:
            print("No cointegrated pairs found!")
        return BacktestResults(
            total_return=0, cagr=0, sharpe_ratio=0, max_drawdown=0,
            num_trades=0, win_rate=0, avg_trade_pnl=0,
            trades=[], pnl_series=np.zeros(n_days)
        )
    
    # Trade the best pair
    best_pair = pairs[0]
    if verbose:
        print(f"\nTrading pair: {best_pair.stock1} / {best_pair.stock2}")
        print(f"  Hedge ratio: {best_pair.hedge_ratio:.3f}")
        print(f"  P-value: {best_pair.p_value:.4f}")
        print(f"  Hurst: {best_pair.hurst:.3f}")
        print(f"  Half-life: {best_pair.half_life:.1f} days")
        print()
    
    # Calculate spread
    price1 = prices[best_pair.stock1]
    price2 = prices[best_pair.stock2]
    spread = price1 - best_pair.hedge_ratio * price2
    
    # Run strategy
    if verbose:
        print("Running strategy...")
    strategy = PairsTradingStrategy(
        entry_threshold=entry_threshold,
        exit_threshold=0.5,
        stop_loss_sigma=3.0,
        lookback=lookback
    )
    
    trades, pnl_series = strategy.backtest(spread, verbose=False)
    
    # Calculate metrics
    results = calculate_metrics(trades, pnl_series)
    
    return results


def print_results(results: BacktestResults):
    """Print backtest results"""
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Total Return:     {results.total_return*100:>10.2f}%")
    print(f"CAGR:             {results.cagr*100:>10.2f}%")
    print(f"Sharpe Ratio:     {results.sharpe_ratio:>14.2f}")
    print(f"Max Drawdown:     {results.max_drawdown*100:>10.2f}%")
    print(f"Num Trades:       {results.num_trades:>14}")
    print(f"Win Rate:         {results.win_rate*100:>10.2f}%")
    print(f"Avg Trade P&L:    ${results.avg_trade_pnl:>13.2f}")
    print("="*60)


def sensitivity_analysis(
    lookback_values: List[int] = [30, 60, 90],
    threshold_values: List[float] = [1.5, 2.0, 2.5],
    n_days: int = 252,
    seed: int = 42
):
    """
    Run sensitivity analysis on parameters
    
    Tests different lookback windows and entry thresholds
    to prevent overfitting
    """
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS")
    print("="*60)
    
    results_grid = []
    
    for lookback in lookback_values:
        for threshold in threshold_values:
            print(f"\nTesting: lookback={lookback}, threshold={threshold}")
            
            results = run_pairs_backtest(
                n_days=n_days,
                entry_threshold=threshold,
                lookback=lookback,
                seed=seed,
                verbose=False
            )
            
            results_grid.append({
                'lookback': lookback,
                'threshold': threshold,
                'cagr': results.cagr,
                'sharpe': results.sharpe_ratio,
                'trades': results.num_trades
            })
            
            print(f"  CAGR: {results.cagr*100:.2f}%, Sharpe: {results.sharpe_ratio:.2f}, Trades: {results.num_trades}")
    
    print("\n" + "="*60)
    print("Best parameters:")
    best = max(results_grid, key=lambda x: x['sharpe'])
    print(f"  Lookback: {best['lookback']}")
    print(f"  Threshold: {best['threshold']}")
    print(f"  CAGR: {best['cagr']*100:.2f}%")
    print(f"  Sharpe: {best['sharpe']:.2f}")
    print("="*60)
