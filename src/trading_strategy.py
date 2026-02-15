"""
Z-Score Based Pairs Trading Strategy

Strategy Logic:
- Calculate rolling Z-score of the spread
- Enter long spread when Z < -threshold (spread too low)
- Enter short spread when Z > +threshold (spread too high)
- Exit when Z crosses zero (spread reverts to mean)
- Dynamic stop-loss based on volatility
"""

import numpy as np
from typing import Tuple, List
from dataclasses import dataclass


@dataclass
class Trade:
    """Record of a trade"""
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    position: int  # +1 for long, -1 for short
    pnl: float


class PairsTradingStrategy:
    """
    Z-score based pairs trading strategy
    
    Entry: |Z-score| > entry_threshold
    Exit: Z-score crosses zero or hits stop-loss
    """
    
    def __init__(
        self,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        stop_loss_sigma: float = 3.0,
        lookback: int = 60,
        position_size: float = 1.0
    ):
        """
        Args:
            entry_threshold: Z-score level to enter (typically 2.0)
            exit_threshold: Z-score level to exit (typically 0.0-0.5)
            stop_loss_sigma: Stop loss in standard deviations
            lookback: Window for calculating mean and std
            position_size: Size of each trade
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_sigma = stop_loss_sigma
        self.lookback = lookback
        self.position_size = position_size
        
        # State
        self.position = 0  # 0 = flat, 1 = long spread, -1 = short spread
        self.entry_price = 0.0
        self.entry_time = 0
        self.trades: List[Trade] = []
        
    def calculate_zscore(
        self,
        spread: np.ndarray,
        lookback: int = None
    ) -> np.ndarray:
        """
        Calculate rolling Z-score of spread
        
        Z = (spread - rolling_mean) / rolling_std
        
        Args:
            spread: Spread time series
            lookback: Rolling window (uses self.lookback if None)
            
        Returns:
            Z-score time series
        """
        if lookback is None:
            lookback = self.lookback
        
        zscore = np.zeros(len(spread))
        
        for i in range(lookback, len(spread)):
            window = spread[i-lookback:i]
            mean = np.mean(window)
            std = np.std(window)
            
            if std > 0:
                zscore[i] = (spread[i] - mean) / std
            else:
                zscore[i] = 0
        
        return zscore
    
    def generate_signals(
        self,
        spread: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate trading signals from spread
        
        Returns:
            (entry_signals, exit_signals)
            entry_signals: +1 = go long, -1 = go short, 0 = no entry
            exit_signals: 1 = exit position, 0 = hold
        """
        zscore = self.calculate_zscore(spread)
        
        entry_signals = np.zeros(len(spread))
        exit_signals = np.zeros(len(spread))
        
        for i in range(len(spread)):
            # Entry signals
            if zscore[i] < -self.entry_threshold:
                entry_signals[i] = 1  # Long spread (buy stock1, sell stock2)
            elif zscore[i] > self.entry_threshold:
                entry_signals[i] = -1  # Short spread (sell stock1, buy stock2)
            
            # Exit signals
            if abs(zscore[i]) < self.exit_threshold:
                exit_signals[i] = 1  # Mean reversion
        
        return entry_signals, exit_signals
    
    def check_stop_loss(
        self,
        spread: np.ndarray,
        current_idx: int
    ) -> bool:
        """
        Check if stop-loss is hit
        
        Stop loss = entry_price ± stop_loss_sigma * std
        
        Returns:
            True if stop-loss hit
        """
        if self.position == 0:
            return False
        
        # Calculate rolling std
        window = spread[max(0, current_idx-self.lookback):current_idx]
        std = np.std(window) if len(window) > 0 else 0
        
        stop_distance = self.stop_loss_sigma * std
        
        if self.position == 1:  # Long spread
            # Stop if spread drops too much
            return spread[current_idx] < (self.entry_price - stop_distance)
        else:  # Short spread
            # Stop if spread rises too much
            return spread[current_idx] > (self.entry_price + stop_distance)
    
    def backtest(
        self,
        spread: np.ndarray,
        verbose: bool = False
    ) -> Tuple[List[Trade], np.ndarray]:
        """
        Backtest the strategy on spread data
        
        Args:
            spread: Spread time series
            verbose: Print trade details
            
        Returns:
            (trades, pnl_series)
        """
        # Generate signals
        entry_signals, exit_signals = self.generate_signals(spread)
        
        # Reset state
        self.position = 0
        self.trades = []
        pnl_series = np.zeros(len(spread))
        current_pnl = 0
        
        for i in range(self.lookback, len(spread)):
            # Check stop-loss first
            if self.check_stop_loss(spread, i):
                if verbose:
                    print(f"Stop-loss hit at {i}")
                
                # Exit on stop-loss
                pnl = (spread[i] - self.entry_price) * self.position * self.position_size
                current_pnl += pnl
                
                trade = Trade(
                    entry_time=self.entry_time,
                    exit_time=i,
                    entry_price=self.entry_price,
                    exit_price=spread[i],
                    position=self.position,
                    pnl=pnl
                )
                self.trades.append(trade)
                
                self.position = 0
            
            # Check exit signals
            if self.position != 0 and exit_signals[i] == 1:
                # Exit position
                pnl = (spread[i] - self.entry_price) * self.position * self.position_size
                current_pnl += pnl
                
                if verbose:
                    print(f"Exit at {i}: PnL = {pnl:.2f}")
                
                trade = Trade(
                    entry_time=self.entry_time,
                    exit_time=i,
                    entry_price=self.entry_price,
                    exit_price=spread[i],
                    position=self.position,
                    pnl=pnl
                )
                self.trades.append(trade)
                
                self.position = 0
            
            # Check entry signals (only if flat)
            if self.position == 0 and entry_signals[i] != 0:
                self.position = int(entry_signals[i])
                self.entry_price = spread[i]
                self.entry_time = i
                
                if verbose:
                    print(f"Enter {'long' if self.position == 1 else 'short'} at {i}")
            
            pnl_series[i] = current_pnl
        
        return self.trades, pnl_series
