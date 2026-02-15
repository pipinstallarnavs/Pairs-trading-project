import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List

@dataclass
class Trade:
    ticker1: str
    ticker2: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_spread: float
    exit_spread: float
    pnl: float
    return_pct: float

class PairsTradingStrategy:
    def __init__(self, entry_z=2.0, exit_z=0.0, stop_loss_z=4.0):
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_loss_z = stop_loss_z
        
        # State
        self.position = 0 # 1 = Long Spread, -1 = Short Spread
        self.entry_price = 0.0
        self.entry_idx = None
        self.trades = []
        
    def run_segment(self, 
                   spread_series: pd.Series, 
                   spread_mean: float, 
                   spread_std: float,
                   ticker1: str,
                   ticker2: str) -> List[Trade]:
        """
        Run strategy on a specific segment of data using FIXED parameters 
        from the formation period (to avoid lookahead bias).
        """
        self.trades = []
        # Reset position at start of new segment? 
        # For simplicity, we assume we flatten at end of every segment.
        self.position = 0 
        
        # Calculate Z-score for this whole segment
        z_scores = (spread_series - spread_mean) / spread_std
        
        for date, z_score in z_scores.items():
            price = spread_series.loc[date]
            
            # Check Exit (Profit Take or Stop Loss)
            if self.position != 0:
                exit_signal = False
                
                # Profit Take: Crosses mean
                if self.position == 1 and z_score >= -self.exit_z: exit_signal = True
                elif self.position == -1 and z_score <= self.exit_z: exit_signal = True
                
                # Stop Loss: Diverges too far
                if abs(z_score) > self.stop_loss_z: exit_signal = True
                
                if exit_signal:
                    pnl = (price - self.entry_price) * self.position
                    ret = pnl / abs(self.entry_price) if self.entry_price != 0 else 0
                    
                    self.trades.append(Trade(
                        ticker1, ticker2, self.entry_idx, date, 
                        self.entry_price, price, pnl, ret
                    ))
                    self.position = 0
            
            # Check Entry
            if self.position == 0:
                if z_score < -self.entry_z:
                    self.position = 1 # Long Spread
                    self.entry_price = price
                    self.entry_idx = date
                elif z_score > self.entry_z:
                    self.position = -1 # Short Spread
                    self.entry_price = price
                    self.entry_idx = date
                    
        return self.trades