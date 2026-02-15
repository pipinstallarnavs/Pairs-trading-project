import pandas as pd
import numpy as np
from src.pairs_finder import find_cointegrated_pairs
from src.trading_strategy import PairsTradingStrategy

class RollingWindowBacktest:
    def __init__(
        self, 
        data: pd.DataFrame, 
        formation_period: int = 60,  # Days to find pairs
        trading_period: int = 20     # Days to trade them
    ):
        self.data = data
        self.formation_period = formation_period
        self.trading_period = trading_period
        self.results = []
        self.equity_curve = [10000.0] # Start with $10k
        
    def run(self):
        total_days = len(self.data)
        
        # Walk forward
        # Step size = trading_period
        for t in range(self.formation_period, total_days, self.trading_period):
            
            # Define indices
            form_start = t - self.formation_period
            form_end = t
            trade_end = min(t + self.trading_period, total_days)
            
            if trade_end <= form_end: break
            
            print(f"[{self.data.index[form_end].date()}] Rebalancing... ", end="")
            
            # 1. Formation: Find pairs on PAST data
            form_data = self.data.iloc[form_start:form_end]
            pairs = find_cointegrated_pairs(form_data, p_threshold=0.05)
            
            if not pairs:
                print("No pairs found.")
                continue
                
            # Pick the single best pair (lowest p-value)
            # Strategy: You could trade top 3 or 5, but let's stick to 1 for clarity
            best_pair = pairs[0]
            print(f"Best Pair: {best_pair.stock1}-{best_pair.stock2} (p={best_pair.p_value:.4f})")
            
            # 2. Trading: Execute on FUTURE data
            trade_data = self.data.iloc[form_end:trade_end]
            
            s1_prices = trade_data[best_pair.stock1]
            s2_prices = trade_data[best_pair.stock2]
            
            # Construct spread
            # Note: We use the hedge ratio found in FORMATION
            spread = s1_prices - best_pair.hedge_ratio * s2_prices
            
            # Run strategy
            strategy = PairsTradingStrategy()
            trades = strategy.run_segment(
                spread_series=spread,
                spread_mean=best_pair.spread_mean,
                spread_std=best_pair.spread_std,
                ticker1=best_pair.stock1,
                ticker2=best_pair.stock2
            )
            
            self.results.extend(trades)
            
    def get_summary(self):
        if not self.results:
            return "No trades generated."
            
        df_trades = pd.DataFrame([vars(t) for t in self.results])
        total_pnl = df_trades['pnl'].sum()
        win_rate = len(df_trades[df_trades['pnl'] > 0]) / len(df_trades)
        
        print("\n" + "="*40)
        print("BACKTEST SUMMARY")
        print("="*40)
        print(f"Total Trades: {len(df_trades)}")
        print(f"Total PnL:    ${total_pnl:.2f}")
        print(f"Win Rate:     {win_rate:.1%}")
        print(f"Avg PnL:      ${df_trades['pnl'].mean():.2f}")
        print("="*40)
        
        return df_trades