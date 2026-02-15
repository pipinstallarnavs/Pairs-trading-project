from src.data_loader import get_sp500_tickers, download_stock_data
from src.backtest import RollingWindowBacktest
import pandas as pd

def main():
    # 1. Configuration
    START_DATE = "2022-01-01"
    END_DATE = "2024-01-01"
    
    # 2. Get Data
    tickers = get_sp500_tickers(top_n=30) # Top 30 stocks
    data = download_stock_data(tickers, START_DATE, END_DATE)
    
    # 3. Initialize Backtest
    # We use a 3-month formation period and 1-month trading period
    backtester = RollingWindowBacktest(
        data=data,
        formation_period=60, # 60 trading days (~3 months)
        trading_period=20    # 20 trading days (~1 month)
    )
    
    # 4. Run
    backtester.run()
    
    # 5. Results
    trades = backtester.get_summary()
    
    if isinstance(trades, pd.DataFrame):
        trades.to_csv("backtest_trades.csv")
        print("Trades saved to backtest_trades.csv")

if __name__ == "__main__":
    main()