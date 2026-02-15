import yfinance as yf
import pandas as pd
from typing import List

def get_sp500_tickers(top_n: int = 20) -> List[str]:
    """Get top N S&P 500 stocks by market cap"""
    # Hardcoded list to ensure we always have valid tickers
    sp500_top = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
        'META', 'TSLA', 'BRK-B', 'LLY', 'V',
        'UNH', 'XOM', 'JPM', 'JNJ', 'WMT',
        'MA', 'PG', 'AVGO', 'HD', 'CVX',
        'MRK', 'ABBV', 'COST', 'PEP', 'KO',
        'ADBE', 'CRM', 'NFLX', 'BAC', 'MCD',
        'CSCO', 'TMO', 'ACN', 'LIN', 'AMD',
        'ORCL', 'NKE', 'DHR', 'DIS', 'WFC',
        'ABT', 'VZ', 'CMCSA', 'TXN', 'INTC',
        'PM', 'NEE', 'COP', 'BMY', 'UPS'
    ]
    return sp500_top[:top_n]

def download_stock_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download stock prices and return a clean DataFrame.
    """
    print(f"Downloading data for {len(tickers)} tickers...")
    
    try:
        # auto_adjust=False ensures we get the raw columns we expect
        raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=False)
    except Exception as e:
        raise RuntimeError(f"Failed to download data: {e}")
    
    if raw_data.empty:
        raise ValueError("Downloaded data is empty. Check your internet connection or ticker symbols.")

    # Robust column selection: Check for 'Adj Close', fallback to 'Close'
    if 'Adj Close' in raw_data.columns:
        data = raw_data['Adj Close']
    elif 'Close' in raw_data.columns:
        print("Warning: 'Adj Close' not found. Using 'Close' instead.")
        data = raw_data['Close']
    else:
        # If neither exists, print available columns for debugging
        cols = raw_data.columns.levels[0] if isinstance(raw_data.columns, pd.MultiIndex) else raw_data.columns
        raise KeyError(f"Neither 'Adj Close' nor 'Close' found. Available columns: {cols}")

    # Handle single-ticker case (returns Series, convert to DataFrame)
    if isinstance(data, pd.Series):
        if len(tickers) == 1:
            data = data.to_frame(name=tickers[0])
        else:
            data = data.to_frame()

    # Drop columns that failed to download (all NaNs)
    data = data.dropna(axis=1, how='all')
    
    # Forward fill missing data (e.g. holidays)
    data = data.ffill().dropna()
    
    if data.empty:
        raise ValueError("Data is empty after cleaning (all NaNs).")

    print(f"Data shape: {data.shape} (Dates x Tickers)")
    return data