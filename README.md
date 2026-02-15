# Statistical Arbitrage: Cointegration Pairs Trading

A quantitative trading framework that identifies and trades cointegrated stock pairs in the S&P 500. This project implements a mean-reversion strategy using the Engle-Granger two-step method and Augmented Dickey-Fuller (ADF) tests to ensure statistical rigor.

Key Feature: To prevent lookahead bias (a common flaw in backtests), this project uses a Rolling Window approach—pairs are selected based only on past data (formation period) and traded on unseen future data (trading period).

## Features

* **Rigorous Statistical Screening:** Uses the Augmented Dickey-Fuller (ADF) test to find pairs with stationary spreads (p-value < 0.05).
* **Walk-Forward Analysis:**
    * **Formation Period (60 days):** Calibrates the hedge ratio (Beta) and tests for cointegration.
    * **Trading Period (20 days):** Executes trades out-of-sample using parameters from the formation period.
* **Dynamic Risk Management:**
    * Entry: Z-Score < -2.0 or > +2.0
    * Exit: Mean reversion (Z-Score = 0)
    * Stop Loss: Spread divergence > 4.0 standard deviations
* **Automated Data Pipeline:** Fetches, cleans, and aligns historical pricing data using yfinance and pandas.

## Tech Stack

* **Python 3.10+**
* **Pandas & NumPy:** Vectorized data manipulation and time-series alignment.
* **Statsmodels:** Econometric tests (OLS regression, ADF unit root test).
* **YFinance:** Market data ingestion.

## Project Structure

```text
├── run.py                 # Main entry point
├── requirements.txt       # Dependencies
└── src/
    ├── __init__.py        # Package exposure
    ├── backtest.py        # RollingWindowBacktest engine
    ├── data_loader.py     # Yahoo Finance data fetcher & cleaner
    ├── pairs_finder.py    # Cointegration logic (ADF test, Half-life)
    └── trading_strategy.py # Z-score signal generation & execution
