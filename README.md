# Statistical Arbitrage Pairs Trading

Implementation of a pairs trading strategy using cointegration and mean reversion.

## Overview

This project identifies cointegrated stock pairs and trades them using a Z-score based mean reversion strategy.

**Key Components:**
- Engle-Granger cointegration test
- Hurst exponent for mean reversion strength
- Z-score trading signals with dynamic stop-loss
- Sensitivity analysis for parameter robustness

## The Strategy

### Step 1: Find Cointegrated Pairs

Use the **Engle-Granger two-step method**:
1. Run OLS regression: `Stock1 = β × Stock2 + α`
2. Test if residuals (spread) are stationary
3. If stationary → pair is cointegrated

### Step 2: Test Mean Reversion

Calculate **Hurst Exponent**:
- H < 0.5: Mean reverting (good for pairs trading)
- H = 0.5: Random walk (no predictability)
- H > 0.5: Trending (not suitable)

### Step 3: Trade the Spread

**Z-score Strategy:**
```
spread = Stock1 - β × Stock2
Z-score = (spread - mean) / std

Entry:
- Long spread when Z < -2.0 (spread too low)
- Short spread when Z > +2.0 (spread too high)

Exit:
- Z-score crosses zero (mean reversion)
- Stop-loss at ±3σ
```

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/pairs-trading.git
cd pairs-trading
pip install -r requirements.txt
```

### Run Backtest

```bash
# Default: 1 year, 10 stocks
python run.py

# Custom parameters
python run.py --days 500 --stocks 20 --threshold 2.5

# Sensitivity analysis
python run.py --sensitivity

# See all options
python run.py --help
```

### Example Output

```
RESULTS
============================================================
Total Return:          18.45%
CAGR:                  18.45%
Sharpe Ratio:            2.34
Max Drawdown:          -12.15%
Num Trades:                 15
Win Rate:               66.67%
Avg Trade P&L:       $  123.00
============================================================
```

## Project Structure

```
pairs-trading/
├── src/
│   ├── pairs_finder.py      # Engle-Granger + Hurst exponent
│   ├── trading_strategy.py  # Z-score strategy
│   └── backtest.py          # Backtesting engine
├── run.py                   # Main script
└── README.md
```

## Understanding the Code

### Engle-Granger Test (`pairs_finder.py`)

Tests if two stocks are cointegrated:
```python
def engle_granger_test(y, x):
    # Step 1: Regression
    hedge_ratio = regression(y, x)
    
    # Step 2: Test residuals for stationarity
    spread = y - hedge_ratio * x
    p_value = stationarity_test(spread)
    
    return hedge_ratio, p_value
```

Low p-value (< 0.05) = cointegrated = good pair

### Hurst Exponent (`pairs_finder.py`)

Measures mean reversion strength:
```python
def calculate_hurst_exponent(series):
    # Use R/S analysis
    # H = slope of log(R/S) vs log(lag)
    return hurst
```

### Z-Score Strategy (`trading_strategy.py`)

Entry/exit logic:
```python
def generate_signals(spread):
    zscore = (spread - mean(spread)) / std(spread)
    
    if zscore < -2.0:
        return LONG  # Buy spread
    elif zscore > 2.0:
        return SHORT  # Sell spread
    elif abs(zscore) < 0.5:
        return EXIT  # Close position
```

## Parameters to Experiment With

### Entry Threshold
- **Conservative (2.5)**: Fewer trades, stronger signals
- **Moderate (2.0)**: Balanced approach (default)
- **Aggressive (1.5)**: More trades, weaker signals

### Lookback Window
- **Short (30 days)**: Faster adaptation, more noise
- **Medium (60 days)**: Balanced (default)
- **Long (90 days)**: More stable, slower reaction

### Stop Loss
- Typically 3× standard deviation
- Prevents large losses from breakdown in cointegration

## Sensitivity Analysis

Prevents overfitting by testing multiple parameter combinations:

```bash
python run.py --sensitivity
```

Tests:
- Lookback: 30, 60, 90 days
- Threshold: 1.5, 2.0, 2.5
- Shows which parameters are most robust

## Typical Performance

On 1-year backtests (simulated data):
- **CAGR**: 15-25%
- **Sharpe Ratio**: 1.5-2.5
- **Max Drawdown**: 10-15%
- **Win Rate**: 60-70%

*Note: These are idealized results on synthetic cointegrated pairs. Real market performance will be lower due to transaction costs, slippage, and pairs breaking down.*

## Limitations

- Assumes cointegration is stable (can break down)
- No transaction costs or slippage modeled
- Simplified stop-loss (could be improved)
- Single pair at a time (could trade portfolio)

## Extensions

Potential improvements:
- Multiple pairs simultaneously
- Dynamic position sizing
- Transaction cost modeling
- Kalman filter for hedge ratio
- Out-of-sample validation

## References

1. Engle, R.F. & Granger, C.W.J. (1987). *Co-integration and Error Correction*

2. Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*

3. Hurst, H.E. (1951). *Long-term Storage Capacity of Reservoirs*

## License

MIT License

## Author

Built as a learning project to understand statistical arbitrage and mean reversion strategies.
