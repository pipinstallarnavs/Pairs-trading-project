import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as ts
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class PairStats:
    stock1: str
    stock2: str
    hedge_ratio: float
    p_value: float
    half_life: float
    spread_mean: float
    spread_std: float

def calculate_half_life(spread: np.ndarray) -> float:
    """Calculate half-life of mean reversion using OLS on lagged spread"""
    spread_lag = np.roll(spread, 1)
    spread_ret = spread - spread_lag
    spread_ret = spread_ret[1:]
    spread_lag = spread_lag[1:]
    
    # Regression: d(spread) ~ theta * spread_lag
    X = spread_lag
    y = spread_ret
    try:
        res = np.linalg.lstsq(X[:, None], y, rcond=None)
        theta = res[0][0]
        if theta >= 0: return np.inf # Not mean reverting
        return -np.log(2) / theta
    except:
        return np.inf

def find_cointegrated_pairs(
    data: pd.DataFrame, 
    p_threshold: float = 0.05
) -> List[PairStats]:
    """
    Identify pairs that are cointegrated (p_value < threshold).
    """
    n = data.shape[1]
    keys = data.columns
    pairs = []
    
    # Loop through all possible pairs
    for i in range(n):
        for j in range(i + 1, n):
            s1 = data.iloc[:, i]
            s2 = data.iloc[:, j]
            
            # Engle-Granger Test
            # The test returns: t-stat, p-value, crit-values
            try:
                # coint_t, p_value, crit_val = ts.coint(s1, s2)
                # Using simpler AD-Fuller on residuals for speed/stability
                
                # 1. Get Hedge Ratio (OLS)
                X = np.vstack([s2, np.ones(len(s2))]).T
                beta, alpha = np.linalg.lstsq(X, s1, rcond=None)[0]
                
                # 2. Get Spread
                spread = s1 - beta * s2 - alpha
                
                # 3. Test Stationarity
                adf_res = ts.adfuller(spread)
                p_value = adf_res[1]
                
                if p_value < p_threshold:
                    hl = calculate_half_life(spread)
                    # Filter for reasonable half-life (e.g., 1 to 30 days)
                    if 1 <= hl <= 40:
                        pairs.append(PairStats(
                            stock1=keys[i],
                            stock2=keys[j],
                            hedge_ratio=beta,
                            p_value=p_value,
                            half_life=hl,
                            spread_mean=np.mean(spread),
                            spread_std=np.std(spread)
                        ))
            except Exception as e:
                continue

    # Sort by strongest cointegration (lowest p-value)
    pairs.sort(key=lambda x: x.p_value)
    return pairs