"""
Statistical Arbitrage Pairs Trading - Main Script
"""

import numpy as np
import argparse
from src.backtest import run_pairs_backtest, print_results, sensitivity_analysis


def main():
    parser = argparse.ArgumentParser(description='Pairs Trading Backtest')
    parser.add_argument('--days', type=int, default=252,
                       help='Number of trading days (default: 252 = 1 year)')
    parser.add_argument('--stocks', type=int, default=10,
                       help='Number of stocks in universe (default: 10)')
    parser.add_argument('--threshold', type=float, default=2.0,
                       help='Z-score entry threshold (default: 2.0)')
    parser.add_argument('--lookback', type=int, default=60,
                       help='Lookback window in days (default: 60)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--sensitivity', action='store_true',
                       help='Run sensitivity analysis on parameters')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    if args.sensitivity:
        # Run sensitivity analysis
        sensitivity_analysis(
            lookback_values=[30, 60, 90],
            threshold_values=[1.5, 2.0, 2.5],
            n_days=args.days,
            seed=args.seed
        )
    else:
        # Run single backtest
        results = run_pairs_backtest(
            n_days=args.days,
            n_stocks=args.stocks,
            entry_threshold=args.threshold,
            lookback=args.lookback,
            seed=args.seed,
            verbose=True
        )
        
        print_results(results)
        
        print("\n" + "="*60)
        print("Backtest complete!")
        print("="*60)
        print("\nTo run sensitivity analysis:")
        print("  python run.py --sensitivity")


if __name__ == '__main__':
    main()
