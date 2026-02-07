#!/usr/bin/env python3
"""Analyze random entry sample trades to verify p-value calculation."""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

def analyze_samples(csv_path: str):
    """Analyze sample trades and compare baseline vs random PnLs."""
    df = pd.read_csv(csv_path)
    
    print("=" * 80)
    print("RANDOM ENTRY SAMPLE ANALYSIS")
    print("=" * 80)
    print(f"\nTotal sample trades: {len(df)}")
    print(f"Iterations in sample: {df['iteration'].nunique()}")
    print(f"Unique iterations: {sorted(df['iteration'].unique())}")
    
    # Calculate aggregate metrics
    baseline_total_pnl = df['baseline_pnl'].sum()
    random_total_pnl = df['random_pnl'].sum()
    
    print(f"\n{'='*80}")
    print("AGGREGATE METRICS (from sample)")
    print(f"{'='*80}")
    print(f"Baseline Total PnL: {baseline_total_pnl:.2f}")
    print(f"Random Total PnL:   {random_total_pnl:.2f}")
    print(f"Difference:         {baseline_total_pnl - random_total_pnl:.2f}")
    print(f"Baseline Better:    {baseline_total_pnl > random_total_pnl}")
    
    # Per-trade comparison
    print(f"\n{'='*80}")
    print("PER-TRADE COMPARISON")
    print(f"{'='*80}")
    df['baseline_better'] = df['baseline_pnl'] > df['random_pnl']
    df['random_better'] = df['random_pnl'] > df['baseline_pnl']
    df['equal'] = np.isclose(df['baseline_pnl'], df['random_pnl'], atol=1e-6)
    
    baseline_wins = df['baseline_better'].sum()
    random_wins = df['random_better'].sum()
    ties = df['equal'].sum()
    
    print(f"Baseline better: {baseline_wins} trades ({baseline_wins/len(df)*100:.1f}%)")
    print(f"Random better:   {random_wins} trades ({random_wins/len(df)*100:.1f}%)")
    print(f"Ties:            {ties} trades ({ties/len(df)*100:.1f}%)")
    
    # Show trades where random significantly outperformed baseline
    print(f"\n{'='*80}")
    print("TRADES WHERE RANDOM SIGNIFICANTLY OUTPERFORMED BASELINE")
    print(f"{'='*80}")
    significant_random_wins = df[df['random_better'] & (df['random_pnl'] - df['baseline_pnl'] > 10)]
    if len(significant_random_wins) > 0:
        print(f"\nFound {len(significant_random_wins)} trades where random outperformed by >$10:")
        for idx, row in significant_random_wins.iterrows():
            diff = row['random_pnl'] - row['baseline_pnl']
            print(f"  Trade {row['baseline_trade_index']:2d}: baseline={row['baseline_pnl']:8.2f}, "
                  f"random={row['random_pnl']:8.2f}, diff={diff:8.2f}")
    else:
        print("None found.")
    
    # Show trades where baseline significantly outperformed random
    print(f"\n{'='*80}")
    print("TRADES WHERE BASELINE SIGNIFICANTLY OUTPERFORMED RANDOM")
    print(f"{'='*80}")
    significant_baseline_wins = df[df['baseline_better'] & (df['baseline_pnl'] - df['random_pnl'] > 10)]
    if len(significant_baseline_wins) > 0:
        print(f"\nFound {len(significant_baseline_wins)} trades where baseline outperformed by >$10:")
        for idx, row in significant_baseline_wins.iterrows():
            diff = row['baseline_pnl'] - row['random_pnl']
            print(f"  Trade {row['baseline_trade_index']:2d}: baseline={row['baseline_pnl']:8.2f}, "
                  f"random={row['random_pnl']:8.2f}, diff={diff:8.2f}")
    else:
        print("None found.")
    
    # Position size analysis
    print(f"\n{'='*80}")
    print("POSITION SIZE ANALYSIS")
    print(f"{'='*80}")
    size_match = np.isclose(df['baseline_position_size'], df['random_position_size'], rtol=1e-6)
    print(f"Position sizes match: {size_match.sum()} / {len(df)} ({size_match.sum()/len(df)*100:.1f}%)")
    
    if not size_match.all():
        mismatches = df[~size_match]
        print(f"\nTrades with position size mismatches ({len(mismatches)}):")
        for idx, row in mismatches.head(10).iterrows():
            print(f"  Trade {row['baseline_trade_index']:2d}: baseline={row['baseline_position_size']:.6f}, "
                  f"random={row['random_position_size']:.6f}")
    
    # Stop distance analysis
    print(f"\n{'='*80}")
    print("STOP DISTANCE ANALYSIS")
    print(f"{'='*80}")
    stop_match = np.isclose(df['baseline_stop_distance'], df['random_stop_distance'], rtol=1e-6)
    print(f"Stop distances match: {stop_match.sum()} / {len(df)} ({stop_match.sum()/len(df)*100:.1f}%)")
    
    if not stop_match.all():
        mismatches = df[~stop_match]
        print(f"\nTrades with stop distance mismatches ({len(mismatches)}):")
        for idx, row in mismatches.head(10).iterrows():
            print(f"  Trade {row['baseline_trade_index']:2d}: baseline={row['baseline_stop_distance']:.2f}, "
                  f"random={row['random_stop_distance']:.2f}")
    
    # P-value interpretation
    print(f"\n{'='*80}")
    print("P-VALUE INTERPRETATION")
    print(f"{'='*80}")
    print("\nThe p-value is calculated across ALL Monte Carlo iterations (typically 1000),")
    print("not just this single sample iteration.")
    print("\nThis CSV shows only ONE iteration (iteration 0) out of many.")
    print("\nFor p-value = 0.0000 to be correct:")
    print("  - Baseline's TOTAL final_pnl must exceed ALL random iterations' total final_pnl")
    print("  - This means across 1000 iterations, baseline always had higher aggregate PnL")
    print("\nHowever, if individual trades show mixed results (some random better, some baseline better),")
    print("the aggregate metrics might still favor baseline if:")
    print("  1. Baseline's large wins outweigh random's small wins")
    print("  2. Baseline's losses are smaller than random's losses")
    print("  3. The sample iteration shown is not representative of the full distribution")
    
    # Calculate what p-value would be if this were the only iteration
    if baseline_total_pnl > random_total_pnl:
        sample_p_value = 0.0  # Baseline better
    elif baseline_total_pnl < random_total_pnl:
        sample_p_value = 1.0  # Random better
    else:
        sample_p_value = 0.5  # Equal
    
    print(f"\nIf this sample iteration were the only one:")
    print(f"  Sample p-value would be: {sample_p_value:.4f}")
    print(f"  (0.0 = baseline better, 1.0 = random better)")
    
    return df

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_random_entry_samples.py <csv_path>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    if not Path(csv_path).exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    
    df = analyze_samples(csv_path)

