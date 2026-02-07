#!/usr/bin/env python3
"""Debug script to check why strategy isn't generating signals."""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategies.ma_alignment.strategy import MAAlignmentStrategy
from config.schema import load_and_validate_strategy_config
from engine.backtest_engine import BacktestEngine


def main():
    # Load config
    config_path = Path(__file__).parent.parent.parent / 'strategies' / 'ma_alignment' / 'config.yml'
    strategy_config = load_and_validate_strategy_config(config_path)
    
    # Create strategy
    strategy = MAAlignmentStrategy(strategy_config)
    
    # Load data
    data_path = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'BTCUSDT-15m-2020.parquet'
    
    # First, let's inspect the raw Parquet file
    print("=== INSPECTING RAW PARQUET FILE ===")
    import pandas as pd
    df_raw = pd.read_parquet(data_path)
    print(f"Raw shape: {df_raw.shape}")
    print(f"Raw index type: {type(df_raw.index)}, dtype: {df_raw.index.dtype}")
    print(f"Raw index name: {df_raw.index.name}")
    print(f"Raw columns: {df_raw.columns.tolist()}")
    if len(df_raw) > 0:
        print(f"First 3 index values: {df_raw.index[:3].tolist()}")
        print(f"First row:\n{df_raw.iloc[0]}")
        # Check if first column might be timestamp
        if len(df_raw.columns) > 0:
            first_col = df_raw.columns[0]
            print(f"\nFirst column '{first_col}':")
            print(f"  Type: {df_raw[first_col].dtype}")
            print(f"  First 3 values: {df_raw[first_col].iloc[:3].tolist()}")
            if pd.api.types.is_numeric_dtype(df_raw[first_col]):
                sample = df_raw[first_col].iloc[0]
                print(f"  Sample value: {sample}")
                if sample > 1e12:
                    print(f"  As milliseconds: {pd.to_datetime(sample, unit='ms')}")
                elif sample > 946684800:
                    print(f"  As seconds: {pd.to_datetime(sample, unit='s')}")
    
    print("\n=== LOADING WITH DATA LOADER ===")
    engine = BacktestEngine(strategy=strategy, initial_capital=10000.0)
    df_base = engine.load_data(data_path)
    
    print(f"Loaded {len(df_base)} bars")
    print(f"Date range: {df_base.index[0]} to {df_base.index[-1]}")
    print(f"Columns: {df_base.columns.tolist()}")
    if len(df_base) > 0:
        print(f"First 3 timestamps: {df_base.index[:3].tolist()}")
        print(f"Last 3 timestamps: {df_base.index[-3:].tolist()}")
    
    # Prepare data
    df_by_tf = engine.prepare_data(df_base)
    
    print(f"\nPrepared timeframes: {list(df_by_tf.keys())}")
    for tf, df in df_by_tf.items():
        print(f"  {tf}: {len(df)} bars")
    
    # Check indicators on signal timeframe
    signal_tf = strategy_config.timeframes.signal_tf
    if signal_tf not in df_by_tf or len(df_by_tf[signal_tf]) == 0:
        print(f"\nERROR: Signal timeframe ({signal_tf}) has no data!")
        print("This usually means:")
        print("  1. The datetime index is incorrect (check date range above)")
        print("  2. The data resampling failed")
        print("  3. The base timeframe doesn't match the data")
        return
    
    df_signal = df_by_tf[signal_tf].copy()
    
    # Calculate indicators first (indicators should already be calculated by prepare_data)
    # But let's recalculate to ensure they're there
    if 'ema5' not in df_signal.columns:
        df_signal = strategy.get_indicators(df_signal)
    
    print(f"\nSignal timeframe ({signal_tf}) sample:")
    print(f"All columns: {df_signal.columns.tolist()}")
    available_cols = ['close']
    indicator_cols = ['ema5', 'ema15', 'ema30', 'sma50', 'ema100', 'macd_hist']
    for col in indicator_cols:
        if col in df_signal.columns:
            available_cols.append(col)
    
    if len(available_cols) > 1:
        print(df_signal[available_cols].tail(10))
    else:
        print("WARNING: No indicator columns found!")
        print(f"Available columns: {df_signal.columns.tolist()}")
    
    # Check alignment conditions
    print(f"\nChecking alignment conditions...")
    alignment_count = 0
    long_alignment_count = 0
    short_alignment_count = 0
    
    for idx, row in df_signal.iterrows():
        required_mas = [name for name, config in strategy_config.moving_averages.items() if config.enabled]
        if not all(name in row and pd.notna(row[name]) for name in required_mas):
            continue
        
        if 'macd_hist' not in row or pd.isna(row['macd_hist']):
            continue
        
        if strategy._check_alignment(row, 'long'):
            long_alignment_count += 1
        if strategy._check_alignment(row, 'short'):
            short_alignment_count += 1
        
        alignment_count += 1
    
    print(f"Bars with all indicators: {alignment_count}")
    print(f"Bars with long alignment: {long_alignment_count}")
    print(f"Bars with short alignment: {short_alignment_count}")
    
    # Check if we have enough data for indicators
    print(f"\nIndicator availability:")
    for col in indicator_cols:
        if col in df_signal.columns:
            non_null = df_signal[col].notna().sum()
            print(f"  {col}: {non_null}/{len(df_signal)} bars have values")
            if non_null == 0:
                print(f"    WARNING: Column exists but all values are NaN!")
        else:
            print(f"  {col}: NOT CALCULATED (column missing)")
    
    # Check if EMA100 specifically has issues
    if 'ema100' in df_signal.columns:
        ema100_non_null = df_signal['ema100'].notna().sum()
        if ema100_non_null == 0:
            print(f"\n  EMA100 issue: Column exists but no valid values.")
            print(f"  This is expected if you have fewer than 100 bars ({len(df_signal)} bars available).")
            print(f"  EMA100 needs at least 100 periods to have meaningful values.")
    
    # Generate signals
    try:
        signals_df = strategy.generate_signals(df_by_tf)
        
        print(f"\nGenerated signals: {len(signals_df)}")
        if len(signals_df) > 0:
            print(signals_df.head(10))
        else:
            print("No signals generated. Possible reasons:")
            print("  1. Alignment conditions too strict (all 5 MAs must align)")
            print("  2. MACD confirmation not met")
            print("  3. Not enough data for indicators")
            print("\nTry:")
            print("  - Reducing MACD bar requirements in config.yml")
            print("  - Checking if MAs are actually aligning in the data")
            print("  - Using more data or different time period")
    except Exception as e:
        print(f"\nERROR generating signals: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

