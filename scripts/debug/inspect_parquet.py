#!/usr/bin/env python3
"""Inspect Parquet file structure in detail."""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def main():
    data_path = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'BTCUSDT-15m-2020.parquet'
    
    print(f"Loading: {data_path}")
    print(f"File exists: {data_path.exists()}")
    
    # Load raw Parquet
    df = pd.read_parquet(data_path)
    
    print(f"\n=== BASIC INFO ===")
    print(f"Shape: {df.shape}")
    print(f"Index type: {type(df.index)}")
    print(f"Index dtype: {df.index.dtype}")
    print(f"Index name: {df.index.name}")
    print(f"Index first 5: {df.index[:5].tolist()}")
    
    print(f"\n=== COLUMNS ===")
    print(f"Number of columns: {len(df.columns)}")
    print(f"First 10 column names: {df.columns[:10].tolist()}")
    print(f"Last 10 column names: {df.columns[-10].tolist() if len(df.columns) > 10 else df.columns.tolist()}")
    
    # Check if first column name is numeric (timestamp)
    first_col = df.columns[0]
    print(f"\n=== FIRST COLUMN ANALYSIS ===")
    print(f"First column name: '{first_col}'")
    print(f"First column type: {type(first_col)}")
    try:
        first_col_as_num = float(str(first_col))
        print(f"First column as number: {first_col_as_num}")
        if first_col_as_num > 1e12:
            print(f"As datetime (ms): {pd.to_datetime(first_col_as_num, unit='ms')}")
        elif first_col_as_num > 946684800:
            print(f"As datetime (s): {pd.to_datetime(first_col_as_num, unit='s')}")
    except:
        print("First column name is not numeric")
    
    print(f"\n=== FIRST ROW ANALYSIS ===")
    print(f"First row index: {df.index[0]}")
    first_row = df.iloc[0]
    print(f"First row (first 10 values):")
    for i, (col, val) in enumerate(first_row.items()):
        if i < 10:
            print(f"  {col}: {val} (type: {type(val).__name__})")
    
    # Check for common OHLCV column names
    print(f"\n=== OHLCV COLUMN DETECTION ===")
    ohlcv_keywords = ['open', 'high', 'low', 'close', 'volume', 'Open', 'High', 'Low', 'Close', 'Volume']
    found_ohlcv = {}
    for col in df.columns:
        col_lower = str(col).lower()
        for keyword in ohlcv_keywords:
            if keyword.lower() in col_lower:
                if keyword.lower() not in found_ohlcv:
                    found_ohlcv[keyword.lower()] = []
                found_ohlcv[keyword.lower()].append(col)
    
    for key, cols in found_ohlcv.items():
        print(f"  {key}: {cols}")
    
    # Check if data might be transposed
    print(f"\n=== CHECKING IF TRANSPOSED ===")
    # If we have way more columns than expected, might be transposed
    if len(df.columns) > 100:
        print(f"WARNING: {len(df.columns)} columns detected - file might be transposed!")
        print(f"Trying transpose...")
        try:
            df_t = df.T
            print(f"Transposed shape: {df_t.shape}")
            print(f"Transposed index (first 10): {df_t.index[:10].tolist()}")
            print(f"Transposed columns (first 10): {df_t.columns[:10].tolist()}")
        except Exception as e:
            print(f"Transpose failed: {e}")
    
    # Try to find timestamp column
    print(f"\n=== TIMESTAMP DETECTION ===")
    # Check if any column name looks like a timestamp
    timestamp_cols = []
    for col in df.columns:
        try:
            col_num = float(str(col))
            if col_num > 946684800:  # After 2000-01-01
                timestamp_cols.append((col, col_num))
        except:
            pass
    
    if timestamp_cols:
        print(f"Found {len(timestamp_cols)} columns with timestamp-like names")
        print(f"First 5: {timestamp_cols[:5]}")
    else:
        print("No timestamp-like column names found")
    
    # Check if index could be timestamps
    if pd.api.types.is_numeric_dtype(df.index):
        sample_idx = df.index[0] if len(df) > 0 else None
        if sample_idx and sample_idx > 946684800:
            print(f"\nIndex might be timestamps!")
            print(f"First index value: {sample_idx}")
            if sample_idx > 1e12:
                print(f"As datetime (ms): {pd.to_datetime(sample_idx, unit='ms')}")
            else:
                print(f"As datetime (s): {pd.to_datetime(sample_idx, unit='s')}")

if __name__ == '__main__':
    main()

