#!/usr/bin/env python3
"""Inspect data file to understand timestamp format."""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def main():
    data_path = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'BTCUSDT-15m-2020.parquet'
    
    print(f"Loading: {data_path}")
    print(f"File exists: {data_path.exists()}")
    
    # Load raw Parquet without any processing
    df = pd.read_parquet(data_path)
    
    print(f"\n=== RAW DATA INFO ===")
    print(f"Shape: {df.shape}")
    print(f"Index type: {type(df.index)}")
    print(f"Index name: {df.index.name}")
    print(f"Index dtype: {df.index.dtype}")
    print(f"\nFirst 5 index values:")
    print(df.index[:5])
    print(f"\nLast 5 index values:")
    print(df.index[-5:])
    
    print(f"\n=== COLUMNS ===")
    print(f"Columns: {df.columns.tolist()}")
    
    print(f"\n=== FIRST ROW ===")
    print(df.iloc[0])
    
    print(f"\n=== INDEX SAMPLE VALUES ===")
    # Check if index is numeric
    if pd.api.types.is_numeric_dtype(df.index):
        print(f"Index is numeric")
        print(f"First value: {df.index[0]}")
        print(f"Last value: {df.index[-1]}")
        print(f"Min: {df.index.min()}, Max: {df.index.max()}")
        
        # Try to interpret as different timestamp formats
        first_val = df.index[0]
        if first_val > 1e12:
            print(f"\nInterpreting as milliseconds:")
            print(f"  First: {pd.to_datetime(first_val, unit='ms')}")
            print(f"  Last: {pd.to_datetime(df.index[-1], unit='ms')}")
        elif first_val > 1e9:
            print(f"\nInterpreting as seconds:")
            print(f"  First: {pd.to_datetime(first_val, unit='s')}")
            print(f"  Last: {pd.to_datetime(df.index[-1], unit='s')}")
        else:
            print(f"\nValue too small to be a Unix timestamp")
    else:
        print(f"Index is not numeric: {df.index.dtype}")
        print(f"First value: {df.index[0]}")
        print(f"Last value: {df.index[-1]}")
    
    # Check if there's a timestamp column
    print(f"\n=== CHECKING FOR TIMESTAMP COLUMNS ===")
    timestamp_cols = ['timestamp', 'datetime', 'date', 'time', 'Timestamp', 'DateTime', 'open_time', 'close_time']
    for col in timestamp_cols:
        if col in df.columns:
            print(f"Found column '{col}':")
            print(f"  Type: {df[col].dtype}")
            print(f"  First value: {df[col].iloc[0]}")
            print(f"  Last value: {df[col].iloc[-1]}")
            if pd.api.types.is_numeric_dtype(df[col]):
                first_val = df[col].iloc[0]
                if first_val > 1e12:
                    print(f"  As milliseconds: {pd.to_datetime(first_val, unit='ms')}")
                elif first_val > 1e9:
                    print(f"  As seconds: {pd.to_datetime(first_val, unit='s')}")

if __name__ == '__main__':
    main()

