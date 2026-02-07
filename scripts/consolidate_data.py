#!/usr/bin/env python3
"""Consolidate multiple data files into a single file for validation."""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.data.data_loader import DataLoader


def consolidate_files(
    input_files: list[Path],
    output_path: Path,
    sort: bool = True,
    remove_duplicates: bool = True
) -> Path:
    """
    Consolidate multiple OHLCV files into one.
    
    Args:
        input_files: List of input file paths
        output_path: Where to save consolidated data
        sort: Sort by timestamp after consolidation
        remove_duplicates: Remove duplicate timestamps
    
    Returns:
        Path to consolidated file
    """
    loader = DataLoader()
    all_data = []
    
    print(f"Consolidating {len(input_files)} files...")
    
    for i, file_path in enumerate(input_files, 1):
        if not file_path.exists():
            print(f"Warning: File not found: {file_path}, skipping...")
            continue
        
        print(f"  [{i}/{len(input_files)}] Loading {file_path.name}...")
        df = loader.load(file_path)
        all_data.append(df)
        print(f"    Loaded {len(df)} bars ({df.index[0]} to {df.index[-1]})")
    
    if not all_data:
        raise ValueError("No valid data files found!")
    
    # Concatenate all dataframes
    print("\nCombining data...")
    consolidated = pd.concat(all_data, axis=0)
    
    # Remove duplicates if requested
    if remove_duplicates:
        initial_len = len(consolidated)
        consolidated = consolidated[~consolidated.index.duplicated(keep='first')]
        removed = initial_len - len(consolidated)
        if removed > 0:
            print(f"  Removed {removed} duplicate timestamps")
    
    # Sort by timestamp
    if sort:
        consolidated = consolidated.sort_index()
    
    # Save consolidated file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == '.parquet':
        consolidated.to_parquet(output_path)
    else:
        consolidated.to_csv(output_path)
    
    print(f"\n✓ Consolidated {len(consolidated)} bars")
    print(f"  Date range: {consolidated.index[0]} to {consolidated.index[-1]}")
    print(f"  Saved to: {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Consolidate multiple data files into one for validation'
    )
    parser.add_argument(
        '--input',
        type=str,
        nargs='+',
        required=True,
        help='Input data files (can use wildcards, e.g., data/raw/*202*.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output consolidated file path'
    )
    parser.add_argument(
        '--no-sort',
        action='store_true',
        help='Do not sort by timestamp (default: sort)'
    )
    parser.add_argument(
        '--keep-duplicates',
        action='store_true',
        help='Keep duplicate timestamps (default: remove)'
    )
    
    args = parser.parse_args()
    
    # Expand input files (handle wildcards)
    input_files = []
    for pattern in args.input:
        path = Path(pattern)
        if '*' in pattern or '?' in pattern:
            # Use glob
            input_files.extend(sorted(path.parent.glob(path.name)))
        else:
            input_files.append(path)
    
    if not input_files:
        print("Error: No input files found!")
        sys.exit(1)
    
    output_path = Path(args.output)
    
    try:
        consolidate_files(
            input_files=input_files,
            output_path=output_path,
            sort=not args.no_sort,
            remove_duplicates=not args.keep_duplicates
        )
        print("\n✓ Consolidation complete!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

