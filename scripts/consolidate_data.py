#!/usr/bin/env python3
"""Consolidate multiple data files into a single file for validation."""

import argparse
import sys
from pathlib import Path
import re
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.data.data_loader import DataLoader


def consolidate_files(
    input_files: list[Path],
    output_path: Path,
    sort: bool = True,
    remove_duplicates: bool = True,
    dedupe_keep: str = 'first',
    allow_mixed_frequency: bool = False,
) -> Path:
    """
    Consolidate multiple OHLCV files into one.
    
    Args:
        input_files: List of input file paths
        output_path: Where to save consolidated data
        sort: Sort by timestamp after consolidation
        remove_duplicates: Remove duplicate timestamps
        dedupe_keep: When removing duplicates, which record to keep ('first' or 'last')
        allow_mixed_frequency: If False, error when input files appear to have different bar intervals
    
    Returns:
        Path to consolidated file
    """
    loader = DataLoader()
    all_data = []

    def _infer_minutes_from_index(idx) -> int | None:
        if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 3:
            return None
        try:
            freq = pd.infer_freq(idx)
            if freq:
                f = str(freq)
                if f.endswith('T') or f.endswith('min'):
                    digits = re.sub(r'\D+', '', f)
                    return int(digits) if digits else None
                if f.lower().endswith('h'):
                    digits = re.sub(r'\D+', '', f)
                    return (int(digits) if digits else 1) * 60
                if f.lower().endswith('d'):
                    digits = re.sub(r'\D+', '', f)
                    return (int(digits) if digits else 1) * 1440
        except Exception:
            pass
        try:
            diffs = idx.to_series().diff().dropna().dt.total_seconds() / 60.0
            if diffs.empty:
                return None
            return int(diffs.median())
        except Exception:
            return None

    def _mins_to_tf(m: int | None) -> str:
        if m is None:
            return 'unknown'
        if m % 1440 == 0:
            days = m // 1440
            return f"{days}d" if days > 1 else '1d'
        if m % 60 == 0:
            hours = m // 60
            return f"{hours}h" if hours > 1 else '1h'
        return f"{int(m)}m"
    
    print(f"Consolidating {len(input_files)} files...")

    inferred_minutes: list[int | None] = []
    
    for i, file_path in enumerate(input_files, 1):
        if not file_path.exists():
            print(f"Warning: File not found: {file_path}, skipping...")
            continue

        if file_path.is_dir():
            print(f"Warning: Input is a directory (not a file): {file_path}, skipping...")
            continue
        
        print(f"  [{i}/{len(input_files)}] Loading {file_path.name}...")
        df = loader.load(file_path)

        if df.empty:
            print(f"    Loaded 0 bars (empty), skipping...")
            continue

        mins = _infer_minutes_from_index(df.index)
        inferred_minutes.append(mins)
        all_data.append(df)
        print(
            f"    Loaded {len(df)} bars ({df.index[0]} to {df.index[-1]})"
            f" | interval≈{_mins_to_tf(mins)}"
        )
    
    if not all_data:
        raise ValueError("No valid data files found!")

    unique_mins = {m for m in inferred_minutes if m is not None}
    if (not allow_mixed_frequency) and len(unique_mins) > 1:
        details = ', '.join(sorted({_mins_to_tf(m) for m in unique_mins}))
        raise ValueError(
            "Input files appear to have mixed bar intervals "
            f"({details}). Resample to a single timeframe before consolidating, "
            "or pass --allow-mixed-frequency if you really intend to merge them."
        )
    
    # Concatenate all dataframes
    print("\nCombining data...")
    consolidated = pd.concat(all_data, axis=0)

    # Sort by timestamp early (helps downstream checks and makes output stable)
    if sort:
        consolidated = consolidated.sort_index()
    
    # Remove duplicates if requested
    if remove_duplicates:
        if dedupe_keep not in {'first', 'last'}:
            raise ValueError("dedupe_keep must be 'first' or 'last'")
        initial_len = len(consolidated)
        consolidated = consolidated[~consolidated.index.duplicated(keep=dedupe_keep)]
        removed = initial_len - len(consolidated)
        if removed > 0:
            print(f"  Removed {removed} duplicate timestamps (kept={dedupe_keep})")
    
    # Save consolidated file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix.lower() == '.parquet':
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
    parser.add_argument(
        '--dedupe-keep',
        type=str,
        choices=['first', 'last'],
        default='first',
        help="When removing duplicates, keep 'first' or 'last' (default: first)"
    )
    parser.add_argument(
        '--allow-mixed-frequency',
        action='store_true',
        help='Allow consolidating inputs with different bar intervals (default: off)'
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
            remove_duplicates=not args.keep_duplicates,
            dedupe_keep=args.dedupe_keep,
            allow_mixed_frequency=bool(args.allow_mixed_frequency),
        )
        print("\n✓ Consolidation complete!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

