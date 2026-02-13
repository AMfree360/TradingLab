#!/usr/bin/env python3
"""Fill missing timestamps in OHLCV data with synthetic flat bars.

Use-case:
- Some exchanges/data sources can have occasional missing candles.
- For research/validation it can be convenient to enforce a perfect grid.

Behavior:
- Reindexes to an exact frequency grid.
- For missing bars: open/high/low/close are set to the previous close, volume=0.
"""

import argparse
from pathlib import Path

import pandas as pd


def fill_missing_bars(df: pd.DataFrame, freq: str) -> tuple[pd.DataFrame, int]:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected DatetimeIndex")
    if df.empty:
        return df.copy(), 0

    df = df.sort_index()
    expected = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    out = df.reindex(expected)

    missing_mask = out["close"].isna()
    missing_count = int(missing_mask.sum())
    if missing_count == 0:
        out.index.name = df.index.name or "timestamp"
        return out, 0

    prev_close = out["close"].ffill()
    for col in ["open", "high", "low", "close"]:
        out.loc[missing_mask, col] = prev_close.loc[missing_mask]

    if "volume" in out.columns:
        out.loc[missing_mask, "volume"] = 0.0

    out.index.name = df.index.name or "timestamp"
    return out, missing_count


def main():
    p = argparse.ArgumentParser(description="Fill missing timegrid bars in OHLCV parquet/csv")
    p.add_argument("--input", required=True, help="Input parquet/csv path")
    p.add_argument("--output", required=True, help="Output parquet/csv path")
    p.add_argument("--freq", default="4h", help="Target frequency (default: 4h)")
    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if in_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(in_path)
    else:
        df = pd.read_csv(in_path, index_col=0, parse_dates=True)

    filled, missing = fill_missing_bars(df, args.freq)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".parquet":
        filled.to_parquet(out_path)
    else:
        filled.to_csv(out_path)

    print(f"Filled {missing} missing bars")
    print(f"Saved: {out_path} (rows={len(filled)})")


if __name__ == "__main__":
    main()
