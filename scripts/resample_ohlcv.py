"""Resample OHLCV files (e.g., 15m -> 4h) and save to an output path.

Usage: python3 scripts/resample_ohlcv.py --src data/raw/BTCUSDT-15m-2023.parquet --dst data/raw/BTCUSDT-4h-2023.parquet --rule 4h
"""
from pathlib import Path
import argparse
import pandas as pd
from adapters.data.data_loader import DataLoader


def resample_file(src: Path, dst: Path, rule: str = '4h') -> Path:
    # Pandas deprecated uppercase hourly offset aliases (e.g., '4H'); normalize to avoid warnings.
    if isinstance(rule, str) and rule.lower().endswith('h') and rule[-1] == 'H':
        rule = rule[:-1] + 'h'

    dl = DataLoader()
    print(f"Loading source: {src}")
    df = dl.load(src)

    print(f"Resampling to {rule}...")
    out = pd.DataFrame()
    out['open'] = df['open'].resample(rule).first()
    out['high'] = df['high'].resample(rule).max()
    out['low'] = df['low'].resample(rule).min()
    out['close'] = df['close'].resample(rule).last()
    out['volume'] = df['volume'].resample(rule).sum()

    # Drop rows with any NA (incomplete bars)
    out = out.dropna()

    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dst)
    print(f"Saved resampled file: {dst} ({len(out)} bars)")
    return dst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True)
    parser.add_argument('--dst', required=True)
    parser.add_argument('--rule', default='4h')
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")

    resample_file(src, dst, args.rule)


if __name__ == '__main__':
    main()
