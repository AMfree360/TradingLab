"""Binance Data Vision downloader.

Binance provides bulk historical kline data as zipped CSVs here:
https://data.binance.vision/

This module downloads monthly klines and normalizes them into the TradingLab
standard OHLCV schema (DatetimeIndex + open/high/low/close/volume).

Notes:
- Uses public endpoints (no API key required)
- Much faster/more reliable for multi-year history than the REST /api/v3/klines loop
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
import zipfile

import pandas as pd
import requests


KLINES_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]


@dataclass(frozen=True)
class VisionSpec:
    symbol: str
    interval: str
    market: str  # "spot" | "futures_um"


class BinanceVisionError(RuntimeError):
    pass


def _month_iter(start: pd.Timestamp, end: pd.Timestamp):
    cursor = pd.Timestamp(year=start.year, month=start.month, day=1, tz=None)
    end_month = pd.Timestamp(year=end.year, month=end.month, day=1, tz=None)
    while cursor <= end_month:
        yield int(cursor.year), int(cursor.month)
        cursor = (cursor + pd.offsets.MonthBegin(1))


def build_monthly_klines_url(spec: VisionSpec, *, year: int, month: int) -> str:
    """Build Data Vision monthly klines URL."""
    mm = f"{month:02d}"

    market = spec.market.lower().strip()
    if market == "spot":
        # https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/15m/BTCUSDT-15m-2020-01.zip
        return (
            "https://data.binance.vision/data/spot/monthly/klines/"
            f"{spec.symbol}/{spec.interval}/{spec.symbol}-{spec.interval}-{year}-{mm}.zip"
        )

    if market in {"futures_um", "um", "futures"}:
        # https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/15m/BTCUSDT-15m-2020-01.zip
        return (
            "https://data.binance.vision/data/futures/um/monthly/klines/"
            f"{spec.symbol}/{spec.interval}/{spec.symbol}-{spec.interval}-{year}-{mm}.zip"
        )

    raise BinanceVisionError(f"Unsupported market '{spec.market}'. Use 'spot' or 'futures_um'.")


def _default_cache_dir() -> Path:
    return Path("data/raw/_cache/binance_vision")


def _download_bytes(url: str, *, timeout: int = 60, retries: int = 3) -> bytes:
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 404:
                raise FileNotFoundError(url)
            r.raise_for_status()
            return r.content
        except FileNotFoundError:
            raise
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt == retries:
                break
    raise BinanceVisionError(f"Failed to download {url}: {last_err}")


def load_monthly_klines_from_zip_bytes(zip_bytes: bytes) -> pd.DataFrame:
    """Parse a Binance Vision kline ZIP into a normalized OHLCV DataFrame."""
    with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise BinanceVisionError("ZIP did not contain a CSV")
        # Typically exactly one CSV
        with zf.open(names[0]) as f:
            df = pd.read_csv(
                f,
                header=None,
                names=KLINES_COLUMNS,
                dtype={
                    "open_time": "int64",
                    "open": "float64",
                    "high": "float64",
                    "low": "float64",
                    "close": "float64",
                    "volume": "float64",
                },
            )

    # Normalize
    # Binance Vision historically used millisecond timestamps, but some newer
    # datasets appear to use microseconds. Auto-detect based on magnitude.
    sample = int(df["open_time"].iloc[0])
    if sample > 10**17:
        unit = "ns"
    elif sample > 10**14:
        unit = "us"
    else:
        unit = "ms"

    ts = pd.to_datetime(df["open_time"], unit=unit, utc=True).dt.tz_convert(None)

    # Avoid index-alignment NaNs: assign the datetime index before selecting columns.
    df = df.set_index(ts)
    out = df[["open", "high", "low", "close", "volume"]].copy()
    out.index.name = "timestamp"
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out


def download_klines_range(
    *,
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
    market: str = "spot",
    cache_dir: Path | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Download klines for [start_date, end_date] (inclusive by date) using Data Vision monthly files."""
    start = pd.to_datetime(start_date).normalize()
    end = pd.to_datetime(end_date).normalize()
    if end < start:
        raise ValueError("end_date must be >= start_date")

    spec = VisionSpec(symbol=symbol, interval=interval, market=market)
    cache_dir = cache_dir or _default_cache_dir()

    parts: list[pd.DataFrame] = []

    months = list(_month_iter(start, end))
    iterator = months
    if show_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(months, desc=f"BinanceVision {symbol} {interval}")
        except Exception:
            iterator = months

    for year, month in iterator:
        url = build_monthly_klines_url(spec, year=year, month=month)

        cache_path = cache_dir / spec.market / "monthly" / "klines" / spec.symbol / spec.interval / f"{spec.symbol}-{spec.interval}-{year}-{month:02d}.zip"
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            zip_bytes = cache_path.read_bytes()
        else:
            try:
                zip_bytes = _download_bytes(url)
            except FileNotFoundError:
                # Missing month (common for some markets/intervals). Skip.
                continue
            cache_path.write_bytes(zip_bytes)

        dfm = load_monthly_klines_from_zip_bytes(zip_bytes)
        parts.append(dfm)

    if not parts:
        raise BinanceVisionError("No data downloaded (no monthly files found in range)")

    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # Filter to requested date range inclusive.
    start_ts = start
    end_ts_exclusive = end + pd.Timedelta(days=1)
    df = df[(df.index >= start_ts) & (df.index < end_ts_exclusive)].copy()

    if df.empty:
        raise BinanceVisionError("Downloaded data but nothing remained after filtering")

    return df


def save_ohlcv(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        df.to_parquet(output_path)
    else:
        df.to_csv(output_path)
    return output_path


def default_output_path(*, symbol: str, interval: str, start_date: str, end_date: str, fmt: str = "parquet") -> Path:
    ext = ".parquet" if fmt == "parquet" else ".csv"
    return Path("data/raw") / f"{symbol}-{interval}-{start_date}_to_{end_date}{ext}"
