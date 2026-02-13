from __future__ import annotations

from dataclasses import dataclass
from io import StringIO

import pandas as pd
import requests

from .base import DownloadRequest, ProviderError, normalize_ohlcv


@dataclass(frozen=True)
class CsvUrlProvider:
    """Download OHLCV from an arbitrary CSV URL.

    This is the "escape hatch" provider to support any market/data vendor
    that can expose OHLCV as CSV.

    Required inputs (via kwargs):
      - url
    Optional mappings (via kwargs):
      - datetime_col (default: 'timestamp')
      - open_col/high_col/low_col/close_col/volume_col
      - date_format (strftime-compatible) or blank for pandas inference
      - tz (e.g. 'UTC'); if provided, localize then convert to tz-naive
      - unit ('s' or 'ms') for epoch timestamps
    """

    name: str = "csv_url"

    def download(self, req: DownloadRequest, **kwargs) -> pd.DataFrame:
        url = kwargs.get("url")
        if not url:
            raise ProviderError("csv_url provider requires --csv-url")

        datetime_col = kwargs.get("datetime_col", "timestamp")
        open_col = kwargs.get("open_col", "open")
        high_col = kwargs.get("high_col", "high")
        low_col = kwargs.get("low_col", "low")
        close_col = kwargs.get("close_col", "close")
        volume_col = kwargs.get("volume_col", "volume")
        date_format = kwargs.get("date_format")
        tz = kwargs.get("tz")
        unit = kwargs.get("unit")

        r = requests.get(url, timeout=60)
        r.raise_for_status()

        df = pd.read_csv(StringIO(r.text))
        if df.empty:
            raise ProviderError(f"No rows returned from {url}")

        if datetime_col not in df.columns:
            raise ProviderError(f"CSV missing datetime column '{datetime_col}'. Columns: {list(df.columns)[:20]}")

        if unit:
            ts = pd.to_datetime(df[datetime_col], unit=unit)
        elif date_format:
            ts = pd.to_datetime(df[datetime_col], format=date_format)
        else:
            ts = pd.to_datetime(df[datetime_col])

        if tz:
            ts = ts.dt.tz_localize(tz).dt.tz_convert(None)

        out = pd.DataFrame(
            {
                "open": df[open_col] if open_col in df.columns else None,
                "high": df[high_col] if high_col in df.columns else None,
                "low": df[low_col] if low_col in df.columns else None,
                "close": df[close_col] if close_col in df.columns else None,
                "volume": df[volume_col] if volume_col in df.columns else 0.0,
            },
            index=ts,
        )

        start = pd.to_datetime(req.start_date)
        end = pd.to_datetime(req.end_date)
        out = out[(out.index >= start) & (out.index <= end)].copy()

        return normalize_ohlcv(out)
