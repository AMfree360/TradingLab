from __future__ import annotations

from dataclasses import dataclass
from io import StringIO

import pandas as pd
import requests

from .base import DownloadRequest, ProviderError, normalize_ohlcv


@dataclass(frozen=True)
class StooqProvider:
    """Stooq daily data provider.

    Stooq supports many equities/indices/FX pairs but is DAILY only.
    Symbols are stooq-specific (e.g. 'aapl.us').
    """

    name: str = "stooq"

    def download(self, req: DownloadRequest, **kwargs) -> pd.DataFrame:
        interval = req.interval.lower().strip()
        if interval not in {"1d", "d", "day", "daily"}:
            raise ProviderError("Stooq provider supports daily data only (use interval=1d)")

        symbol = req.symbol.lower().strip()
        url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
        r = requests.get(url, timeout=60)
        r.raise_for_status()

        df = pd.read_csv(StringIO(r.text))

        if df.empty or "Date" not in df.columns:
            raise ProviderError(f"No data returned for symbol '{req.symbol}' from Stooq")

        df["Date"] = pd.to_datetime(df["Date"], utc=False)
        df = df.set_index("Date")

        # Normalize column names to match normalize_ohlcv expectations
        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

        start = pd.to_datetime(req.start_date)
        end = pd.to_datetime(req.end_date)
        df = df[(df.index >= start) & (df.index <= end)].copy()

        return normalize_ohlcv(df)
