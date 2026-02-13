from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


class ProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class DownloadRequest:
    symbol: str
    interval: str
    start_date: str
    end_date: str


class DataProvider(Protocol):
    name: str

    def download(self, req: DownloadRequest, **kwargs) -> pd.DataFrame:
        raise NotImplementedError


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common provider outputs to TradingLab OHLCV schema."""
    if df is None or len(df) == 0:
        raise ProviderError("Provider returned empty data")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ProviderError("Expected DatetimeIndex")

    out = df.copy()
    # tz-naive
    if out.index.tz is not None:
        out.index = out.index.tz_convert(None)

    out = out.sort_index()
    out = out[~out.index.duplicated(keep="first")]

    # column normalization
    cols = {c: c.lower() for c in out.columns}
    out = out.rename(columns=cols)

    required = ["open", "high", "low", "close"]
    for col in required:
        if col not in out.columns:
            raise ProviderError(f"Missing required OHLC column '{col}'")

    if "volume" not in out.columns:
        out["volume"] = 0.0

    out = out[["open", "high", "low", "close", "volume"]].astype(float)
    out.index.name = "timestamp"
    return out
