from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .base import DownloadRequest, ProviderError, normalize_ohlcv


_INTERVAL_MAP = {
    "1m": "1m",
    "2m": "2m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "60m",
    "60m": "60m",
    "90m": "90m",
    "1d": "1d",
    "5d": "5d",
    "1wk": "1wk",
    "1mo": "1mo",
    "3mo": "3mo",
}


@dataclass(frozen=True)
class YFinanceProvider:
    """Yahoo Finance provider via yfinance.

    Works across many asset classes depending on Yahoo symbol support:
    - Stocks: AAPL, MSFT
    - Indices: ^GSPC, ^NDX
    - FX: EURUSD=X
    - Futures: ES=F, GC=F
    """

    name: str = "yfinance"

    def download(self, req: DownloadRequest, **kwargs) -> pd.DataFrame:
        try:
            import yfinance as yf
        except Exception as e:  # noqa: BLE001
            raise ProviderError(
                "yfinance is not installed. Install with: pip install yfinance"
            ) from e

        interval = req.interval.lower().strip()

        # yfinance does not support 4h directly; recommend downloading 1h and resampling.
        if interval == "4h":
            raise ProviderError("yfinance does not support interval=4h directly; download 1h and use --resample-to 4h")

        yf_interval = _INTERVAL_MAP.get(interval)
        if not yf_interval:
            raise ProviderError(f"Unsupported interval for yfinance: {req.interval}")

        start = pd.to_datetime(req.start_date)
        end = pd.to_datetime(req.end_date) + pd.Timedelta(days=1)  # inclusive by date
        auto_adjust = bool(kwargs.get("auto_adjust", False))

        df = yf.download(
            req.symbol,
            start=start,
            end=end,
            interval=yf_interval,
            auto_adjust=auto_adjust,
            progress=False,
            group_by="column",
            threads=True,
        )

        if df is None or df.empty:
            raise ProviderError(f"No data returned for symbol '{req.symbol}' from Yahoo Finance")

        # Standard yfinance columns: Open High Low Close Adj Close Volume
        cols = {c: c.lower().replace(" ", "_") for c in df.columns}
        df = df.rename(columns=cols)

        # Prefer 'close' (or adjusted 'close' if auto_adjust True)
        if "open" not in df.columns and "open" in cols.values():
            pass

        if "volume" not in df.columns:
            df["volume"] = 0.0

        # Some symbols can return 'adj_close' too; ignore.
        base = df[[c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]].copy()

        # yfinance index may be tz-aware
        if isinstance(base.index, pd.DatetimeIndex) and base.index.tz is not None:
            base.index = base.index.tz_convert(None)

        return normalize_ohlcv(base)
