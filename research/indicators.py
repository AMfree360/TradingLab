from __future__ import annotations

import pandas as pd

from research.dsl import IndicatorRequest


def _ensure_series(out: pd.DataFrame, series_name: str) -> pd.DataFrame:
    if series_name in out.columns:
        return out
    if series_name == "hl2":
        out[series_name] = (out["high"] + out["low"]) / 2
        return out
    if series_name == "hlc3":
        out[series_name] = (out["high"] + out["low"] + out["close"]) / 3
        return out
    if series_name == "ohlc4":
        out[series_name] = (out["open"] + out["high"] + out["low"] + out["close"]) / 4
        return out
    return out


def ensure_indicators(df: pd.DataFrame, requests: list[IndicatorRequest]) -> pd.DataFrame:
    """Compute required indicator columns into df (in-place-ish).

    Naming conventions:
      - ema_{series}_{length}
      - sma_{series}_{length}
      - rsi_{series}_{length}
      - atr_{length}
      - donchian_high_{length}
      - donchian_low_{length}

    This function is deterministic and does not mutate OHLCV columns.
    """

    if df.empty:
        return df

    out = df.copy()

    for r in requests:
        if r.name in {"ema", "sma", "rsi"}:
            series_name, length = r.args
            series_name = str(series_name)
            length = int(length)

            out = _ensure_series(out, series_name)

            col = f"{r.name}_{series_name}_{length}"
            if col in out.columns:
                continue

            s = out[series_name]
            if r.name == "ema":
                out[col] = s.ewm(span=length, adjust=False).mean()
            elif r.name == "sma":
                out[col] = s.rolling(window=length, min_periods=1).mean()
            elif r.name == "rsi":
                delta = s.diff()
                gain = delta.clip(lower=0)
                loss = (-delta).clip(lower=0)
                avg_gain = gain.rolling(window=length, min_periods=length).mean()
                avg_loss = loss.rolling(window=length, min_periods=length).mean()
                rs = avg_gain / avg_loss.replace(0.0, pd.NA)
                out[col] = 100 - (100 / (1 + rs))

        elif r.name == "atr":
            (length,) = r.args
            length = int(length)
            col = f"atr_{length}"
            if col in out.columns:
                continue

            high_low = out["high"] - out["low"]
            high_close = (out["high"] - out["close"].shift(1)).abs()
            low_close = (out["low"] - out["close"].shift(1)).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            out[col] = tr.rolling(window=length, min_periods=1).mean()

        elif r.name in {"donchian_high", "donchian_low"}:
            (length,) = r.args
            length = int(length)
            col = f"{r.name}_{length}"
            if col in out.columns:
                continue

            if r.name == "donchian_high":
                out[col] = out["high"].rolling(window=length, min_periods=1).max()
            else:
                out[col] = out["low"].rolling(window=length, min_periods=1).min()

        else:
            raise ValueError(f"Unsupported indicator request: {r}")

    return out
