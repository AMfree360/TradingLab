"""Timeframe resampling utilities for converting base timeframe to target timeframes."""

import pandas as pd
from typing import Union
import re


# Mapping of timeframe strings to pandas frequency strings
TIMEFRAME_MAP = {
    '1m': '1min',
    '5m': '5min',
    '15m': '15min',
    '30m': '30min',
    '1h': '1h',
    '4h': '4h',
    '1d': '1D',
    '1w': '1W',
}


def parse_timeframe(tf: str) -> str:
    """
    Parse timeframe string to pandas frequency string.
    
    Args:
        tf: Timeframe string (e.g., '1h', '15m', '1d')
    
    Returns:
        Pandas frequency string
    
    Raises:
        ValueError if timeframe is not supported
    """
    tf_lower = tf.lower().strip()
    
    if tf_lower in TIMEFRAME_MAP:
        return TIMEFRAME_MAP[tf_lower]
    
    # Try to parse custom timeframes (e.g., '2h', '3d')
    match = re.match(r'^(\d+)([mhdw])$', tf_lower)
    if match:
        value = int(match.group(1))
        unit = match.group(2)
        
        unit_map = {
            'm': 'min',  # minutes
            'h': 'h',  # hours
            'd': 'D',  # days
            'w': 'W',  # weeks
        }
        
        if unit in unit_map:
            return f"{value}{unit_map[unit]}"
    
    raise ValueError(f"Unsupported timeframe: {tf}. Supported: {list(TIMEFRAME_MAP.keys())}")


def resample_ohlcv(
    df: pd.DataFrame,
    target_tf: str,
    base_tf: str = None,
    label: str = 'right',
    closed: str = 'right'
) -> pd.DataFrame:
    """
    Resample OHLCV data to target timeframe.
    
    Proper OHLC aggregation:
    - open: first value in period
    - high: maximum value in period
    - low: minimum value in period
    - close: last value in period
    - volume: sum of volumes in period
    
    Args:
        df: DataFrame with datetime index and OHLCV columns
        target_tf: Target timeframe string (e.g., '1h', '15m')
        base_tf: Base timeframe string (optional, for validation)
        label: Label for resampled index ('left' or 'right', default 'right')
        closed: Which side of interval is closed ('left' or 'right', default 'right')
    
    Returns:
        Resampled DataFrame with same columns
    
    Raises:
        ValueError if required columns are missing or timeframe is invalid
    """
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")
    
    if df.empty:
        return df.copy()
    
    # Parse target timeframe
    freq = parse_timeframe(target_tf)
    
    # Validate that target timeframe is >= base timeframe
    if base_tf:
        base_freq = parse_timeframe(base_tf)
        # Simple check: if target has larger unit or same unit with larger value
        # This is approximate but works for common cases
        target_minutes = _timeframe_to_minutes(target_tf)
        base_minutes = _timeframe_to_minutes(base_tf)
        if target_minutes < base_minutes:
            raise ValueError(
                f"Target timeframe ({target_tf}) must be >= base timeframe ({base_tf})"
            )
    
    # Resample with proper aggregation
    resampled = df.resample(
        freq,
        label=label,
        closed=closed,
        origin='start'
    ).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    })
    
    # Drop any rows where all OHLC are NaN (empty periods)
    resampled = resampled.dropna(subset=['open', 'high', 'low', 'close'], how='all')
    
    return resampled


def _timeframe_to_minutes(tf: str) -> int:
    """
    Convert timeframe string to minutes (approximate).
    
    Args:
        tf: Timeframe string
    
    Returns:
        Approximate minutes
    """
    tf_lower = tf.lower().strip()
    match = re.match(r'^(\d+)([mhdw])$', tf_lower)
    if match:
        value = int(match.group(1))
        unit = match.group(2)
        
        unit_multipliers = {
            'm': 1,
            'h': 60,
            'd': 1440,  # 24 * 60
            'w': 10080,  # 7 * 24 * 60
        }
        
        if unit in unit_multipliers:
            return value * unit_multipliers[unit]
    
    # Fallback to known timeframes
    known_minutes = {
        '1m': 1,
        '5m': 5,
        '15m': 15,
        '30m': 30,
        '1h': 60,
        '4h': 240,
        '1d': 1440,
        '1w': 10080,
    }
    return known_minutes.get(tf_lower, 1)


def resample_multiple(
    df: pd.DataFrame,
    target_tfs: list[str],
    base_tf: str = None
) -> dict[str, pd.DataFrame]:
    """
    Resample data to multiple target timeframes.
    
    Args:
        df: Base DataFrame with OHLCV data
        target_tfs: List of target timeframe strings
        base_tf: Base timeframe string (optional)
    
    Returns:
        Dictionary mapping timeframe strings to resampled DataFrames
    """
    result = {}
    for tf in target_tfs:
        result[tf] = resample_ohlcv(df, tf, base_tf)
    return result


def align_timeframes(
    df_by_tf: dict[str, pd.DataFrame],
    reference_tf: str = None
) -> dict[str, pd.DataFrame]:
    """
    Align multiple timeframes to a reference timeframe index.
    
    This is useful when you need to align signals from different timeframes.
    Uses forward fill to align higher timeframes to lower timeframes.
    
    Args:
        df_by_tf: Dictionary mapping timeframe strings to DataFrames
        reference_tf: Reference timeframe to align to (default: smallest timeframe)
    
    Returns:
        Dictionary with aligned DataFrames
    """
    if not df_by_tf:
        return {}
    
    # Find reference timeframe (smallest by default)
    if reference_tf is None:
        tf_minutes = {tf: _timeframe_to_minutes(tf) for tf in df_by_tf.keys()}
        reference_tf = min(tf_minutes, key=tf_minutes.get)
    
    if reference_tf not in df_by_tf:
        raise ValueError(f"Reference timeframe {reference_tf} not found in data")
    
    reference_df = df_by_tf[reference_tf]
    reference_index = reference_df.index
    
    aligned = {reference_tf: reference_df.copy()}
    
    for tf, df in df_by_tf.items():
        if tf == reference_tf:
            continue
        
        # Reindex to reference index, forward filling
        aligned_df = df.reindex(reference_index, method='ffill')
        aligned[tf] = aligned_df
    
    return aligned


