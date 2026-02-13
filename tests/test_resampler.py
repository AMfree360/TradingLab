"""Tests for data resampler."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from engine.resampler import (
    resample_ohlcv,
    resample_multiple,
    parse_timeframe,
    _timeframe_to_minutes,
    align_timeframes,
)


def create_sample_1m_data(n_bars=100):
    """Create sample 1-minute OHLCV data."""
    dates = pd.date_range(start='2024-01-01 00:00:00', periods=n_bars, freq='1min')
    np.random.seed(42)
    
    base_price = 100.0
    prices = []
    for i in range(n_bars):
        change = np.random.randn() * 0.1
        base_price += change
        prices.append(base_price)
    
    df = pd.DataFrame({
        'open': prices,
        'high': [p * 1.001 for p in prices],
        'low': [p * 0.999 for p in prices],
        'close': [p * 1.0005 for p in prices],
        'volume': np.random.randint(100, 1000, n_bars),
    }, index=dates)
    
    return df


def test_parse_timeframe():
    """Test timeframe parsing."""
    assert parse_timeframe('1m') == '1min'
    assert parse_timeframe('15m') == '15min'
    assert parse_timeframe('1h') == '1h'
    assert parse_timeframe('1d') == '1D'
    assert parse_timeframe('2h') == '2h'
    assert parse_timeframe('3d') == '3D'
    
    with pytest.raises(ValueError):
        parse_timeframe('invalid')


def test_timeframe_to_minutes():
    """Test timeframe to minutes conversion."""
    assert _timeframe_to_minutes('1m') == 1
    assert _timeframe_to_minutes('15m') == 15
    assert _timeframe_to_minutes('1h') == 60
    assert _timeframe_to_minutes('1d') == 1440
    assert _timeframe_to_minutes('2h') == 120


def test_resample_ohlcv_basic():
    """Test basic OHLCV resampling."""
    df_1m = create_sample_1m_data(60)  # 60 minutes = 1 hour
    
    df_1h = resample_ohlcv(df_1m, '1h', base_tf='1m')
    
    # With label/closed='right', resampling produces an initial partial bin at the first timestamp.
    assert len(df_1h) == 2
    assert df_1h.index[0].minute == 0
    assert 'open' in df_1h.columns
    assert 'high' in df_1h.columns
    assert 'low' in df_1h.columns
    assert 'close' in df_1h.columns
    assert 'volume' in df_1h.columns
    
    # Check aggregation across all resampled bars
    assert df_1h.iloc[0]['open'] == df_1m.iloc[0]['open']  # First open preserved
    assert df_1h.iloc[-1]['close'] == df_1m.iloc[-1]['close']  # Last close preserved
    assert df_1h['high'].max() == df_1m['high'].max()
    assert df_1h['low'].min() == df_1m['low'].min()
    assert df_1h['volume'].sum() == df_1m['volume'].sum()  # Volume conserved


def test_resample_ohlcv_15m():
    """Test resampling to 15-minute timeframe."""
    df_1m = create_sample_1m_data(60)  # 60 minutes
    
    df_15m = resample_ohlcv(df_1m, '15m', base_tf='1m')
    
    # With label/closed='right', there is an initial partial bin
    assert len(df_15m) == 5


def test_resample_ohlcv_validation():
    """Test resampling validation."""
    df_1m = create_sample_1m_data(10)
    
    # Missing column
    df_bad = df_1m.drop(columns=['close'])
    with pytest.raises(ValueError):
        resample_ohlcv(df_bad, '1h')
    
    # Non-datetime index
    df_bad2 = df_1m.reset_index()
    with pytest.raises(ValueError):
        resample_ohlcv(df_bad2, '1h')
    
    # Invalid target timeframe (smaller than base)
    with pytest.raises(ValueError):
        resample_ohlcv(df_1m, '30s', base_tf='1m')


def test_resample_multiple():
    """Test resampling to multiple timeframes."""
    df_1m = create_sample_1m_data(240)  # 4 hours of data
    
    result = resample_multiple(df_1m, ['15m', '1h', '4h'], base_tf='1m')
    
    assert '15m' in result
    assert '1h' in result
    assert '4h' in result
    
    # With label/closed='right', there is an initial partial bin
    assert len(result['4h']) == 2
    assert len(result['1h']) == 5
    assert len(result['15m']) == 17


def test_align_timeframes():
    """Test timeframe alignment."""
    # Create data for different timeframes
    dates_1h = pd.date_range(start='2024-01-01 00:00:00', periods=4, freq='1h')
    dates_15m = pd.date_range(start='2024-01-01 00:00:00', periods=16, freq='15min')
    
    df_1h = pd.DataFrame({
        'close': [100, 101, 102, 103],
    }, index=dates_1h)
    
    df_15m = pd.DataFrame({
        'close': range(16),
    }, index=dates_15m)
    
    df_by_tf = {
        '1h': df_1h,
        '15m': df_15m,
    }
    
    aligned = align_timeframes(df_by_tf, reference_tf='15m')
    
    assert '1h' in aligned
    assert '15m' in aligned
    assert len(aligned['1h']) == len(aligned['15m'])
    assert len(aligned['15m']) == 16


