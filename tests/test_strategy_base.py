"""Tests for base strategy interface."""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from strategies.base import StrategyBase
from config.schema import StrategyConfig


class ConcreteStrategy(StrategyBase):
    """Concrete implementation for testing."""
    
    def generate_signals(self, df_by_tf):
        signals = self._create_signal_dataframe()
        return signals
    
    def get_indicators(self, df, tf=None):
        df['sma_20'] = df['close'].rolling(20).mean()
        return df


def create_sample_data(n_bars=100):
    """Create sample OHLCV data."""
    dates = pd.date_range(start='2024-01-01', periods=n_bars, freq='1H')
    df = pd.DataFrame({
        'open': 100 + pd.Series(range(n_bars)) * 0.1,
        'high': 101 + pd.Series(range(n_bars)) * 0.1,
        'low': 99 + pd.Series(range(n_bars)) * 0.1,
        'close': 100.5 + pd.Series(range(n_bars)) * 0.1,
        'volume': 1000,
    }, index=dates)
    return df


def test_strategy_base_initialization():
    """Test strategy base initialization."""
    config_dict = {
        "strategy_name": "test_strategy",
        "market": {"exchange": "binance", "symbol": "BTCUSDT"},
        "timeframes": {"signal_tf": "1h", "entry_tf": "15m"},
        "moving_averages": {"ema5": {"enabled": True, "length": 5}},
        "alignment_rules": {
            "long": {"macd_bars_signal_tf": 1, "macd_bars_entry_tf": 2},
            "short": {"macd_bars_signal_tf": 1, "macd_bars_entry_tf": 2},
        },
    }
    config = StrategyConfig(**config_dict)
    strategy = ConcreteStrategy(config)
    
    assert strategy.name == "test_strategy"
    assert strategy.config == config


def test_get_required_timeframes():
    """Test getting required timeframes."""
    config_dict = {
        "strategy_name": "test_strategy",
        "market": {"exchange": "binance", "symbol": "BTCUSDT"},
        "timeframes": {"signal_tf": "1h", "entry_tf": "15m"},
        "moving_averages": {"ema5": {"enabled": True, "length": 5}},
        "alignment_rules": {
            "long": {"macd_bars_signal_tf": 1, "macd_bars_entry_tf": 2},
            "short": {"macd_bars_signal_tf": 1, "macd_bars_entry_tf": 2},
        },
    }
    config = StrategyConfig(**config_dict)
    strategy = ConcreteStrategy(config)
    
    timeframes = strategy.get_required_timeframes()
    assert "1h" in timeframes
    assert "15m" in timeframes


def test_get_indicators():
    """Test indicator computation."""
    config_dict = {
        "strategy_name": "test_strategy",
        "market": {"exchange": "binance", "symbol": "BTCUSDT"},
        "timeframes": {"signal_tf": "1h", "entry_tf": "15m"},
        "moving_averages": {"ema5": {"enabled": True, "length": 5}},
        "alignment_rules": {
            "long": {"macd_bars_signal_tf": 1, "macd_bars_entry_tf": 2},
            "short": {"macd_bars_signal_tf": 1, "macd_bars_entry_tf": 2},
        },
    }
    config = StrategyConfig(**config_dict)
    strategy = ConcreteStrategy(config)
    
    df = create_sample_data(50)
    df_with_indicators = strategy.get_indicators(df)
    
    assert 'sma_20' in df_with_indicators.columns
    assert len(df_with_indicators) == len(df)


def test_validate_dataframe():
    """Test DataFrame validation."""
    config_dict = {
        "strategy_name": "test_strategy",
        "market": {"exchange": "binance", "symbol": "BTCUSDT"},
        "timeframes": {"signal_tf": "1h", "entry_tf": "15m"},
        "moving_averages": {"ema5": {"enabled": True, "length": 5}},
        "alignment_rules": {
            "long": {"macd_bars_signal_tf": 1, "macd_bars_entry_tf": 2},
            "short": {"macd_bars_signal_tf": 1, "macd_bars_entry_tf": 2},
        },
    }
    config = StrategyConfig(**config_dict)
    strategy = ConcreteStrategy(config)
    
    # Valid DataFrame
    df = create_sample_data()
    strategy._validate_dataframe(df)
    
    # Missing column
    df_bad = df.drop(columns=['close'])
    with pytest.raises(ValueError):
        strategy._validate_dataframe(df_bad)
    
    # Non-datetime index
    df_bad2 = df.reset_index()
    with pytest.raises(ValueError):
        strategy._validate_dataframe(df_bad2)


def test_create_signal_dataframe():
    """Test signal DataFrame creation."""
    config_dict = {
        "strategy_name": "test_strategy",
        "market": {"exchange": "binance", "symbol": "BTCUSDT"},
        "timeframes": {"signal_tf": "1h", "entry_tf": "15m"},
        "moving_averages": {"ema5": {"enabled": True, "length": 5}},
        "alignment_rules": {
            "long": {"macd_bars_signal_tf": 1, "macd_bars_entry_tf": 2},
            "short": {"macd_bars_signal_tf": 1, "macd_bars_entry_tf": 2},
        },
    }
    config = StrategyConfig(**config_dict)
    strategy = ConcreteStrategy(config)
    
    signals = strategy._create_signal_dataframe()
    assert isinstance(signals, pd.DataFrame)
    assert len(signals) == 0
    # Note: index name will be 'timestamp' after set_index


