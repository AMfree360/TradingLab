"""Tests for configuration system."""

import pytest
from pathlib import Path
import yaml
from config.schema import (
    StrategyConfig,
    validate_strategy_config,
    load_and_validate_strategy_config,
    load_defaults,
)


def test_load_defaults():
    """Test loading default configuration."""
    defaults = load_defaults()
    assert isinstance(defaults, dict)
    assert "backtest" in defaults
    assert "risk" in defaults


def test_strategy_config_validation():
    """Test strategy configuration validation."""
    config_dict = {
        "strategy_name": "test_strategy",
        "market": {
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "base_timeframe": "1m",
        },
        "timeframes": {
            "signal_tf": "1h",
            "entry_tf": "15m",
        },
        "moving_averages": {
            "ema5": {"enabled": True, "length": 5},
            "ema15": {"enabled": True, "length": 15},
        },
        "macd": {
            "fast": 15,
            "slow": 30,
            "signal": 9,
        },
        "alignment_rules": {
            "long": {
                "macd_bars_signal_tf": 1,
                "macd_bars_entry_tf": 2,
            },
            "short": {
                "macd_bars_signal_tf": 1,
                "macd_bars_entry_tf": 2,
            },
        },
    }
    
    config = validate_strategy_config(config_dict)
    assert config.strategy_name == "test_strategy"
    assert config.market.exchange == "binance"
    assert config.timeframes.signal_tf == "1h"
    assert config.moving_averages["ema5"].length == 5


def test_strategy_config_defaults():
    """Test that defaults are applied correctly."""
    config_dict = {
        "strategy_name": "test_strategy",
        "market": {
            "exchange": "binance",
            "symbol": "BTCUSDT",
        },
        "timeframes": {
            "signal_tf": "1h",
            "entry_tf": "15m",
        },
        "moving_averages": {
            "ema5": {"enabled": True, "length": 5},
        },
        "alignment_rules": {
            "long": {"macd_bars_signal_tf": 1, "macd_bars_entry_tf": 2},
            "short": {"macd_bars_signal_tf": 1, "macd_bars_entry_tf": 2},
        },
    }
    
    config = validate_strategy_config(config_dict)
    # Check defaults
    assert config.macd.fast == 15
    assert config.macd.slow == 30
    assert config.macd.signal == 9
    assert config.stop_loss.type == "SMA"
    assert config.risk.sizing_mode == "account_size"
    assert config.risk.risk_per_trade_pct == 1.0


def test_strategy_config_validation_errors():
    """Test that validation catches errors."""
    config_dict = {
        "strategy_name": "test_strategy",
        "market": {
            "exchange": "binance",
            "symbol": "BTCUSDT",
        },
        "timeframes": {
            "signal_tf": "1h",
            "entry_tf": "15m",
        },
        "moving_averages": {
            "ema5": {"enabled": True, "length": -5},  # Invalid: negative length
        },
        "alignment_rules": {
            "long": {"macd_bars_signal_tf": 1, "macd_bars_entry_tf": 2},
            "short": {"macd_bars_signal_tf": 1, "macd_bars_entry_tf": 2},
        },
    }
    
    with pytest.raises(Exception):  # Should raise validation error
        validate_strategy_config(config_dict)


