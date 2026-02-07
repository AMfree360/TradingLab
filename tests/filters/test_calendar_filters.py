import pandas as pd

from config.schema import StrategyConfig
from strategies.base.strategy_base import StrategyBase


class _OneSignalStrategy(StrategyBase):
    def generate_signals(self, df_by_tf):
        raise NotImplementedError()

    def get_indicators(self, df, tf=None):
        return df


def _base_config_dict():
    return {
        "strategy_name": "test",
        "market": {"exchange": "binance", "symbol": "BTCUSDT", "base_timeframe": "1m"},
        "timeframes": {"signal_tf": "1h", "entry_tf": "15m"},
        "moving_averages": {"ema5": {"enabled": True, "length": 5}},
        "news_filter": {"enabled": False},
        "regime_filters": {
            "adx": {"enabled": False},
            "atr_percentile": {"enabled": False},
            "ema_expansion": {"enabled": False},
            "composite": {
                "enabled": False,
                "use_adx": False,
                "use_atr_percentile": False,
                "use_ema_expansion": False,
                "use_swing": False,
            },
        },
        "alignment_rules": {
            "long": {"macd_bars_signal_tf": 0, "macd_bars_entry_tf": 0},
            "short": {"macd_bars_signal_tf": 0, "macd_bars_entry_tf": 0},
        },
    }


def test_day_of_week_filter_blocks_weekend_when_enabled():
    cfg = _base_config_dict()
    cfg["calendar_filters"] = {
        "master_filters_enabled": True,
        "day_of_week": {"enabled": True, "allowed_days": [0, 1, 2, 3, 4]},
    }
    config = StrategyConfig(**cfg)
    strategy = _OneSignalStrategy(config)

    saturday = pd.Timestamp("2026-02-07 12:00:00")  # Saturday
    passed = strategy.apply_filters(
        signal=pd.Series({"direction": "long", "entry_price": 100.0, "stop_price": 99.0}),
        timestamp=saturday,
        symbol="BTCUSDT",
        df_by_tf={},
    )
    assert passed is False


def test_trading_session_filter_allows_only_inside_session_when_enabled():
    cfg = _base_config_dict()
    cfg["calendar_filters"] = {
        "master_filters_enabled": True,
        "trading_sessions_enabled": True,
        "trading_sessions": {
            # Choose a window that doesn't overlap the default master sessions
            # (London 07:00-16:00, NewYork 13:00-21:00, Asia 23:00-08:00)
            "Test": {"enabled": True, "start": "21:30", "end": "22:30"},
        },
    }
    config = StrategyConfig(**cfg)
    strategy = _OneSignalStrategy(config)

    inside = pd.Timestamp("2026-02-06 22:00:00")
    outside = pd.Timestamp("2026-02-06 22:45:00")

    signal = pd.Series({"direction": "long", "entry_price": 100.0, "stop_price": 99.0})

    assert strategy.apply_filters(signal=signal, timestamp=inside, symbol="BTCUSDT", df_by_tf={}) is True
    assert strategy.apply_filters(signal=signal, timestamp=outside, symbol="BTCUSDT", df_by_tf={}) is False


def test_trading_session_filter_does_not_apply_when_not_explicitly_enabled():
    cfg = _base_config_dict()
    cfg["calendar_filters"] = {
        "master_filters_enabled": True,
        "trading_sessions_enabled": False,
        "trading_sessions": {
            "Test": {"enabled": True, "start": "09:00", "end": "10:00"},
        },
    }
    config = StrategyConfig(**cfg)
    strategy = _OneSignalStrategy(config)

    outside = pd.Timestamp("2026-02-06 10:30:00")
    signal = pd.Series({"direction": "long", "entry_price": 100.0, "stop_price": 99.0})
    assert strategy.apply_filters(signal=signal, timestamp=outside, symbol="BTCUSDT", df_by_tf={}) is True


def test_master_filters_enabled_false_short_circuits_filter_application():
    cfg = _base_config_dict()
    cfg["calendar_filters"] = {
        "master_filters_enabled": False,
        "trading_sessions_enabled": True,
        "trading_sessions": {
            "Test": {"enabled": True, "start": "09:00", "end": "10:00"},
        },
        "day_of_week": {"enabled": True, "allowed_days": [0, 1, 2, 3, 4]},
    }
    config = StrategyConfig(**cfg)
    strategy = _OneSignalStrategy(config)

    saturday = pd.Timestamp("2026-02-07 12:00:00")
    outside = pd.Timestamp("2026-02-06 10:30:00")
    signal = pd.Series({"direction": "long", "entry_price": 100.0, "stop_price": 99.0})

    assert strategy.apply_filters(signal=signal, timestamp=saturday, symbol="BTCUSDT", df_by_tf={}) is True
    assert strategy.apply_filters(signal=signal, timestamp=outside, symbol="BTCUSDT", df_by_tf={}) is True
