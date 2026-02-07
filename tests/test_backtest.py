"""Tests for backtest engine.

Note: These tests validate the current public behavior of BacktestEngine.
The engine API evolved; older helper methods (load_data/apply_costs) were removed
in favor of running via BacktestEngine.run() and using BrokerModel internally.
"""

import pandas as pd
import numpy as np

from engine.backtest_engine import BacktestEngine, BacktestResult, Position
from strategies.base import StrategyBase
from config.schema import StrategyConfig


class DummyStrategy(StrategyBase):
    """Dummy strategy for testing."""
    
    def generate_signals(self, df_by_tf):
        entry_tf = self.config.timeframes.entry_tf
        df_entry = df_by_tf[entry_tf]
        ts = df_entry.index[0]
        first_close = float(df_entry.iloc[0]["close"])

        # Use immediate_execution so this test doesn't depend on entry-gating config.
        signal = {
            "timestamp": ts,
            "direction": "long",
            "entry_price": first_close,
            "stop_price": first_close * 0.99,
            "weight": 1.0,
            "metadata": {},
            "immediate_execution": True,
        }
        return pd.DataFrame([signal]).set_index("timestamp")

    def get_indicators(self, df, tf=None):
        return df


class NoStopStrategy(DummyStrategy):
    """Strategy that fails to compute stop-loss (forces engine fallback when enabled)."""

    def generate_signals(self, df_by_tf):
        entry_tf = self.config.timeframes.entry_tf
        df_entry = df_by_tf[entry_tf]
        # Emit signal after enough history for ATR(14)
        ts = df_entry.index[15]
        close_ = float(df_entry.loc[ts]["close"])

        signal = {
            "timestamp": ts,
            "direction": "long",
            "entry_price": close_,
            "stop_price": None,
            "weight": 1.0,
            "metadata": {},
            "immediate_execution": True,
        }
        return pd.DataFrame([signal]).set_index("timestamp")

    def _calculate_stop_loss(self, entry_row, dir_int):
        return None


def test_backtest_engine_initialization():
    """Test backtest engine initialization."""
    config_dict = {
        "strategy_name": "test_strategy",
        "market": {"exchange": "binance", "symbol": "BTCUSDT", "base_timeframe": "1m"},
        "timeframes": {"signal_tf": "1h", "entry_tf": "15m"},
        "moving_averages": {"ema5": {"enabled": True, "length": 5}},
        "alignment_rules": {
            "long": {"macd_bars_signal_tf": 1, "macd_bars_entry_tf": 2},
            "short": {"macd_bars_signal_tf": 1, "macd_bars_entry_tf": 2},
        },
    }
    config = StrategyConfig(**config_dict)
    strategy = DummyStrategy(config)
    
    engine = BacktestEngine(strategy, initial_capital=10000.0)
    
    assert engine.initial_capital == 10000.0
    assert engine.strategy == strategy


def test_backtest_engine_run_with_dataframe():
    """Smoke test: engine runs on an in-memory DataFrame and produces a result."""
    idx = pd.date_range(start="2024-01-01 00:00:00", periods=10, freq="1h")
    base = 100.0 + np.arange(len(idx)) * 0.1
    df = pd.DataFrame(
        {
            "open": base,
            "high": base + 0.2,
            "low": base - 0.2,
            "close": base + 0.1,
            "volume": 1000,
        },
        index=idx,
    )

    config_dict = {
        "strategy_name": "test_strategy",
        "market": {"exchange": "binance", "symbol": "BTCUSDT", "base_timeframe": "1h"},
        "timeframes": {"signal_tf": "1h", "entry_tf": "1h"},
        "moving_averages": {"ema5": {"enabled": True, "length": 5}},
        "alignment_rules": {
            "long": {"macd_bars_signal_tf": 0, "macd_bars_entry_tf": 0},
            "short": {"macd_bars_signal_tf": 0, "macd_bars_entry_tf": 0},
        },
        "execution": {"fill_timing": "same_close"},
        "backtest": {"commissions": 0.0, "slippage_ticks": 0.0},
        "risk": {"sizing_mode": "account_size", "risk_per_trade_pct": 1.0, "account_size": 10000.0},
    }
    config = StrategyConfig(**config_dict)
    strategy = DummyStrategy(config)
    engine = BacktestEngine(strategy, initial_capital=10000.0, enforce_margin_checks=False)

    result = engine.run(df)

    assert isinstance(result, BacktestResult)
    assert result.strategy_name == "test_strategy"
    assert result.initial_capital == 10000.0
    assert len(result.trades) == 1


def test_calculate_position_size():
    """Test position size calculation."""
    config_dict = {
        "strategy_name": "test_strategy",
        "market": {"exchange": "binance", "symbol": "BTCUSDT", "base_timeframe": "1m"},
        "timeframes": {"signal_tf": "1h", "entry_tf": "15m"},
        "moving_averages": {"ema5": {"enabled": True, "length": 5}},
        "alignment_rules": {
            "long": {"macd_bars_signal_tf": 1, "macd_bars_entry_tf": 2},
            "short": {"macd_bars_signal_tf": 1, "macd_bars_entry_tf": 2},
        },
    }
    config = StrategyConfig(**config_dict)
    strategy = DummyStrategy(config)
    engine = BacktestEngine(strategy, initial_capital=10000.0)

    # Current engine exposes the sizing logic as an internal helper.
    size = engine._calculate_position_size(100.0, 99.0, 1.0)
    assert size > 0


def test_apply_costs():
    """Test cost application."""
    config_dict = {
        "strategy_name": "test_strategy",
        "market": {"exchange": "binance", "symbol": "BTCUSDT", "base_timeframe": "1m"},
        "timeframes": {"signal_tf": "1h", "entry_tf": "15m"},
        "moving_averages": {"ema5": {"enabled": True, "length": 5}},
        "alignment_rules": {
            "long": {"macd_bars_signal_tf": 1, "macd_bars_entry_tf": 2},
            "short": {"macd_bars_signal_tf": 1, "macd_bars_entry_tf": 2},
        },
    }
    config = StrategyConfig(**config_dict)
    strategy = DummyStrategy(config)
    engine = BacktestEngine(strategy, commission_rate=0.0004, slippage_ticks=0.1)

    price = engine.broker.apply_slippage(100.0, 1.0, is_entry=True)
    commission = engine.broker.calculate_commission(price, 1.0)
    assert price > 100.0  # Slippage added
    assert commission > 0  # Commission


def test_backtest_result_creation():
    """Test BacktestResult creation."""
    result = BacktestResult(
        trades=[],
        equity_curve=pd.Series([10000.0]),
        initial_capital=10000.0,
        final_capital=10000.0,
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        total_pnl=0.0,
        total_commission=0.0,
        total_slippage=0.0,
        max_drawdown=0.0,
        win_rate=0.0,
    )
    
    assert result.initial_capital == 10000.0
    assert result.total_trades == 0


def test_portfolio_open_risk_cap_scales_down_size():
    config_dict = {
        "strategy_name": "test_strategy",
        "market": {"exchange": "binance", "symbol": "BTCUSDT", "base_timeframe": "1m"},
        "timeframes": {"signal_tf": "1h", "entry_tf": "15m"},
        "moving_averages": {"ema5": {"enabled": True, "length": 5}},
        "alignment_rules": {
            "long": {"macd_bars_signal_tf": 1, "macd_bars_entry_tf": 2},
            "short": {"macd_bars_signal_tf": 1, "macd_bars_entry_tf": 2},
        },
        "risk": {"sizing_mode": "account_size", "risk_per_trade_pct": 1.0, "account_size": 10000.0},
        "portfolio": {"enabled": True, "max_open_risk_pct": 2.0},
    }
    config = StrategyConfig(**config_dict)
    strategy = DummyStrategy(config)
    engine = BacktestEngine(strategy, initial_capital=10000.0, enforce_margin_checks=False)

    # Existing open position consumes $150 of open risk (price_risk=1, qty=150)
    engine.positions = [
        Position(
            instrument="BTCUSDT",
            direction="long",
            size=150.0,
            entry_price=100.0,
            stop_price=99.0,
        )
    ]

    # New trade would normally risk $100 (1% of 10k) => qty=100
    # Portfolio cap is $200 (2% of 10k) with $150 already used => remaining $50 => qty=50
    size = engine._calculate_position_size(100.0, 99.0, 1.0)
    assert np.isclose(size, 50.0)

    # If open risk already exceeds cap, new sizing should return 0
    engine.positions = [
        Position(
            instrument="BTCUSDT",
            direction="long",
            size=250.0,
            entry_price=100.0,
            stop_price=99.0,
        )
    ]
    size2 = engine._calculate_position_size(100.0, 99.0, 1.0)
    assert size2 == 0.0


def test_atr_stop_fallback_used_when_stop_missing():
    # Enough bars for ATR(14)
    idx = pd.date_range(start="2024-01-01 00:00:00", periods=20, freq="1h")
    base = 100.0 + np.arange(len(idx)) * 0.5
    df = pd.DataFrame(
        {
            "open": base,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base + 0.2,
            "volume": 1000,
        },
        index=idx,
    )

    config_dict = {
        "strategy_name": "test_strategy",
        "market": {"exchange": "binance", "symbol": "BTCUSDT", "base_timeframe": "1h"},
        "timeframes": {"signal_tf": "1h", "entry_tf": "1h"},
        "moving_averages": {"ema5": {"enabled": True, "length": 5}},
        "alignment_rules": {
            "long": {"macd_bars_signal_tf": 0, "macd_bars_entry_tf": 0},
            "short": {"macd_bars_signal_tf": 0, "macd_bars_entry_tf": 0},
        },
        "execution": {"fill_timing": "same_close"},
        "backtest": {"commissions": 0.0, "slippage_ticks": 0.0},
        "risk": {"sizing_mode": "account_size", "risk_per_trade_pct": 1.0, "account_size": 10000.0},
        "portfolio": {
            "enabled": True,
            "stop_fallback_enabled": True,
            "stop_fallback_atr_period": 14,
            "stop_fallback_atr_multiplier": 3.0,
        },
    }
    config = StrategyConfig(**config_dict)
    strategy = NoStopStrategy(config)
    engine = BacktestEngine(strategy, initial_capital=10000.0, enforce_margin_checks=False)

    result = engine.run(df)
    assert len(result.trades) == 1
    assert result.trades[0].stop_price is not None


