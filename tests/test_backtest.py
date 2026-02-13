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


class MultiSignalStrategy(StrategyBase):
    """Strategy that emits multiple immediate-execution signals at specified timestamps."""

    def __init__(self, config: StrategyConfig, signal_times: list[pd.Timestamp]):
        super().__init__(config)
        self._signal_times = list(signal_times)

    def generate_signals(self, df_by_tf):
        entry_tf = self.config.timeframes.entry_tf
        df_entry = df_by_tf[entry_tf]
        rows = []

        for ts in self._signal_times:
            if ts not in df_entry.index:
                continue
            close_ = float(df_entry.loc[ts]["close"])
            rows.append(
                {
                    "timestamp": ts,
                    "direction": "long",
                    "entry_price": close_,
                    "stop_price": close_ - 5.0,
                    "weight": 1.0,
                    "metadata": {},
                    "immediate_execution": True,
                }
            )

        return pd.DataFrame(rows).set_index("timestamp")

    def get_indicators(self, df, tf=None):
        return df


class TwoSignalStrategy(StrategyBase):
    """Emits two immediate-execution signals on different bars."""

    def generate_signals(self, df_by_tf):
        entry_tf = self.config.timeframes.entry_tf
        df_entry = df_by_tf[entry_tf]
        ts0 = df_entry.index[0]
        ts2 = df_entry.index[2]

        c0 = float(df_entry.loc[ts0]["close"])
        c2 = float(df_entry.loc[ts2]["close"])

        signals = [
            {
                "timestamp": ts0,
                "direction": "long",
                "entry_price": c0,
                "stop_price": c0 * 0.96,  # 4% stop
                "weight": 1.0,
                "metadata": {},
                "immediate_execution": True,
            },
            {
                "timestamp": ts2,
                "direction": "long",
                "entry_price": c2,
                "stop_price": c2 * 0.99,
                "weight": 1.0,
                "metadata": {},
                "immediate_execution": True,
            },
        ]
        return pd.DataFrame(signals).set_index("timestamp")

    def get_indicators(self, df, tf=None):
        return df


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


def test_max_daily_loss_pct_blocks_new_entries_same_day():
    """After a large realized loss, engine should stop taking new entries for that day."""
    idx = pd.date_range(start="2024-01-01 00:00:00", periods=6, freq="1h")
    # Price drops below stop on the second bar to force a stop-out.
    df = pd.DataFrame(
        {
            "open": [100, 100, 96, 96, 96, 96],
            "high": [101, 101, 97, 97, 97, 97],
            "low": [99, 94, 95, 95, 95, 95],
            "close": [100, 95, 96, 96, 96, 96],
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
        "risk": {"sizing_mode": "account_size", "risk_per_trade_pct": 5.0, "account_size": 10000.0},
        "trade_limits": {"max_trades_per_day": 10, "max_daily_loss_pct": 3.0},
    }
    config = StrategyConfig(**config_dict)

    # Two signals on the same day. The first should stop out; the second should be blocked.
    strategy = MultiSignalStrategy(config, signal_times=[idx[0], idx[3]])
    engine = BacktestEngine(strategy, initial_capital=10000.0, enforce_margin_checks=False)
    result = engine.run(df)

    # Expect only one ENTRY trade record (plus any partials, which we don't use here).
    entry_trades = [t for t in result.trades if t.exit_reason != 'partial']
    assert len(entry_trades) == 1
    assert entry_trades[0].pnl_after_costs < 0


def test_daily_equity_sizing_constant_within_day():
    """daily_equity sizing should not increase after intraday gains."""
    idx = pd.date_range(start="2024-01-01 00:00:00", periods=8, freq="1h")
    # Construct bars to hit 1R take-profit twice.
    df = pd.DataFrame(
        {
            "open": [100, 100, 101, 101, 100, 100, 101, 101],
            "high": [101, 102, 102, 102, 101, 102, 102, 102],
            "low": [99, 99, 100, 100, 99, 99, 100, 100],
            "close": [100, 101, 101, 101, 100, 101, 101, 101],
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
        "risk": {"sizing_mode": "daily_equity", "risk_per_trade_pct": 1.0, "account_size": 10000.0},
        "trade_limits": {"max_trades_per_day": 10, "max_daily_loss_pct": None},
        # Make take-profit easy to hit: 1R
        "take_profit": {"enabled": True, "levels": [{"type": "r_based", "target_r": 1.0, "exit_pct": 100.0}]},
    }
    config = StrategyConfig(**config_dict)

    # Two signals in the same day; both should size off the SAME day-start base.
    strategy = MultiSignalStrategy(config, signal_times=[idx[0], idx[4]])
    engine = BacktestEngine(strategy, initial_capital=10000.0, enforce_margin_checks=False)
    result = engine.run(df)

    entry_trades = [t for t in result.trades if t.exit_reason != 'partial']
    assert len(entry_trades) == 2
    assert np.isclose(entry_trades[0].quantity, entry_trades[1].quantity)


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


def test_default_max_trades_per_day_is_1_blocks_second_entry():
    idx = pd.date_range(start="2024-01-01 00:00:00", periods=6, freq="1h")
    closes = np.array([100.0, 95.0, 110.0, 111.0, 112.0, 113.0])
    df = pd.DataFrame(
        {
            "open": closes,
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
            "volume": 1000,
        },
        index=idx,
    )

    # Do NOT specify trade_limits; schema defaults should apply (max_trades_per_day=1)
    config_dict = {
        "strategy_name": "two_signal",
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
    strategy = TwoSignalStrategy(config)
    engine = BacktestEngine(strategy, initial_capital=10000.0, enforce_margin_checks=False)

    result = engine.run(df)
    assert len(result.trades) == 1


def test_max_daily_loss_pct_blocks_further_entries_same_day():
    idx = pd.date_range(start="2024-01-01 00:00:00", periods=6, freq="1h")
    closes = np.array([100.0, 95.0, 110.0, 111.0, 112.0, 113.0])
    df = pd.DataFrame(
        {
            "open": closes,
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
            "volume": 1000,
        },
        index=idx,
    )

    config_dict = {
        "strategy_name": "two_signal_daily_loss",
        "market": {"exchange": "binance", "symbol": "BTCUSDT", "base_timeframe": "1h"},
        "timeframes": {"signal_tf": "1h", "entry_tf": "1h"},
        "moving_averages": {"ema5": {"enabled": True, "length": 5}},
        "alignment_rules": {
            "long": {"macd_bars_signal_tf": 0, "macd_bars_entry_tf": 0},
            "short": {"macd_bars_signal_tf": 0, "macd_bars_entry_tf": 0},
        },
        "execution": {"fill_timing": "same_close"},
        "backtest": {"commissions": 0.0, "slippage_ticks": 0.0},
        # First trade is designed to lose ~4% of base at stop; daily loss cap is 3%.
        "risk": {"sizing_mode": "account_size", "risk_per_trade_pct": 4.0, "account_size": 10000.0},
        "trade_limits": {"max_trades_per_day": 10, "max_daily_loss_pct": 3.0},
    }
    config = StrategyConfig(**config_dict)
    strategy = TwoSignalStrategy(config)
    engine = BacktestEngine(strategy, initial_capital=10000.0, enforce_margin_checks=False)

    result = engine.run(df)
    assert len(result.trades) == 1
    assert result.trades[0].pnl_after_costs < 0


def test_daily_equity_sizing_uses_day_start_base():
    config_dict = {
        "strategy_name": "sizing_daily",
        "market": {"exchange": "binance", "symbol": "BTCUSDT", "base_timeframe": "1m"},
        "timeframes": {"signal_tf": "1h", "entry_tf": "15m"},
        "moving_averages": {"ema5": {"enabled": True, "length": 5}},
        "alignment_rules": {
            "long": {"macd_bars_signal_tf": 1, "macd_bars_entry_tf": 2},
            "short": {"macd_bars_signal_tf": 1, "macd_bars_entry_tf": 2},
        },
        "risk": {"sizing_mode": "daily_equity", "risk_per_trade_pct": 1.0, "account_size": 10000.0},
    }
    config = StrategyConfig(**config_dict)
    strategy = DummyStrategy(config)
    engine = BacktestEngine(strategy, initial_capital=10000.0, enforce_margin_checks=False)

    day = pd.Timestamp("2024-01-01 00:00:00")
    engine._day_start_equity_by_day[day.normalize()] = 10000.0
    engine._current_day_key = day.normalize()

    size1 = engine._calculate_position_size(100.0, 99.0, 1.0, current_time=day)
    engine.account.cash = 20000.0  # change equity intraday
    size2 = engine._calculate_position_size(100.0, 99.0, 1.0, current_time=day + pd.Timedelta(hours=1))

    assert np.isclose(size1, 100.0)
    assert np.isclose(size2, 100.0)


