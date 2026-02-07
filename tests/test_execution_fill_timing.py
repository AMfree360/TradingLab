import pandas as pd

from config.schema import StrategyConfig
from engine.backtest_engine import BacktestEngine
from engine.market import MarketSpec
from strategies.base import StrategyBase


class ImmediateSignalStrategy(StrategyBase):
    def generate_signals(self, df_by_tf):
        # Emit a single immediate signal at the first bar of the entry timeframe.
        entry_tf = self.config.timeframes.entry_tf
        df = df_by_tf[entry_tf]
        ts = df.index[0]
        signal = {
            "timestamp": ts,
            "direction": "long",
            "entry_price": float(df.iloc[0]["close"]),
            "stop_price": float(df.iloc[0]["close"]) * 0.99,
            "weight": 1.0,
            "metadata": {},
            "immediate_execution": True,
        }
        out = pd.DataFrame([signal]).set_index("timestamp")
        return out

    def get_indicators(self, df, tf=None):
        return df


def test_fill_timing_next_open_is_enforced():
    # 3 bars so a next_open fill is always possible
    idx = pd.date_range("2024-01-01 00:00:00", periods=3, freq="1h")
    df = pd.DataFrame(
        {
            "open": [100.0, 105.0, 106.0],
            "high": [101.0, 106.0, 107.0],
            "low": [99.0, 104.0, 105.0],
            "close": [100.0, 105.5, 106.5],
            "volume": [1000, 1000, 1000],
        },
        index=idx,
    )

    config = StrategyConfig(
        **{
            "strategy_name": "immediate_signal",
            "market": {"exchange": "test", "symbol": "TEST", "base_timeframe": "1h"},
            "timeframes": {"signal_tf": "1h", "entry_tf": "1h"},
            "moving_averages": {"ema5": {"enabled": True, "length": 5}},
            "alignment_rules": {
                "long": {"macd_bars_signal_tf": 0, "macd_bars_entry_tf": 0},
                "short": {"macd_bars_signal_tf": 0, "macd_bars_entry_tf": 0},
            },
            "execution": {"fill_timing": "next_open"},
            "risk": {"sizing_mode": "account_size", "risk_per_trade_pct": 1.0, "account_size": 10000.0},
            "backtest": {"commissions": 0.0, "slippage_ticks": 0.0},
        }
    )

    strategy = ImmediateSignalStrategy(config)
    market_spec = MarketSpec(
        symbol="TEST",
        exchange="test",
        asset_class="crypto",
        market_type="spot",
        leverage=1.0,
        commission_rate=0.0,
        slippage_ticks=0.0,
    )

    engine = BacktestEngine(strategy=strategy, market_spec=market_spec, initial_capital=10000.0, enforce_margin_checks=False)
    result = engine.run(df)

    assert len(result.trades) == 1
    trade = result.trades[0]

    # Decision is on bar 0 close; fill must be on bar 1 open
    assert trade.entry_time == idx[1]
    assert trade.entry_price == df.iloc[1]["open"]
