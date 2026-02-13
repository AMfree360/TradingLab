from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from config.schema import validate_strategy_config
from engine.backtest_engine import BacktestEngine
from research.compiler import StrategyCompiler


def _load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _make_ohlcv_1m(
    *,
    start: str = "2024-01-01",
    minutes: int = 2000,
    close: np.ndarray | None = None,
) -> pd.DataFrame:
    idx = pd.date_range(start, periods=int(minutes), freq="1min")
    if close is None:
        close = 100.0 + np.linspace(0.0, 10.0, int(minutes))
    close = np.asarray(close, dtype=float)
    assert len(close) == len(idx)

    df = pd.DataFrame(index=idx)
    df["close"] = close
    df["open"] = np.r_[close[0], close[:-1]]
    df["high"] = np.maximum(df["open"].to_numpy(), df["close"].to_numpy()) + 1.0
    df["low"] = np.minimum(df["open"].to_numpy(), df["close"].to_numpy()) - 1.0
    df["volume"] = 1.0
    return df


def _compile_and_load_strategy(tmp_path: Path, spec_rel_path: str):
    repo_root = Path(__file__).resolve().parents[1]
    compiler = StrategyCompiler(repo_root=repo_root)
    spec_path = repo_root / spec_rel_path
    spec = compiler.load_spec(spec_path)
    out_dir = compiler.compile_to_folder(spec, out_dir=tmp_path / "out")

    config = yaml.safe_load((out_dir / "config.yml").read_text(encoding="utf-8"))
    cfg_obj = validate_strategy_config(config)

    mod = _load_module_from_path("_generated_smoke_strategy", out_dir / "strategy.py")
    StrategyCls = getattr(mod, "GeneratedResearchStrategy")
    return StrategyCls(cfg_obj)


def _first_trade_and_decision_row(result, direction: str):
    trades = [t for t in (result.trades or []) if t.direction == direction]
    assert trades, f"Expected at least one {direction} trade"
    trade = trades[0]
    df_entry = result.price_df
    assert df_entry is not None and len(df_entry) > 5
    assert trade.entry_time in df_entry.index
    entry_idx = int(df_entry.index.get_loc(trade.entry_time))
    assert entry_idx >= 1
    decision_row = df_entry.iloc[entry_idx - 1]
    return trade, decision_row


def test_stop_ma_buffer_backtest_smoke(tmp_path: Path):
    # Spec: MA stop on 15m with 25.0 absolute buffer (price units)
    strategy = _compile_and_load_strategy(tmp_path, "research_specs/example_stop_ma_buffer.yml")
    df = _make_ohlcv_1m(minutes=3000, close=10000.0 + np.linspace(0, 200, 3000))

    engine = BacktestEngine(strategy=strategy, initial_capital=10000.0, commission_rate=None, slippage_ticks=None)
    result = engine.run(df)

    trade, decision_row = _first_trade_and_decision_row(result, direction="long")
    assert trade.stop_price is not None
    expected = float(decision_row["ema_close_50"]) - 25.0
    assert abs(float(trade.stop_price) - expected) < 1e-9


def test_stop_structure_backtest_smoke(tmp_path: Path):
    # Spec: structure stop on 30m using Donchian lookback=20 with 10.0 absolute buffer
    strategy = _compile_and_load_strategy(tmp_path, "research_specs/example_stop_structure.yml")

    minutes = 30 * 50
    close = np.full(minutes, 100.0)
    # Force a breakout in a later 30m bar so crosses_above can trigger.
    breakout_start = 30 * 25
    close[breakout_start:] = 150.0
    df = _make_ohlcv_1m(minutes=minutes, close=close)

    engine = BacktestEngine(strategy=strategy, initial_capital=10000.0, commission_rate=None, slippage_ticks=None)
    result = engine.run(df)

    trade, decision_row = _first_trade_and_decision_row(result, direction="long")
    assert trade.stop_price is not None
    expected = float(decision_row["donchian_low_20"]) - 10.0
    assert abs(float(trade.stop_price) - expected) < 1e-9


def test_stop_candle_backtest_smoke(tmp_path: Path):
    # Spec: candle stop on 5m using prior candle high/low (bars_back=1) + 5.0 absolute buffer
    strategy = _compile_and_load_strategy(tmp_path, "research_specs/example_stop_candle.yml")

    minutes = 5 * 400
    close = np.full(minutes, 100.0)
    # Create an upswing so EMA9/EMA21 crosses_above triggers on 5m.
    close[5 * 250 :] = 100.0 + np.linspace(0.0, 50.0, minutes - 5 * 250)
    df = _make_ohlcv_1m(minutes=minutes, close=close)

    engine = BacktestEngine(strategy=strategy, initial_capital=10000.0, commission_rate=None, slippage_ticks=None)
    result = engine.run(df)

    trade, decision_row = _first_trade_and_decision_row(result, direction="long")
    assert trade.stop_price is not None
    expected = float(decision_row["low_shift_1"]) - 5.0
    assert abs(float(trade.stop_price) - expected) < 1e-9


def test_stop_separate_sides_short_uses_structure(tmp_path: Path):
    # Spec: separate LONG/SHORT stops; verify SHORT uses structure stop (Donchian high + buffer)
    strategy = _compile_and_load_strategy(tmp_path, "research_specs/example_stop_separate_sides.yml")

    minutes = 15 * 200
    close = 10000.0 - np.linspace(0, 500, minutes)
    df = _make_ohlcv_1m(minutes=minutes, close=close)

    engine = BacktestEngine(strategy=strategy, initial_capital=10000.0, commission_rate=None, slippage_ticks=None)
    result = engine.run(df)

    trade, decision_row = _first_trade_and_decision_row(result, direction="short")
    assert trade.stop_price is not None
    expected = float(decision_row["donchian_high_20"]) + 10.0
    assert abs(float(trade.stop_price) - expected) < 1e-9
