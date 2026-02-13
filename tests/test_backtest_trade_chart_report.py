from __future__ import annotations

import pandas as pd

from engine.backtest_engine import BacktestResult, Trade
from reports.report_generator import ReportGenerator


def test_backtest_report_includes_trade_chart_when_visuals_enabled(tmp_path):
    idx = pd.date_range("2024-01-01", periods=10, freq="1h")
    df = pd.DataFrame(
        {
            "open": [100 + i for i in range(10)],
            "high": [101 + i for i in range(10)],
            "low": [99 + i for i in range(10)],
            "close": [100.5 + i for i in range(10)],
            "ema_close_20": [100.2 + i for i in range(10)],
        },
        index=idx,
    )

    trades = [
        Trade(
            entry_time=idx[2],
            exit_time=idx[5],
            direction="long",
            entry_price=float(df.loc[idx[2], "close"]),
            exit_price=float(df.loc[idx[5], "close"]),
            quantity=1.0,
            pnl_after_costs=1.0,
            stop_price=95.0,
            exit_reason="signal_exit",
            partial_exits=[
                {
                    "exit_time": idx[4],
                    "exit_price": float(df.loc[idx[4], "close"]),
                    "quantity": 0.5,
                    "pnl": 0.5,
                }
            ],
        )
    ]

    result = BacktestResult(
        initial_capital=10000.0,
        final_capital=10001.0,
        total_trades=len(trades),
        winning_trades=1,
        losing_trades=0,
        total_pnl=1.0,
        trades=trades,
        entry_tf="1h",
        price_df=df,
    )

    rg = ReportGenerator(output_dir=tmp_path)
    html_path = rg.generate_backtest_report(result, include_visuals=True, output_name="x")
    html = html_path.read_text(encoding="utf-8")

    # Default engine is lightweight-charts.
    assert "trade-chart-lw" in html
    assert "LightweightCharts.createChart" in html
    assert "var partials" in html


def test_backtest_report_can_force_plotly_trade_chart(tmp_path):
    idx = pd.date_range("2024-01-01", periods=10, freq="1h")
    df = pd.DataFrame(
        {
            "open": [100 + i for i in range(10)],
            "high": [101 + i for i in range(10)],
            "low": [99 + i for i in range(10)],
            "close": [100.5 + i for i in range(10)],
            "ema_close_20": [100.2 + i for i in range(10)],
        },
        index=idx,
    )

    trades = [
        Trade(
            entry_time=idx[2],
            exit_time=idx[5],
            direction="long",
            entry_price=float(df.loc[idx[2], "close"]),
            exit_price=float(df.loc[idx[5], "close"]),
            quantity=1.0,
            pnl_after_costs=1.0,
            stop_price=95.0,
            exit_reason="signal_exit",
            partial_exits=[
                {
                    "exit_time": idx[4],
                    "exit_price": float(df.loc[idx[4], "close"]),
                    "quantity": 0.5,
                    "pnl": 0.5,
                }
            ],
        )
    ]

    result = BacktestResult(
        initial_capital=10000.0,
        final_capital=10001.0,
        total_trades=len(trades),
        winning_trades=1,
        losing_trades=0,
        total_pnl=1.0,
        trades=trades,
        entry_tf="1h",
        price_df=df,
    )

    rg = ReportGenerator(output_dir=tmp_path, trade_chart_engine="plotly")
    html_path = rg.generate_backtest_report(result, include_visuals=True, output_name="x_plotly")
    html = html_path.read_text(encoding="utf-8")

    assert "trade-chart" in html
    assert "candlestick" in html
    assert "Partial Exits" in html


def test_trade_chart_honors_report_overlay_cols_and_excludes_sl_ma(tmp_path):
    idx = pd.date_range("2024-01-01", periods=10, freq="1h")
    df = pd.DataFrame(
        {
            "open": [100 + i for i in range(10)],
            "high": [101 + i for i in range(10)],
            "low": [99 + i for i in range(10)],
            "close": [100.5 + i for i in range(10)],
            "ema_close_20": [100.2 + i for i in range(10)],
            # Present, but should not be plotted unless explicitly requested.
            "sl_ma": [100.1 + i for i in range(10)],
        },
        index=idx,
    )
    df.attrs["report_overlay_cols"] = ["ema_close_20"]

    trades = [
        Trade(
            entry_time=idx[2],
            exit_time=idx[5],
            direction="long",
            entry_price=float(df.loc[idx[2], "close"]),
            exit_price=float(df.loc[idx[5], "close"]),
            quantity=1.0,
            pnl_after_costs=1.0,
            stop_price=95.0,
            exit_reason="signal_exit",
        )
    ]

    result = BacktestResult(
        initial_capital=10000.0,
        final_capital=10001.0,
        total_trades=len(trades),
        winning_trades=1,
        losing_trades=0,
        total_pnl=1.0,
        trades=trades,
        entry_tf="1h",
        price_df=df,
    )

    rg = ReportGenerator(output_dir=tmp_path)
    html_path = rg.generate_backtest_report(result, include_visuals=True, output_name="x2")
    html = html_path.read_text(encoding="utf-8")

    assert "ema_close_20" in html
    assert "sl_ma" not in html
