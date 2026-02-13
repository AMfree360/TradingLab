from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import yaml

from config.schema import validate_strategy_config
from research.compiler import StrategyCompiler


def _load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def test_research_compiler_emits_trade_management_and_stop_logic(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    compiler = StrategyCompiler(repo_root=repo_root)

    spec_dict = {
        "name": "research_tm_test",
        "description": "test",
        "market": {"symbol": "BTCUSDT", "exchange": "binance", "market_type": "spot"},
        "entry_tf": "4h",
        "extra_tfs": [],
        "long": {
            "enabled": True,
            "conditions_all": [{"tf": "4h", "expr": "close > sma(close, 2)"}],
            "stop_type": "percent",
            "stop_percent": 2.0,  # 2% (supports >1 meaning percent)
        },
        "short": {
            "enabled": True,
            "conditions_all": [{"tf": "4h", "expr": "close < sma(close, 2)"}],
            "stop_type": "atr",
            "atr_length": 14,
            "atr_multiplier": 3.0,
        },
        "filters": {"calendar_filters": {"master_filters_enabled": False}},
        "risk_per_trade_pct": 1.0,
        "trade_management": {
            "trailing_stop": {"enabled": True, "type": "EMA", "length": 21, "activation_r": 0.5},
            "take_profit": {"enabled": True, "levels": [{"target_r": 2.0, "exit_pct": 50.0}]},
            "partial_exit": {"enabled": True, "levels": [{"level_r": 1.5, "exit_pct": 80.0}]},
        },
    }

    spec_path = tmp_path / "spec.yml"
    spec_path.write_text(yaml.safe_dump(spec_dict, sort_keys=False))
    spec = compiler.load_spec(spec_path)

    out_dir = compiler.compile_to_folder(spec, out_dir=tmp_path / "out")

    # Config should validate and include TM overrides.
    config_path = out_dir / "config.yml"
    config = yaml.safe_load(config_path.read_text())
    validate_strategy_config(config)

    assert config["trailing_stop"]["enabled"] is True
    assert config["trailing_stop"]["type"] == "EMA"
    assert config["take_profit"]["enabled"] is True
    assert config["take_profit"]["levels"][0]["target_r"] == 2.0
    assert config["partial_exit"]["enabled"] is True
    assert config["partial_exit"]["levels"][0]["level_r"] == 1.5

    # Generated strategy should compute stops from the spec.
    mod = _load_module_from_path("research_tm_test_strategy", out_dir / "strategy.py")
    StrategyCls = getattr(mod, "GeneratedResearchStrategy")
    strategy = StrategyCls(validate_strategy_config(config))

    entry_row_long = pd.Series({"close": 100.0})
    stop_long = strategy._calculate_stop_loss(entry_row_long, dir_int=1)
    assert abs(stop_long - 98.0) < 1e-9

    entry_row_short = pd.Series({"close": 100.0, "atr_14": 2.0})
    stop_short = strategy._calculate_stop_loss(entry_row_short, dir_int=-1)
    assert abs(stop_short - 106.0) < 1e-9
