from __future__ import annotations

import pandas as pd
import pytest

from validation.pipeline import ValidationPipeline
from validation.state import ValidationStateManager
from strategies.base import StrategyBase
from config.schema import WalkForwardConfig


class NoopStrategy(StrategyBase):
    def generate_signals(self, df_by_tf):
        return pd.DataFrame()

    def get_indicators(self, df, tf=None):
        return df


def _minimal_strategy_config() -> dict:
    return {
        'strategy_name': 'noop',
        'market': {'exchange': 'test', 'symbol': 'TEST', 'base_timeframe': '1h'},
        'timeframes': {'signal_tf': '1h', 'entry_tf': '1h'},
        'risk': {'sizing_mode': 'account_size', 'risk_per_trade_pct': 1.0, 'account_size': 10000.0},
        'stop_loss': {'type': 'SMA', 'length': 10, 'buffer_pips': 1.0, 'buffer_unit': 'price'},
        'take_profit': {'enabled': False, 'target_r': 2.0},
        'trade_direction': {'allow_long': True, 'allow_short': False},
        'moving_averages': {},
        'alignment_rules': {},
    }


def test_phase2_marks_consumed_before_validation_runs(tmp_path, monkeypatch):
    # Prepare state file with Phase 1 passed
    state_dir = tmp_path / "state"
    mgr = ValidationStateManager(state_dir)

    data_file = tmp_path / "data.csv"
    df = pd.DataFrame(
        {
            'open': [1, 1, 1, 1],
            'high': [1, 1, 1, 1],
            'low': [1, 1, 1, 1],
            'close': [1, 1, 1, 1],
            'volume': [1, 1, 1, 1],
        },
        index=pd.date_range('2020-01-01', periods=4, freq='1H'),
    )
    df.to_csv(data_file)

    state = mgr.create_or_load_state(
        strategy_name="noop",
        data_file=str(data_file),
        training_start="2020-01-01",
        training_end="2020-01-01",
    )
    state.phase1_training_completed = True
    state.phase1_passed = True
    state.phase1_results = {'failure_reasons': []}
    mgr.save_state(state)

    # Pipeline with state_dir
    pipeline = ValidationPipeline(strategy_class=NoopStrategy, state_dir=state_dir)

    # Force validator to raise, to prove consumption happens before validation completes
    def boom(*args, **kwargs):
        raise RuntimeError("validator exploded")

    monkeypatch.setattr(pipeline.oos_validator, "validate", boom)

    with pytest.raises(RuntimeError, match="validator exploded"):
        pipeline.run_phase2_oos(
            strategy_name="noop",
            data_file=str(data_file),
            oos_data=df,
            strategy_config=_minimal_strategy_config(),
            wf_config=WalkForwardConfig(
                start_date=str(df.index[0]),
                end_date=str(df.index[-1]),
                window_type="expanding",
                training_period={"duration": "1 day"},
                test_period={"duration": "1 day"},
                min_training_period="1 day",
            ),
            oos_start="2020-01-01",
            oos_end="2020-01-01",
            full_data=df,
        )

    # Reload state and confirm consumed
    state2 = mgr.load_state("noop", str(data_file))
    assert state2 is not None
    assert state2.phase2_oos_consumed is True
    assert state2.can_use_oos_data() is False
