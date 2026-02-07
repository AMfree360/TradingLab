import pytest
import pandas as pd

from validation.state import ValidationState


def test_phase_lock_mismatch_raises():
    state = ValidationState(strategy_name="s", data_file="d")
    lock_a = {"manifest_hash": "abc", "manifest": {"x": 1}}
    lock_b = {"manifest_hash": "def", "manifest": {"x": 2}}

    state.verify_or_set_phase_lock("phase1", lock_a)
    with pytest.raises(ValueError, match="Phase 1"):
        state.verify_or_set_phase_lock("phase1", lock_b)


def test_phase2_consumed_field_migrates_from_completed():
    data = {
        "strategy_name": "s",
        "data_file": "d",
        "phase2_oos_completed": True,
        "phase2_completed_at": "2020-01-01T00:00:00",
    }
    state = ValidationState.from_dict(data)
    assert state.phase2_oos_consumed is True
    assert state.phase2_oos_consumed_at == "2020-01-01T00:00:00"


def test_manifest_helpers_infer_bar_seconds_and_missing_estimates():
    # Basic smoke to ensure manifest helpers behave on regular data
    from repro.dataset_manifest import infer_bar_seconds, estimate_missing_bars

    idx = pd.date_range("2020-01-01", periods=10, freq="1H")
    assert infer_bar_seconds(idx) == 3600

    # Remove 2 hours (one gap of 3h implies 2 missing bars)
    idx2 = idx.delete([5, 6])
    assert estimate_missing_bars(idx2, 3600) == 2
