import pandas as pd
from pathlib import Path
from fastapi.testclient import TestClient

import pytest
pytest.skip("Builder V3 removed; tests disabled", allow_module_level=True)

import pandas as pd
from pathlib import Path
from fastapi.testclient import TestClient

import gui_launcher.app as appmod


def test_context_visual_returns_fig(monkeypatch):
    # Build a minimal OHLCV DataFrame with a DatetimeIndex
    idx = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='H')
    df = pd.DataFrame({
        'open': 1.0,
        'high': 2.0,
        'low': 0.5,
        'close': 1.5,
        'volume': 100.0,
    }, index=idx)

    # Patch DataLoader.load to return our DataFrame regardless of path
    def _fake_load(self, path):
        return df

    monkeypatch.setattr('adapters.data.data_loader.DataLoader.load', _fake_load)

    # Patch _find_latest_gui_dataset to return a dummy path so the v2 handler proceeds
    monkeypatch.setattr('gui_launcher.app._find_latest_gui_dataset', lambda repo_root: Path('dummy_dataset'))

    client = TestClient(appmod.app)
    resp = client.post('/api/builder_v3/context_visual', json={'entry_tf': '1h', 'primary_context_type': 'ma_stack', 'ma_fast': 10, 'ma_mid': 20, 'ma_slow': 50})
    assert resp.status_code == 200, resp.text
    j = resp.json()
    # Accept either top-level 'fig' (v2 shape) or nested under data.fig (v3 envelope)
    fig = j.get('fig') or (j.get('data') and j['data'].get('fig'))
    assert fig is not None, f"Expected fig in response, got: {j}"
