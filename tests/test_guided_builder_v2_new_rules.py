from fastapi.testclient import TestClient

import gui_launcher.app as appmod

client = TestClient(appmod.app)


def test_context_atr_spike_and_bollinger_context():
    payload = {
        "entry_tf": "1h",
        "long_enabled": True,
        "short_enabled": False,
        "context_rules": [{"type": "atr_spike", "atr_len": 14, "mult": 1.5}, {"type": "bollinger_context", "length": 20, "mult": 2.0}],
    }
    resp = client.post("/api/guided/builder_v2/preview", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "long" in data
    ctx = data["long"]["context"]
    labels = [c.get("label", "") for c in ctx]
    assert any("ATR spike" in l for l in labels), labels
    assert any("Bollinger" in l for l in labels), labels


def test_signals_zscore_bollinger_atr_delta_volume():
    payload = {
        "entry_tf": "1h",
        "long_enabled": True,
        "short_enabled": False,
        "signal_rules": [
            {"type": "z_score", "length": 20, "op": ">", "threshold": 2.0},
            {"type": "bollinger_outlier", "length": 20, "mult": 2.0},
            {"type": "atr_deviation", "length": 20, "atr_len": 14, "threshold": 1.0},
            {"type": "delta_divergence", "threshold": 0.0},
            {"type": "volume_rejection", "vol_len": 20, "mult": 3.0},
        ],
    }
    resp = client.post("/api/guided/builder_v2/preview", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "long" in data
    sigs = data["long"]["signals"]
    labels = [s.get("label", "") for s in sigs]
    # Check expected labels present
    assert any("Z-score" in l for l in labels), labels
    assert any("Bollinger outlier" in l for l in labels), labels
    assert any("ATR deviation" in l for l in labels), labels
    assert any("Delta divergence" in l for l in labels) or any("delta" in (s.get("expr") or "") for s in sigs)
    assert any("Volume" in l for l in labels), labels


def test_trigger_volume_and_delta():
    payload = {
        "entry_tf": "1h",
        "long_enabled": True,
        "short_enabled": False,
        "trigger_rules": [
            {"type": "volume_rejection", "vol_len": 20, "mult": 3.0},
            {"type": "delta_divergence", "threshold": 0.0},
        ],
    }
    resp = client.post("/api/guided/builder_v2/preview", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    trg = data["long"]["triggers"]
    labels = [t.get("label", "") for t in trg]
    assert any("Volume rejection" in l for l in labels), labels
    assert any("Delta divergence" in l for l in labels), labels
