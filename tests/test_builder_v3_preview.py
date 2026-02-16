from fastapi.testclient import TestClient

import gui_launcher.app as appmod

client = TestClient(appmod.app)


def test_builder_v3_preview_basic():
    payload = {
        "entry_tf": "1h",
        "context_tf": "",
        "signal_tf": "",
        "trigger_tf": "",
        "long_enabled": True,
        "short_enabled": True,
    }
    resp = client.post("/api/builder_v3/preview", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    # Expect the preview structure produced by _builder_v2_preview
    assert isinstance(data, dict)
    assert "entry_tf" in data
    assert "long" in data and "short" in data
    assert isinstance(data["long"], dict)
    assert isinstance(data["short"], dict)


def test_builder_v3_preview_invalid_json():
    # Send malformed JSON body and expect a 400 response
    from fastapi.testclient import TestClient
    client = TestClient(appmod.app)
    resp = client.post("/api/builder_v3/preview", data="{bad json", headers={"content-type": "application/json"})
    assert resp.status_code == 400


def test_builder_v3_preview_numeric_coercion():
    # Send numeric fields as strings and ensure preview still succeeds
    payload = {
        "entry_tf": "1h",
        "ma_fast": "30",
        "ma_mid": "60",
        "ma_slow": "200",
        "long_enabled": True,
        "short_enabled": False,
    }
    resp = client.post("/api/builder_v3/preview", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "long" in data and isinstance(data["long"], dict)
