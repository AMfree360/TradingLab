from fastapi.testclient import TestClient

import gui_launcher.app as appmod

client = TestClient(appmod.app)


def test_builder_v3_validate_valid_payload():
    payload = {
        "symbol": "AAPL",
        "entry_tf": "1h",
        "long_enabled": True,
        "short_enabled": False,
    }
    resp = client.post("/api/builder_v3/validate", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data.get("ok") is True
    assert data.get("valid") is True


def test_builder_v3_validate_missing_required():
    # missing required top-level fields like symbol
    payload = {
        "entry_tf": "1h",
        "long_enabled": True,
        "short_enabled": False,
    }
    resp = client.post("/api/builder_v3/validate", json=payload)
    assert resp.status_code == 422
    data = resp.json()
    assert data.get("ok") is False
    assert data.get("valid") is False
    assert isinstance(data.get("errors"), list)
    # Expect an error about missing 'symbol'
    assert any("symbol" in e.get("path", "") or "symbol" in e.get("message", "") for e in data.get("errors", []))
