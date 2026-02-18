from pathlib import Path
from fastapi.testclient import TestClient
import pytest
pytest.skip("Builder V3 removed; tests disabled", allow_module_level=True)

from gui_launcher.app import app
import json


def test_csrf_token_not_configured():
    client = TestClient(app)
    # ensure no secret is set
    old_secret = getattr(app.state, "builder_v3_csrf_secret", None)
    try:
        app.state.builder_v3_csrf_secret = None
        r = client.get("/api/builder_v3/csrf_token")
        assert r.status_code == 200
        body = r.json()
        assert body.get("ok") is False
    finally:
        app.state.builder_v3_csrf_secret = old_secret


def test_csrf_token_issue_and_use_header(tmp_path):
    client = TestClient(app)
    # configure secret and enable enforcement
    old_secret = getattr(app.state, "builder_v3_csrf_secret", None)
    old_enforce = getattr(app.state, "builder_v3_csrf_enforce", False)
    try:
        app.state.builder_v3_csrf_secret = "supersecret"
        app.state.builder_v3_csrf_enforce = True

        # ensure research_specs exists
        repo_root = Path(__file__).resolve().parents[1]
        rs_dir = repo_root / "research_specs"
        rs_dir.mkdir(parents=True, exist_ok=True)

        # request a token
        r = client.get("/api/builder_v3/csrf_token")
        assert r.status_code == 200
        body = r.json()
        assert body.get("ok") is True
        token = body.get("token")
        assert isinstance(token, str) and token.count(".") == 1

        # attempt save without token should be forbidden
        payload = {"symbol": "BTCUSD", "entry_tf": "1h", "long_enabled": True, "short_enabled": False}
        r2 = client.post("/api/builder_v3/save", json=payload)
        assert r2.status_code == 403

        # now send with header
        r3 = client.post("/api/builder_v3/save", json=payload, headers={"X-CSRF-Token": token})
        assert r3.status_code == 200
        data = r3.json().get("data")
        assert data and ("draft_id" in data or "spec_name" in data or "spec_path" in data)
    finally:
        app.state.builder_v3_csrf_secret = old_secret
        app.state.builder_v3_csrf_enforce = old_enforce


def test_csrf_token_use_in_body(tmp_path):
    client = TestClient(app)
    old_secret = getattr(app.state, "builder_v3_csrf_secret", None)
    old_enforce = getattr(app.state, "builder_v3_csrf_enforce", False)
    try:
        app.state.builder_v3_csrf_secret = "anothersecret"
        app.state.builder_v3_csrf_enforce = True

        r = client.get("/api/builder_v3/csrf_token")
        assert r.status_code == 200
        token = r.json().get("token")

        payload = {"symbol": "ETHUSD", "entry_tf": "1h", "long_enabled": True, "short_enabled": False, "csrf_token": token}
        r2 = client.post("/api/builder_v3/save", json=payload)
        assert r2.status_code == 200
        data = r2.json().get("data")
        assert data is not None
    finally:
        app.state.builder_v3_csrf_secret = old_secret
        app.state.builder_v3_csrf_enforce = old_enforce
