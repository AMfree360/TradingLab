import os
import json
from pathlib import Path
from fastapi.testclient import TestClient

from gui_launcher.app import app


def test_save_enforces_csrf(tmp_path):
    client = TestClient(app)
    # enable CSRF token in app state
    app.state.builder_v3_csrf = "token-123"

    repo_root = Path(__file__).resolve().parents[1]
    rs_dir = repo_root / "research_specs"
    rs_dir.mkdir(parents=True, exist_ok=True)
    before = set([p.name for p in rs_dir.iterdir()])

    payload = {"symbol": "BTCUSD", "entry_tf": "1h", "long_enabled": True, "short_enabled": False}
    # Missing token -> should be rejected
    r = client.post("/api/builder_v3/save", json=payload)
    assert r.status_code == 403

    # With header token -> accepted
    headers = {"X-CSRF-Token": "token-123"}
    r2 = client.post("/api/builder_v3/save", json=payload, headers=headers)
    assert r2.status_code == 200
    body = r2.json()
    assert body.get("ok") is True

    # created draft JSON should exist
    after = set([p.name for p in rs_dir.iterdir()])
    new = list(after - before)
    assert any(n.startswith("draft-") and n.endswith(".json") for n in new)


def test_save_name_conflict_falls_back(tmp_path):
    client = TestClient(app)
    # ensure CSRF not required for this test
    app.state.builder_v3_csrf = None

    repo_root = Path(__file__).resolve().parents[1]
    rs_dir = repo_root / "research_specs"
    rs_dir.mkdir(parents=True, exist_ok=True)

    # create an existing spec file with safe name 'conflict'
    p = rs_dir / "conflict.yml"
    p.write_text("name: conflict\n")

    before = set([pp.name for pp in rs_dir.iterdir()])

    payload = {"name": "conflict", "symbol": "BTCUSD", "entry_tf": "1h", "long_enabled": True, "short_enabled": False}
    r = client.post("/api/builder_v3/save", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body.get("ok") is True
    data = body.get("data") or {}
    draft_id = data.get("draft_id")
    assert isinstance(draft_id, str) and draft_id.startswith("draft-")

    # ensure at least one new draft json exists
    after = set([pp.name for pp in rs_dir.iterdir()])
    new = after - before
    assert any(n.endswith('.json') for n in new)
