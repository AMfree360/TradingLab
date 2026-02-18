import os
import json
from pathlib import Path
from fastapi.testclient import TestClient

import pytest
pytest.skip("Builder V3 removed; tests disabled", allow_module_level=True)

from gui_launcher.app import app


def _write_spec_file(repo_root: Path, name: str, content: str = "old") -> Path:
    out_dir = repo_root / "research_specs"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.yml"
    p.write_text(content, encoding="utf-8")
    return p


def test_save_conflict_without_force(tmp_path, monkeypatch):
    client = TestClient(app)
    repo_root = Path(__file__).resolve().parents[1]
    # Ensure target exists
    target = _write_spec_file(repo_root, "mystrategy", "existing: true")

    payload = {"name": "mystrategy", "symbol": "BTCUSD", "entry_tf": "1h", "long_enabled": True, "short_enabled": False}
    r = client.post("/api/builder_v3/save", json=payload)
    # Legacy behavior: name conflict falls back to creating a draft JSON and returns 200
    assert r.status_code == 200
    body = r.json()
    assert body.get("ok") is True
    data = body.get("data") or {}
    # Should have created a draft_id and not a canonical spec_path
    assert data.get("draft_id")
    assert data.get("spec_path") is None or data.get("spec_path") == ""


def test_save_with_force_and_flag_allows_overwrite(tmp_path):
    client = TestClient(app)
    repo_root = Path(__file__).resolve().parents[1]
    # Ensure target exists
    target = _write_spec_file(repo_root, "force_me", "existing: 1")

    # Enable overwrite by flag
    app.state.builder_v3_allow_overwrite = True

    payload = {"name": "force_me", "symbol": "BTCUSD", "entry_tf": "1h", "long_enabled": True, "short_enabled": False, "force": True}
    r = client.post("/api/builder_v3/save", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body.get("ok") is True
    data = body.get("data") or {}
    spec_path = data.get("spec_path")
    assert spec_path is not None
    # File should exist and be updated (not the exact content check, but path present)
    assert Path(spec_path).exists()
