import os
import re
import json
from pathlib import Path

from fastapi.testclient import TestClient

from gui_launcher.app import app


def _find_new_draft_file(repo_root: Path, before: set[str]) -> str | None:
    rs = repo_root / "research_specs"
    if not rs.exists():
        return None
    for p in rs.iterdir():
        if p.name in before:
            continue
        if re.match(r"^draft-.*\.json$", p.name):
            return str(p)
    return None


def test_save_creates_draft_and_returns_draft_id(tmp_path):
    client = TestClient(app)
    repo_root = Path(__file__).resolve().parents[1]
    rs_dir = repo_root / "research_specs"
    rs_dir.mkdir(parents=True, exist_ok=True)

    before = set([p.name for p in rs_dir.iterdir()])

    payload = {"symbol": "BTCUSD", "entry_tf": "1h", "long_enabled": True, "short_enabled": False}
    r = client.post("/api/builder_v3/save", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body.get("ok") is True
    data = body.get("data") or {}
    draft_id = data.get("draft_id")
    assert isinstance(draft_id, str) and draft_id.startswith("draft-")

    # find created file
    new_file = _find_new_draft_file(repo_root, before)
    assert new_file is not None

    # cleanup created file
    try:
        Path(new_file).unlink()
    except Exception:
        pass


def test_save_invalid_payload_returns_422():
    client = TestClient(app)
    # missing required fields: symbol
    payload = {"entry_tf": "1h", "long_enabled": True, "short_enabled": False}
    r = client.post("/api/builder_v3/save", json=payload)
    assert r.status_code == 422
    body = r.json()
    assert body.get("ok") is False
    assert "errors" in body
