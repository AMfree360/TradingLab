from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AIBundlePaths:
    root: Path
    input_notes: Path
    prompt: Path
    response_raw: Path
    output: Path
    meta: Path


def create_ai_bundle_dir(*, base_dir: Path, prefix: str = "draft") -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    # Small random-ish suffix based on time to avoid collisions in fast runs.
    suffix = hashlib.sha256(f"{ts}".encode("utf-8")).hexdigest()[:8]
    out = base_dir / f"{ts}_{prefix}_{suffix}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def bundle_paths(root: Path) -> AIBundlePaths:
    return AIBundlePaths(
        root=root,
        input_notes=root / "input_notes.txt",
        prompt=root / "prompt.json",
        response_raw=root / "response_raw.json",
        output=root / "output.txt",
        meta=root / "meta.json",
    )


def write_ai_bundle(
    *,
    root: Path,
    input_notes: str,
    prompt: dict[str, Any],
    response_raw: dict[str, Any] | None,
    output_text: str,
    meta: dict[str, Any],
) -> AIBundlePaths:
    paths = bundle_paths(root)

    paths.input_notes.write_text(input_notes)
    paths.output.write_text(output_text)

    # Make meta rich and stable.
    stable_meta = {
        **meta,
        "input_notes_sha256": _sha256_text(input_notes),
        "output_sha256": _sha256_text(output_text),
        "prompt_sha256": _sha256_text(json.dumps(prompt, sort_keys=True, ensure_ascii=False)),
    }

    paths.prompt.write_text(json.dumps(prompt, indent=2, sort_keys=True, ensure_ascii=False))
    paths.response_raw.write_text(
        json.dumps(response_raw or {}, indent=2, sort_keys=True, ensure_ascii=False)
    )
    paths.meta.write_text(json.dumps(stable_meta, indent=2, sort_keys=True, ensure_ascii=False))

    return paths
