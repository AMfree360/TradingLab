from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class RunBundle:
    dir: Path

    @property
    def stdout_path(self) -> Path:
        return self.dir / "stdout.log"

    @property
    def stderr_path(self) -> Path:
        return self.dir / "stderr.log"

    @property
    def inputs_path(self) -> Path:
        return self.dir / "inputs.json"

    @property
    def meta_path(self) -> Path:
        return self.dir / "meta.json"

    @property
    def command_path(self) -> Path:
        return self.dir / "command.txt"


def create_run_bundle(*, repo_root: Path, workflow: str, strategy_name: str) -> RunBundle:
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    safe_name = "".join(ch for ch in strategy_name if ch.isalnum() or ch in {"-", "_"}).strip() or "strategy"
    safe_flow = "".join(ch for ch in workflow if ch.isalnum() or ch in {"-", "_"}).strip() or "workflow"

    base = repo_root / "data" / "logs" / "gui_runs"
    base.mkdir(parents=True, exist_ok=True)

    run_dir = base / f"{ts}_{safe_flow}_{safe_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    return RunBundle(dir=run_dir)


def write_bundle_inputs(bundle: RunBundle, inputs: dict[str, Any]) -> None:
    bundle.inputs_path.write_text(json.dumps(inputs, indent=2, sort_keys=True))


def write_bundle_meta(bundle: RunBundle, *, python_executable: str, repo_root: Path) -> None:
    meta: dict[str, Any] = {
        "python": python_executable,
        "cwd": str(repo_root),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "env": {
            "USER": os.environ.get("USER"),
        },
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    bundle.meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))


def write_bundle_command(bundle: RunBundle, argv: list[str]) -> None:
    # Store a shell-ish representation for copy/paste.
    def _q(s: str) -> str:
        if not s:
            return "''"
        if any(ch.isspace() for ch in s) or any(ch in s for ch in ['"', "'", "\\"]):
            return "'" + s.replace("'", "'\\''") + "'"
        return s

    bundle.command_path.write_text(" ".join(_q(a) for a in argv) + "\n")


def write_bundle_logs(bundle: RunBundle, *, stdout: str, stderr: str) -> None:
    bundle.stdout_path.write_text(stdout or "")
    bundle.stderr_path.write_text(stderr or "")
