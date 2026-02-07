from __future__ import annotations

import json
import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _try_git_sha(repo_dir: Path) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout.strip() or None
    except Exception:
        return None


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _to_json(v: Any) -> str:
    return json.dumps(v, sort_keys=True, default=str)


@dataclass(frozen=True)
class RunRecord:
    strategy: str
    phase: str
    data_file: str
    dataset_manifest_hash: Optional[str]
    config_hash: Optional[str]
    passed: Optional[bool]
    report_path: Optional[str]
    metrics: Optional[dict[str, Any]] = None
    params: Optional[dict[str, Any]] = None
    outcome: Optional[dict[str, Any]] = None
    command: Optional[str] = None
    git_sha: Optional[str] = None


class ExperimentRegistry:
    def __init__(self, db_path: Path | str = ".experiments/registry.sqlite"):
        self.db_path = Path(db_path)
        _ensure_parent(self.db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    data_file TEXT NOT NULL,
                    dataset_manifest_hash TEXT,
                    config_hash TEXT,
                    git_sha TEXT,
                    passed INTEGER,
                    report_path TEXT,
                    metrics_json TEXT,
                    params_json TEXT,
                    outcome_json TEXT,
                    command TEXT
                )
                """
            )

            # Lightweight migrations for existing registries.
            # Keep this idempotent and backwards compatible.
            existing_cols = {
                row["name"] for row in conn.execute("PRAGMA table_info(runs)").fetchall()
            }
            if "outcome_json" not in existing_cols:
                conn.execute("ALTER TABLE runs ADD COLUMN outcome_json TEXT")

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_runs_strategy_phase ON runs(strategy, phase)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at)")

    def record_run(self, record: RunRecord, repo_dir: Optional[Path] = None) -> int:
        git_sha = record.git_sha
        if git_sha is None and repo_dir is not None:
            git_sha = _try_git_sha(repo_dir)

        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO runs (
                    created_at, strategy, phase, data_file,
                    dataset_manifest_hash, config_hash, git_sha,
                    passed, report_path, metrics_json, params_json, outcome_json, command
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _utc_now_iso(),
                    record.strategy,
                    record.phase,
                    record.data_file,
                    record.dataset_manifest_hash,
                    record.config_hash,
                    git_sha,
                    None if record.passed is None else (1 if record.passed else 0),
                    record.report_path,
                    None if record.metrics is None else _to_json(record.metrics),
                    None if record.params is None else _to_json(record.params),
                    None if record.outcome is None else _to_json(record.outcome),
                    record.command,
                ),
            )
            return int(cur.lastrowid)

    def list_runs(self, strategy: Optional[str] = None, phase: Optional[str] = None, limit: int = 50):
        q = "SELECT * FROM runs"
        clauses = []
        args: list[Any] = []
        if strategy:
            clauses.append("strategy = ?")
            args.append(strategy)
        if phase:
            clauses.append("phase = ?")
            args.append(phase)
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " ORDER BY id DESC LIMIT ?"
        args.append(limit)

        with self._connect() as conn:
            return [dict(r) for r in conn.execute(q, args).fetchall()]

    def list_latest_runs(
        self,
        strategy: Optional[str] = None,
        phase: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Return the latest run per (strategy, phase).

        Useful for building a passed/failed strategy catalog.
        """
        q = """
            SELECT r.*
            FROM runs r
            JOIN (
                SELECT strategy, phase, MAX(id) AS max_id
                FROM runs
                GROUP BY strategy, phase
            ) m
            ON r.id = m.max_id
        """
        clauses = []
        args: list[Any] = []
        if strategy:
            clauses.append("r.strategy = ?")
            args.append(strategy)
        if phase:
            clauses.append("r.phase = ?")
            args.append(phase)
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " ORDER BY r.strategy ASC, r.phase ASC"

        with self._connect() as conn:
            return [dict(r) for r in conn.execute(q, args).fetchall()]

    def get_run(self, run_id: int) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            return dict(row) if row else None

    def compare_runs(self, run_a: int, run_b: int) -> dict[str, Any]:
        a = self.get_run(run_a)
        b = self.get_run(run_b)
        if a is None or b is None:
            raise ValueError("Run id not found")

        def parse_metrics(r):
            if not r.get("metrics_json"):
                return {}
            try:
                return json.loads(r["metrics_json"])
            except Exception:
                return {}

        ma = parse_metrics(a)
        mb = parse_metrics(b)
        keys = sorted(set(ma.keys()) | set(mb.keys()))
        diffs = {}
        for k in keys:
            va = ma.get(k)
            vb = mb.get(k)
            if va != vb:
                diffs[k] = {"a": va, "b": vb}

        return {
            "a": a,
            "b": b,
            "metric_diffs": diffs,
        }
