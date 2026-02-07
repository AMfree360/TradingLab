from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_file_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    file_path = Path(file_path)
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def infer_bar_seconds(index: pd.DatetimeIndex) -> Optional[int]:
    if len(index) < 3:
        return None

    # Use sorted unique timestamps to avoid duplicates biasing inference
    idx = pd.DatetimeIndex(index).sort_values().unique()
    if len(idx) < 3:
        return None

    deltas = (idx[1:] - idx[:-1]).to_series().dt.total_seconds()
    deltas = deltas[deltas > 0]
    if deltas.empty:
        return None

    # Robust inference: pick the mode if possible, else median
    try:
        mode = deltas.round().mode()
        if not mode.empty:
            sec = int(mode.iloc[0])
            return sec if sec > 0 else None
    except Exception:
        pass

    sec = int(deltas.median().round())
    return sec if sec > 0 else None


def estimate_missing_bars(index: pd.DatetimeIndex, expected_bar_seconds: int) -> int:
    if expected_bar_seconds <= 0:
        return 0
    idx = pd.DatetimeIndex(index).sort_values().unique()
    if len(idx) < 2:
        return 0

    deltas = (idx[1:] - idx[:-1]).to_series().dt.total_seconds()
    missing = 0
    for d in deltas:
        if d <= expected_bar_seconds * 1.5:
            continue
        gap = int(round(d / expected_bar_seconds)) - 1
        if gap > 0:
            missing += gap
    return int(missing)


def canonical_json_dumps(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DatasetManifest:
    """A deterministic description of a dataset slice used in a phase."""

    schema_version: int
    created_at: str
    purpose: str

    file_path: str
    file_resolved_path: str
    file_size_bytes: int
    file_mtime_utc: str
    file_sha256: str

    slice_start: Optional[str]
    slice_end: Optional[str]

    rows: int
    first_timestamp: Optional[str]
    last_timestamp: Optional[str]
    inferred_bar_seconds: Optional[int]
    duplicate_timestamps: int
    estimated_missing_bars: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "purpose": self.purpose,
            "file": {
                "path": self.file_path,
                "resolved_path": self.file_resolved_path,
                "size_bytes": self.file_size_bytes,
                "mtime_utc": self.file_mtime_utc,
                "sha256": self.file_sha256,
            },
            "slice": {
                "start": self.slice_start,
                "end": self.slice_end,
            },
            "stats": {
                "rows": self.rows,
                "first_timestamp": self.first_timestamp,
                "last_timestamp": self.last_timestamp,
                "inferred_bar_seconds": self.inferred_bar_seconds,
                "duplicate_timestamps": self.duplicate_timestamps,
                "estimated_missing_bars": self.estimated_missing_bars,
            },
        }

    def identity_dict(self) -> dict[str, Any]:
        """Deterministic dataset identity for locking.

        Excludes run-varying fields like created_at and path resolution/mtime.
        """
        return {
            "schema_version": self.schema_version,
            "purpose": self.purpose,
            "file": {
                "path": self.file_path,
                "sha256": self.file_sha256,
            },
            "slice": {
                "start": self.slice_start,
                "end": self.slice_end,
            },
            "stats": {
                "rows": self.rows,
                "first_timestamp": self.first_timestamp,
                "last_timestamp": self.last_timestamp,
                "inferred_bar_seconds": self.inferred_bar_seconds,
                "duplicate_timestamps": self.duplicate_timestamps,
                "estimated_missing_bars": self.estimated_missing_bars,
            },
        }

    def manifest_hash(self) -> str:
        return sha256_text(canonical_json_dumps(self.identity_dict()))


def build_manifest(
    *,
    file_path: Path,
    df: pd.DataFrame,
    purpose: str,
    slice_start: Optional[str] = None,
    slice_end: Optional[str] = None,
) -> DatasetManifest:
    file_path = Path(file_path)
    resolved = str(file_path.resolve())

    stat = file_path.stat()
    mtime_utc = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    file_sha = compute_file_sha256(file_path)

    idx = pd.DatetimeIndex(df.index)
    rows = int(len(df))
    first_ts = str(idx.min()) if rows > 0 else None
    last_ts = str(idx.max()) if rows > 0 else None

    inferred = infer_bar_seconds(idx)
    duplicates = int(pd.Index(idx).duplicated().sum())
    missing = estimate_missing_bars(idx, inferred) if inferred else 0

    return DatasetManifest(
        schema_version=1,
        created_at=_utc_now_iso(),
        purpose=purpose,
        file_path=str(file_path),
        file_resolved_path=resolved,
        file_size_bytes=int(stat.st_size),
        file_mtime_utc=mtime_utc,
        file_sha256=file_sha,
        slice_start=slice_start,
        slice_end=slice_end,
        rows=rows,
        first_timestamp=first_ts,
        last_timestamp=last_ts,
        inferred_bar_seconds=inferred,
        duplicate_timestamps=duplicates,
        estimated_missing_bars=missing,
    )


def write_manifest_json(manifest: DatasetManifest, output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(manifest.file_path).stem
    short = manifest.manifest_hash()[:12]
    out_path = output_dir / f"{stem}.{manifest.purpose}.{short}.json"
    out_path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True, default=str))
    return out_path
