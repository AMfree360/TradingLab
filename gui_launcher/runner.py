from __future__ import annotations

import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Job:
    id: str
    argv: list[str]
    cwd: Path
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""

    # Optional extracted outputs
    report_path: Optional[str] = None
    spec_path: Optional[str] = None
    built_strategy_path: Optional[str] = None


class JobRunner:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}

    def create(self, *, argv: list[str], cwd: Path) -> Job:
        job = Job(id=str(uuid.uuid4()), argv=argv, cwd=cwd, created_at=time.time())
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_recent(self, limit: int = 50) -> list[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def start(self, job_id: str) -> None:
        job = self.get(job_id)
        if not job:
            raise KeyError(job_id)
        t = threading.Thread(target=self._run_job, args=(job_id,), daemon=True)
        t.start()

    def _run_job(self, job_id: str) -> None:
        job = self.get(job_id)
        if not job:
            return

        job.started_at = time.time()
        try:
            proc = subprocess.run(
                job.argv,
                cwd=str(job.cwd),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            job.exit_code = int(proc.returncode)
            job.stdout = proc.stdout or ""
            job.stderr = proc.stderr or ""
        except Exception as e:
            job.exit_code = 1
            job.stderr = f"Job runner error: {type(e).__name__}: {e}\n"
        finally:
            job.finished_at = time.time()
