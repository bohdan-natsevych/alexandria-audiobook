"""Filesystem-backed runner for isolated, disposable compute jobs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import uuid
import queue
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional


class JobContext:
    def __init__(self, manager: "JobManager", job_id: str):
        self.manager = manager
        self.job_id = job_id
        self.workspace = manager.root / job_id
        self.input_dir = self.workspace / "input"
        self.output_dir = self.workspace / "output"
        self.work_dir = self.workspace / "work"

    def log(self, message: str) -> None:
        self.manager._append_log(self.job_id, message)

    def cancelled(self) -> bool:
        return bool(self.manager.get(self.job_id).get("cancel_requested"))

    def set_progress(self, completed: int, total: int, message: Optional[str] = None) -> None:
        changes = {"progress": {"completed": completed, "total": total}}
        if message:
            changes["message"] = message
        self.manager._update(self.job_id, **changes)

    def set_result(self, path: Path, media_type: str, filename: Optional[str] = None) -> None:
        self.manager._update(
            self.job_id,
            result={
                "path": str(path.relative_to(self.workspace)),
                "media_type": media_type,
                "filename": filename or path.name,
            },
        )

    def set_artifacts(self, artifacts: list[dict]) -> None:
        """Publish small result files without exposing workspace paths."""
        self.manager._update(self.job_id, artifacts=artifacts)

    def run_subprocess(self, command: list[str], cwd: Path) -> int:
        """Stream a child process into job logs and terminate it on cancel."""
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        output: queue.Queue = queue.Queue()

        def read_output():
            try:
                for line in process.stdout:
                    output.put(line)
            finally:
                output.put(None)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        finished = False
        while not finished:
            if self.cancelled() and process.poll() is None:
                self.log("Cancellation requested; terminating subprocess")
                process.terminate()
            try:
                line = output.get(timeout=0.1)
                if line is None:
                    finished = True
                elif line.strip():
                    self.log(line.strip())
            except queue.Empty:
                if process.poll() is not None:
                    finished = True
        reader.join(timeout=1)
        if process.stdout:
            process.stdout.close()
        try:
            return process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            return process.wait()


class JobManager:
    def __init__(self, root: str, ttl_hours: int = 24):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
        self._lock = threading.RLock()

    def create(self, kind: str) -> JobContext:
        job_id = uuid.uuid4().hex
        context = JobContext(self, job_id)
        for directory in (context.input_dir, context.work_dir, context.output_dir):
            directory.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc)
        self._write(
            job_id,
            {
                "id": job_id,
                "kind": kind,
                "state": "queued",
                "message": "Queued",
                "progress": {"completed": 0, "total": 0},
                "logs": [],
                "cancel_requested": False,
                "created_at": now.isoformat(),
                "expires_at": (now + self.ttl).isoformat(),
                "result": None,
                "artifacts": [],
            },
        )
        return context

    def submit(self, kind: str, runner: Callable[[JobContext], None]) -> dict:
        self.cleanup_expired()
        context = self.create(kind)

        def run() -> None:
            self._update(context.job_id, state="running", message="Running")
            try:
                runner(context)
                job = self.get(context.job_id)
                if job.get("cancel_requested"):
                    self._update(context.job_id, state="cancelled", message="Cancelled")
                elif not job.get("result"):
                    raise RuntimeError("Job completed without a result")
                else:
                    self._update(context.job_id, state="succeeded", message="Complete")
            except Exception as exc:
                context.log(f"ERROR: {exc}")
                self._update(context.job_id, state="failed", message=str(exc))

        threading.Thread(target=run, name=f"alexandria-job-{context.job_id[:8]}", daemon=True).start()
        return self.get(context.job_id)

    def get(self, job_id: str) -> dict:
        path = self._state_path(job_id)
        if not path.exists():
            raise KeyError(job_id)
        with self._lock, path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def cancel(self, job_id: str) -> dict:
        job = self.get(job_id)
        if job["state"] in {"queued", "running"}:
            self._update(job_id, cancel_requested=True, message="Cancellation requested")
        return self.get(job_id)

    def delete(self, job_id: str) -> None:
        """Delete a completed workspace on explicit client request."""
        job = self.get(job_id)
        if job["state"] in {"queued", "running"}:
            raise ValueError("Cannot delete a running job")
        shutil.rmtree(self.root / job_id)

    def result_path(self, job_id: str) -> tuple[Path, dict]:
        job = self.get(job_id)
        result = job.get("result")
        if job.get("state") != "succeeded" or not result:
            raise ValueError("Job result is not ready")
        path = (self.root / job_id / result["path"]).resolve()
        workspace = (self.root / job_id).resolve()
        if workspace not in path.parents or not path.is_file():
            raise FileNotFoundError("Job result is missing")
        return path, result

    def artifact_path(self, job_id: str, artifact_id: str) -> tuple[Path, dict]:
        job = self.get(job_id)
        artifact = next((item for item in job.get("artifacts", []) if item.get("id") == artifact_id), None)
        if not artifact:
            raise FileNotFoundError("Job artifact is missing")
        path = (self.root / job_id / artifact["path"]).resolve()
        workspace = (self.root / job_id).resolve()
        if workspace not in path.parents or not path.is_file():
            raise FileNotFoundError("Job artifact is missing")
        return path, artifact

    def cleanup_expired(self) -> int:
        removed = 0
        now = datetime.now(timezone.utc)
        for directory in self.root.iterdir():
            if not directory.is_dir():
                continue
            try:
                job = self.get(directory.name)
                if datetime.fromisoformat(job["expires_at"]) <= now and job["state"] not in {"running", "queued"}:
                    shutil.rmtree(directory)
                    removed += 1
            except (KeyError, ValueError, OSError, json.JSONDecodeError):
                continue
        return removed

    def _state_path(self, job_id: str) -> Path:
        return self.root / job_id / "state.json"

    def _write(self, job_id: str, data: dict) -> None:
        path = self._state_path(job_id)
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False)
        os.replace(temporary, path)

    def _update(self, job_id: str, **changes) -> None:
        with self._lock:
            job = self.get(job_id)
            job.update(changes)
            self._write(job_id, job)

    def _append_log(self, job_id: str, message: str) -> None:
        with self._lock:
            job = self.get(job_id)
            logs = job.setdefault("logs", [])
            logs.append({"at": datetime.now(timezone.utc).isoformat(), "message": message})
            del logs[:-500]
            self._write(job_id, job)
