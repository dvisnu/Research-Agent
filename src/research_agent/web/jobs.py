"""Runs started from the web UI, each an `agent run` or `agent resume` in a child process.

A child process, not a thread: a run takes hours and loads SPECTER2, and if it crashes
or is killed the server carries on. Each child reads .env afresh, so a key switched
after a quota ran out counts on the next resume without restarting the UI. A child is
in a process group of its own, so stopping the server (Ctrl+C) leaves its runs going.

One run at a time: the free tiers' quotas and the machine's memory don't stretch to two.
"""

import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from research_agent.config import Settings
from research_agent.storage import Database, utc_now

# What a run started here writes to the console goes to this file in its run directory.
CONSOLE_LOG = "console.log"


class Busy(RuntimeError):
    """A run started here is still going."""


@dataclass
class Job:
    run_id: str
    action: str  # run, resume
    process: subprocess.Popen[bytes]
    started_at: str

    @property
    def alive(self) -> bool:
        return self.process.poll() is None


class Jobs:
    def __init__(self, settings: Settings, command: list[str] | None = None) -> None:
        self.settings = settings
        self._command = command or [sys.executable, "-m", "research_agent"]
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()

    def active(self) -> Job | None:
        with self._lock:
            return next((job for job in self._jobs.values() if job.alive), None)

    def get(self, run_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(run_id)

    def start(self, run_id: str, question: str, uploads: list[Path], *, refresh: bool = False) -> Job:
        args = ["run", "--run-id", run_id, *(f"--upload={path}" for path in uploads)]
        if refresh:
            args.append("--refresh")
        return self._spawn(run_id, "run", [*args, "--", question])

    def resume(self, run_id: str) -> Job:
        return self._spawn(run_id, "resume", ["resume", run_id])

    def stop(self, run_id: str) -> bool:
        """Stop a run started here. It keeps the stages it finished, so it can be resumed."""
        job = self.get(run_id)
        if job is None or not job.alive:
            return False
        job.process.terminate()
        try:
            job.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            job.process.kill()
        with Database(self.settings.db_path) as db:
            row = db.get_run(run_id)
            if row is not None and row["status"] == "running":
                db.set_run_status(run_id, "failed", "Stopped from the web UI.")
        return True

    def _spawn(self, run_id: str, action: str, args: list[str]) -> Job:
        with self._lock:
            active = self.active()
            if active is not None:
                raise Busy(f"Run {active.run_id} is still going; one run at a time.")
            out_dir = self.settings.runs_dir / run_id
            out_dir.mkdir(parents=True, exist_ok=True)
            env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
            group = (
                {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
                if sys.platform == "win32" else {"start_new_session": True}
            )
            with (out_dir / CONSOLE_LOG).open("ab") as log:
                log.write(f"\n=== {action} {run_id} at {utc_now()} ===\n".encode())
                log.flush()
                process = subprocess.Popen(
                    [*self._command, *args], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    env=env, cwd=Path.cwd(), **group,
                )
            job = Job(run_id, action, process, utc_now())
            self._jobs[run_id] = job
            return job
