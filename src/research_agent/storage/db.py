"""SQLite access: one shared connection, schema creation, run bookkeeping."""

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any, Self

_MIGRATIONS = sorted(
    (path for path in resources.files(__package__).joinpath("migrations").iterdir()
     if path.name.endswith(".sql")),
    key=lambda path: path.name,
)
# Migration files are numbered 0001, 0002, ...; the schema version is the last one applied.
SCHEMA_VERSION = len(_MIGRATIONS)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class Database:
    """A connection safe to share across LangGraph's worker threads."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self._lock = threading.RLock()
        self._migrate()

    def _migrate(self) -> None:
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise RuntimeError(
                f"{self.path} has schema version {version}; this code knows up to {SCHEMA_VERSION}"
            )
        for number, migration in enumerate(_MIGRATIONS[version:], start=version + 1):
            sql = migration.read_text(encoding="utf-8")
            with self._lock:
                self.conn.executescript(f"BEGIN;\n{sql}\nPRAGMA user_version = {number};\nCOMMIT;")

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self.conn.execute(sql, params)

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(sql, params).fetchall()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                yield self.conn
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            self.conn.execute("COMMIT")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # Runs

    def create_run(self, run_id: str, question: str) -> None:
        now = utc_now()
        self.execute(
            "INSERT INTO runs (run_id, question, status, created_at, updated_at)"
            " VALUES (?, ?, 'running', ?, ?)",
            (run_id, question, now, now),
        )

    def set_run_status(self, run_id: str, status: str, error: str | None = None) -> None:
        self.execute(
            "UPDATE runs SET status = ?, error = ?, updated_at = ? WHERE run_id = ?",
            (status, error, utc_now(), run_id),
        )

    def get_run(self, run_id: str) -> sqlite3.Row | None:
        rows = self.query("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        return rows[0] if rows else None

    def list_runs(self, limit: int = 20) -> list[sqlite3.Row]:
        return self.query("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,))
