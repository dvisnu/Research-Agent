"""Kept responses of the literature search APIs, so a rerun sees the results its run saw.

A search engine's answer to the same query drifts from day to day (new papers,
re-ranking, updated citation counts), and every downstream stage follows from it, so
without a cache a rerun of the same question can read different papers and write a
different plan. Successful responses (200, and 404 where the caller allows it) are
kept in their own SQLite file, keyed on the request without its credentials, and
reused while younger than ``max_age_days``. ``max_age_days=0`` reads nothing but still
writes, so a run that refreshes its searches is itself reproducible afterwards.

Full texts (arxiv.org pages, PDFs) are not kept here: their parses are kept as paper
texts (``literature.texts``).
"""

import hashlib
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from research_agent.storage import utc_now

_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key           TEXT PRIMARY KEY,   -- sha256 of the method, the URL without credentials and the body
    service       TEXT NOT NULL,
    method        TEXT NOT NULL,
    url           TEXT NOT NULL,      -- without credentials
    status        INTEGER NOT NULL,
    content_type  TEXT,
    content       BLOB NOT NULL,
    fetched_at    TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class Kept:
    status: int
    content_type: str | None
    content: bytes
    fetched_at: str

    def response(self, request: httpx.Request) -> httpx.Response:
        headers = {"content-type": self.content_type} if self.content_type else {}
        return httpx.Response(self.status, headers=headers, content=self.content, request=request)


class ResponseCache:
    def __init__(self, path: Path, max_age_days: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.max_age_days = max_age_days
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(_SCHEMA)
        self._lock = threading.Lock()

    @staticmethod
    def key(method: str, url: str, body: bytes) -> str:
        hasher = hashlib.sha256()
        for part in (method.encode(), url.encode(), body):
            hasher.update(part)
            hasher.update(b"\x00")
        return hasher.hexdigest()

    def get(self, key: str) -> Kept | None:
        """The kept response, if there is one young enough."""
        if self.max_age_days <= 0:
            return None
        oldest = (datetime.now(UTC) - timedelta(days=self.max_age_days)).isoformat(timespec="milliseconds")
        with self._lock:
            row = self._conn.execute(
                "SELECT status, content_type, content, fetched_at FROM responses WHERE key = ? AND fetched_at >= ?",
                (key, oldest),
            ).fetchone()
        return Kept(*row) if row else None

    def put(self, key: str, service: str, method: str, url: str, response: httpx.Response) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses (key, service, method, url, status, content_type, content, fetched_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (key, service, method, url, response.status_code, response.headers.get("content-type"),
                 response.content, utc_now()),
            )

    def close(self) -> None:
        self._conn.close()
