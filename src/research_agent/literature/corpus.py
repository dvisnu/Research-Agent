"""A run's corpus: the papers it found, how it found them, and what it decided."""

import json
import sqlite3
from typing import Any

from research_agent.storage import Database

# Reciprocal rank fusion constant: rank r in a channel scores 1 / (RRF_K + r).
RRF_K = 60


class Corpus:
    def __init__(self, db: Database, run_id: str) -> None:
        self.db = db
        self.run_id = run_id

    def add(self, paper_id: str, found: dict[str, Any]) -> None:
        """Record that this run found ``paper_id``, and how (channel, rank, seed, ...)."""
        with self.db.transaction() as conn:
            row = conn.execute(
                "SELECT found_by FROM run_papers WHERE run_id = ? AND paper_id = ?",
                (self.run_id, paper_id),
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO run_papers (run_id, paper_id, found_by) VALUES (?, ?, ?)",
                    (self.run_id, paper_id, json.dumps([found])),
                )
            else:
                conn.execute(
                    "UPDATE run_papers SET found_by = ? WHERE run_id = ? AND paper_id = ?",
                    (json.dumps([*json.loads(row["found_by"]), found]), self.run_id, paper_id),
                )

    def papers(self, read_level: str | None = None) -> list[sqlite3.Row]:
        """Corpus papers joined with their paper rows, best-ranked first."""
        sql = (
            "SELECT p.*, r.found_by, r.score, r.rank, r.read_level, r.reason, r.relevance FROM run_papers r"
            " JOIN papers p USING (paper_id) WHERE r.run_id = ?"
        )
        params: tuple[Any, ...] = (self.run_id,)
        if read_level:
            sql += " AND r.read_level = ?"
            params += (read_level,)
        return self.db.query(sql + " ORDER BY r.rank IS NULL, r.rank, p.paper_id", params)

    def search_leaders(self, count: int) -> list[sqlite3.Row]:
        """The papers the search channels agree on most (reciprocal rank fusion)."""
        def fused(row: sqlite3.Row) -> float:
            return sum(1 / (RRF_K + f["rank"]) for f in json.loads(row["found_by"]) if "rank" in f)

        return sorted(self.papers(), key=fused, reverse=True)[:count]

    def set_rank(
        self, paper_id: str, *, score: float, rank: int, read_level: str, reason: str, relevance: str | None = None
    ) -> None:
        self.db.execute(
            "UPDATE run_papers SET score = ?, rank = ?, read_level = ?, reason = ?, relevance = ?"
            " WHERE run_id = ? AND paper_id = ?",
            (score, rank, read_level, reason, relevance, self.run_id, paper_id),
        )

    def set_level(self, paper_id: str, read_level: str, reason: str) -> None:
        self.db.execute(
            "UPDATE run_papers SET read_level = ?, reason = ? WHERE run_id = ? AND paper_id = ?",
            (read_level, reason, self.run_id, paper_id),
        )
