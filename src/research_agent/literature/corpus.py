"""A run's corpus: the papers it found, how it found them, and what it decided."""

import json
import sqlite3
from typing import Any

from research_agent.literature.papers import SOURCE_PRIORITY, PaperRecord
from research_agent.storage import Database

# Reciprocal rank fusion constant: rank r in a channel scores 1 / (RRF_K + r).
RRF_K = 60
# What a run keeps of the records it fetched, to rank on.
SEEN = ("title", "abstract", "year", "citation_count")


class Corpus:
    def __init__(self, db: Database, run_id: str) -> None:
        self.db = db
        self.run_id = run_id

    def add(self, paper_id: str, found: dict[str, Any], record: PaperRecord | None = None) -> None:
        """Record that this run found ``paper_id``, how (channel, rank, seed, ...), and what the record said."""
        with self.db.transaction() as conn:
            row = conn.execute(
                "SELECT found_by, seen FROM run_papers WHERE run_id = ? AND paper_id = ?",
                (self.run_id, paper_id),
            ).fetchone()
            seen = json.loads(row["seen"]) if row is not None and row["seen"] else None
            if record is not None:
                seen = _seen(seen, record)
            seen_json = json.dumps(seen) if seen else None
            if row is None:
                conn.execute(
                    "INSERT INTO run_papers (run_id, paper_id, found_by, seen) VALUES (?, ?, ?, ?)",
                    (self.run_id, paper_id, json.dumps([found]), seen_json),
                )
            else:
                conn.execute(
                    "UPDATE run_papers SET found_by = ?, seen = ? WHERE run_id = ? AND paper_id = ?",
                    (json.dumps([*json.loads(row["found_by"]), found]), seen_json, self.run_id, paper_id),
                )

    def papers(self, read_level: str | None = None) -> list[sqlite3.Row]:
        """Corpus papers joined with their paper rows, best-ranked first, then in the order found."""
        sql = (
            "SELECT p.*, r.found_by, r.score, r.rank, r.read_level, r.reason, r.relevance FROM run_papers r"
            " JOIN papers p USING (paper_id) WHERE r.run_id = ?"
        )
        params: tuple[Any, ...] = (self.run_id,)
        if read_level:
            sql += " AND r.read_level = ?"
            params += (read_level,)
        return self.db.query(sql + " ORDER BY r.rank IS NULL, r.rank, r.rowid", params)

    def candidates(self) -> list[dict[str, Any]]:
        """Corpus papers as this run's own records described them, in the order it found them.

        The ranking reads these, not the paper rows: a paper row merges what every run
        has fetched since, so a fresher citation count fetched later would reorder a
        rerun that reads the same kept responses. Runs from before the run's view was
        kept fall back to the paper row.
        """
        rows = self.db.query(
            "SELECT p.*, r.found_by, r.seen FROM run_papers r JOIN papers p USING (paper_id)"
            " WHERE r.run_id = ? ORDER BY r.rowid",
            (self.run_id,),
        )
        candidates = []
        for row in rows:
            paper = dict(row)
            if row["seen"]:
                seen = json.loads(row["seen"])
                paper.update({field: seen.get(field) for field in SEEN})
            candidates.append(paper)
        return candidates

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


def _seen(before: dict[str, Any] | None, record: PaperRecord) -> dict[str, Any]:
    """A run's view of a paper after one more record, merged as the paper store merges sources."""
    seen = dict(before or {})
    trusted = SOURCE_PRIORITY[record.source] > SOURCE_PRIORITY.get(seen.get("source"), -1)
    for field in ("title", "abstract", "year"):
        value = getattr(record, field)
        if value is not None and (trusted or seen.get(field) is None):
            seen[field] = value
    cited = seen.get("citation_count")
    if record.citation_count is not None and record.citation_count > (-1 if cited is None else cited):
        seen["citation_count"] = record.citation_count
    if trusted:
        seen["source"] = record.source
    return seen
