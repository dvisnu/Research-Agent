"""Paper identity: normalising IDs and titles, and merging what different
sources say about one paper into a single row.

Sources disagree, so the title and abstract come from the most trusted source
that has them (``SOURCE_PRIORITY``). OpenAlex ranks low: it has been seen to
attach the wrong title to an arXiv DOI. An uploaded PDF ranks lowest, since its
title is guessed from the file; the upload itself is recorded in the run's
corpus and as the paper's text source.
"""

import json
import re
import sqlite3
from typing import Any

from pydantic import BaseModel, model_validator

from research_agent.clients import ArxivEntry, OpenAlexWork, S2Paper
from research_agent.storage import Database, new_id, utc_now

SOURCE_PRIORITY = {"arxiv": 3, "s2": 2, "openalex": 1, "upload": 0}
ID_COLUMNS = ("arxiv_id", "doi", "s2_id", "openalex_id")
ARXIV_DOI_PREFIX = "10.48550/arxiv."
# Titles at least this long (after normalising) match regardless of year.
LONG_TITLE_KEY = 40
MIN_TITLE_KEY = 15


def normalise_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    doi = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "doi:"):
        doi = doi.removeprefix(prefix)
    return doi or None


def normalise_arxiv_id(arxiv_id: str | None) -> str | None:
    if not arxiv_id:
        return None
    arxiv_id = arxiv_id.strip()
    arxiv_id = re.sub(r"^(arxiv:|https?://arxiv\.org/(abs|pdf|html)/)", "", arxiv_id, flags=re.I)
    arxiv_id = re.sub(r"(\.pdf)?$", "", arxiv_id)
    return re.sub(r"v\d+$", "", arxiv_id) or None


def normalise_openalex_id(openalex_id: str | None) -> str | None:
    return openalex_id.rsplit("/", 1)[-1] if openalex_id else None


def title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", title.lower())


class PaperRecord(BaseModel):
    """What one source says about a paper."""

    source: str
    title: str
    abstract: str | None = None
    year: int | None = None
    venue: str | None = None
    authors: list[str] = []
    arxiv_id: str | None = None
    doi: str | None = None
    s2_id: str | None = None
    openalex_id: str | None = None
    citation_count: int | None = None
    pdf_url: str | None = None

    @model_validator(mode="after")
    def _normalise(self) -> "PaperRecord":
        self.title = " ".join(self.title.split())
        self.abstract = " ".join(self.abstract.split()) if self.abstract else None
        self.doi = normalise_doi(self.doi)
        if self.doi and self.doi.startswith(ARXIV_DOI_PREFIX):
            self.arxiv_id = self.arxiv_id or self.doi.removeprefix(ARXIV_DOI_PREFIX)
            self.doi = None  # arXiv DOIs stand for the arXiv ID; keep the column for the published DOI
        self.arxiv_id = normalise_arxiv_id(self.arxiv_id)
        self.openalex_id = normalise_openalex_id(self.openalex_id)
        self.pdf_url = self.pdf_url or None
        return self


def from_arxiv(entry: ArxivEntry) -> PaperRecord:
    return PaperRecord(
        source="arxiv",
        title=entry.title,
        abstract=entry.summary,
        year=int(entry.published[:4]) if entry.published else None,
        authors=entry.authors,
        arxiv_id=entry.arxiv_id,
        doi=entry.doi,
    )


def from_s2(paper: S2Paper) -> PaperRecord | None:
    if not paper.title:
        return None
    ids = paper.external_ids or {}
    return PaperRecord(
        source="s2",
        title=paper.title,
        abstract=paper.abstract,
        year=paper.year,
        venue=paper.venue or None,
        authors=[a.name for a in paper.authors if a.name],
        arxiv_id=ids.get("ArXiv"),
        doi=ids.get("DOI"),
        s2_id=paper.paper_id,
        citation_count=paper.citation_count,
        pdf_url=(paper.open_access_pdf or {}).get("url"),
    )


def from_openalex(work: OpenAlexWork) -> PaperRecord | None:
    if not work.title:
        return None
    return PaperRecord(
        source="openalex",
        title=work.title,
        abstract=work.abstract,
        year=work.publication_year,
        venue=work.venue,
        authors=work.author_names,
        doi=work.doi,
        openalex_id=work.id,
        citation_count=work.cited_by_count,
        pdf_url=(work.best_oa_location or {}).get("pdf_url"),
    )


class PaperStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    def upsert(self, record: PaperRecord) -> str:
        """Store a record, merging it into the matching paper if there is one."""
        with self.db.transaction() as conn:
            row = self._match(conn, record)
            if row is None:
                return self._insert(conn, record)
            self._merge(conn, row, record)
            return row["paper_id"]

    def _match(self, conn: sqlite3.Connection, record: PaperRecord) -> sqlite3.Row | None:
        for column in ID_COLUMNS:
            value = getattr(record, column)
            if value:
                row = conn.execute(f"SELECT * FROM papers WHERE {column} = ?", (value,)).fetchone()
                if row:
                    return row
        key = title_key(record.title)
        if len(key) < MIN_TITLE_KEY:
            return None
        for row in conn.execute("SELECT * FROM papers WHERE title_key = ?", (key,)):
            years = (row["year"], record.year)
            if len(key) >= LONG_TITLE_KEY or None in years or abs(years[0] - years[1]) <= 1:
                return row
        return None

    def _insert(self, conn: sqlite3.Connection, record: PaperRecord) -> str:
        paper_id = new_id("p")
        conn.execute(
            "INSERT INTO papers (paper_id, title, abstract, year, venue, authors, arxiv_id, doi,"
            " s2_id, openalex_id, origin, title_key, meta_source, citation_count, pdf_url, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                paper_id,
                record.title,
                record.abstract,
                record.year,
                record.venue,
                json.dumps(record.authors),
                record.arxiv_id,
                record.doi,
                record.s2_id,
                record.openalex_id,
                "user-provided" if record.source == "upload" else "retrieved",
                title_key(record.title),
                record.source,
                record.citation_count,
                record.pdf_url,
                utc_now(),
            ),
        )
        return paper_id

    def _merge(self, conn: sqlite3.Connection, row: sqlite3.Row, record: PaperRecord) -> None:
        updates: dict[str, Any] = {}
        for column in ID_COLUMNS:
            value = getattr(record, column)
            # An ID already held by another row is left alone rather than moved.
            if value and row[column] is None and not conn.execute(
                f"SELECT 1 FROM papers WHERE {column} = ?", (value,)
            ).fetchone():
                updates[column] = value

        trusted = SOURCE_PRIORITY[record.source] > SOURCE_PRIORITY.get(row["meta_source"], -1)
        for column in ("title", "abstract", "year", "venue"):
            value = getattr(record, column)
            if value is not None and (trusted or row[column] is None):
                updates[column] = value
        if record.authors and (trusted or row["authors"] == "[]"):
            updates["authors"] = json.dumps(record.authors)
        if trusted:
            updates["meta_source"] = record.source
        if "title" in updates:
            updates["title_key"] = title_key(record.title)

        if record.citation_count is not None and record.citation_count > (row["citation_count"] or -1):
            updates["citation_count"] = record.citation_count
        if record.pdf_url and not row["pdf_url"]:
            updates["pdf_url"] = record.pdf_url
        if updates:
            assignments = ", ".join(f"{column} = ?" for column in updates)
            conn.execute(
                f"UPDATE papers SET {assignments} WHERE paper_id = ?",
                (*updates.values(), row["paper_id"]),
            )
