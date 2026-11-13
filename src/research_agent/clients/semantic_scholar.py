"""Semantic Scholar: the Academic Graph and Recommendations APIs.

RATE LIMIT, to be kept strictly at all times: our API key allows 1 request per
second, cumulative across all endpoints. Every Semantic Scholar request must go
through this client, which spaces request starts at least ``MIN_INTERVAL``
seconds apart across all processes on this machine (``FilePacer``), retries
included. Never call api.semanticscholar.org directly, in parallel, or from a
second client that bypasses this pacing. Prefer the batch endpoint to many
single-paper requests.

Paper IDs can be Semantic Scholar IDs or prefixed external IDs such as
``arXiv:2106.09685``, ``DOI:10.1038/nature14539`` or ``CorpusId:215416146``.
"""

import tempfile
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from research_agent.clients.http import ApiClient, FilePacer
from research_agent.config import Settings

PAPER_FIELDS = "paperId,externalIds,title,abstract,year,venue,authors,citationCount,openAccessPdf"
BATCH_SIZE = 500  # the most IDs /paper/batch accepts

# 1 request per second, plus a margin for network jitter.
MIN_INTERVAL = 1.1
# Shared by every process on this machine, whatever its data directory.
RATE_FILE = Path(tempfile.gettempdir()) / "research-agent-semantic-scholar.last-request"


class _Model(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class S2Author(_Model):
    author_id: str | None = Field(None, alias="authorId")
    name: str | None = None


class S2Paper(_Model):
    paper_id: str | None = Field(None, alias="paperId")  # None for unresolved references
    title: str | None = None
    abstract: str | None = None
    year: int | None = None
    venue: str | None = None
    external_ids: dict[str, Any] | None = Field(None, alias="externalIds")
    authors: list[S2Author] = []
    citation_count: int | None = Field(None, alias="citationCount")
    open_access_pdf: dict[str, Any] | None = Field(None, alias="openAccessPdf")


class S2Snippet(_Model):
    """A passage from inside a paper, as returned by snippet search."""

    text: str
    kind: str | None = None
    section: str | None = None
    score: float | None = None
    corpus_id: str | None = None
    paper_title: str | None = None


class SemanticScholar(ApiClient):
    def __init__(self, settings: Settings, *, rate_file: Path = RATE_FILE, **kwargs: Any) -> None:
        # Without a key, requests share a public rate limit that is often saturated,
        # so most of them get 429s; retry longer than for the other APIs.
        key = settings.s2_api_key.get_secret_value() if settings.s2_api_key else None
        kwargs.setdefault("max_attempts", 5 if key else 10)
        super().__init__(
            service="semantic_scholar",
            base_url="https://api.semanticscholar.org",
            headers={"x-api-key": key} if key else None,
            pacer=FilePacer(rate_file, MIN_INTERVAL, kwargs.get("sleep") or time.sleep),
            **kwargs,
        )

    def paper(self, paper_id: str, *, fields: str = PAPER_FIELDS) -> S2Paper | None:
        response = self.get(f"/graph/v1/paper/{paper_id}", params={"fields": fields}, allow_404=True)
        return S2Paper.model_validate(response.json()) if response else None

    def batch(self, paper_ids: list[str], *, fields: str = PAPER_FIELDS) -> list[S2Paper | None]:
        """Look up many papers, one request per 500 IDs. The result lines up with
        ``paper_ids``; unknown IDs give None."""
        papers: list[S2Paper | None] = []
        for start in range(0, len(paper_ids), BATCH_SIZE):
            chunk = paper_ids[start : start + BATCH_SIZE]
            response = self.post("/graph/v1/paper/batch", params={"fields": fields}, json={"ids": chunk})
            papers += [S2Paper.model_validate(p) if p else None for p in response.json()]
        return papers

    def search(
        self, query: str, *, limit: int = 20, fields: str = PAPER_FIELDS, year: str | None = None
    ) -> list[S2Paper]:
        """``year`` limits publication years, e.g. "2025-" or "2020-2023"."""
        params: dict[str, Any] = {"query": query, "limit": limit, "fields": fields}
        if year:
            params["year"] = year
        response = self.get("/graph/v1/paper/search", params=params)
        return [S2Paper.model_validate(p) for p in response.json().get("data") or []]

    def references(
        self, paper_id: str, *, limit: int = 100, fields: str = PAPER_FIELDS
    ) -> list[S2Paper]:
        """Papers this paper cites."""
        response = self.get(
            f"/graph/v1/paper/{paper_id}/references", params={"limit": limit, "fields": fields}
        )
        return [S2Paper.model_validate(r["citedPaper"]) for r in response.json().get("data") or []]

    def citations(
        self, paper_id: str, *, limit: int = 100, fields: str = PAPER_FIELDS
    ) -> list[S2Paper]:
        """Papers that cite this paper."""
        response = self.get(
            f"/graph/v1/paper/{paper_id}/citations", params={"limit": limit, "fields": fields}
        )
        return [S2Paper.model_validate(c["citingPaper"]) for c in response.json().get("data") or []]

    def recommendations(
        self, paper_id: str, *, limit: int = 20, fields: str = PAPER_FIELDS
    ) -> list[S2Paper]:
        response = self.get(
            f"/recommendations/v1/papers/forpaper/{paper_id}",
            params={"limit": limit, "fields": fields},
        )
        return [S2Paper.model_validate(p) for p in response.json().get("recommendedPapers") or []]

    def snippets(self, query: str, *, limit: int = 10) -> list[S2Snippet]:
        """Search passages inside papers."""
        response = self.get("/graph/v1/snippet/search", params={"query": query, "limit": limit})
        snippets = []
        for item in response.json().get("data") or []:
            snippet, paper = item.get("snippet") or {}, item.get("paper") or {}
            snippets.append(
                S2Snippet(
                    text=snippet.get("text", ""),
                    kind=snippet.get("snippetKind"),
                    section=snippet.get("section"),
                    score=item.get("score"),
                    corpus_id=str(paper["corpusId"]) if paper.get("corpusId") else None,
                    paper_title=paper.get("title"),
                )
            )
        return snippets
