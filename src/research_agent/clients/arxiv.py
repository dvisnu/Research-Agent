"""arXiv API (Atom feed). arXiv asks for at most one request every three seconds."""

import re
import xml.etree.ElementTree as ET
from typing import Any

from pydantic import BaseModel

from research_agent.clients.http import ApiClient

_NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
_ABS_ID = re.compile(r"arxiv\.org/abs/(?P<id>.+)v(?P<version>\d+)$")


class ArxivEntry(BaseModel):
    arxiv_id: str  # without the version suffix
    version: int
    title: str
    summary: str
    authors: list[str]
    published: str
    updated: str
    categories: list[str]
    primary_category: str | None = None
    doi: str | None = None
    pdf_url: str | None = None


class Arxiv(ApiClient):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            service="arxiv", base_url="https://export.arxiv.org", min_interval=3.0, **kwargs
        )

    def search(
        self, query: str, *, max_results: int = 20, start: int = 0, sort_by: str = "relevance"
    ) -> list[ArxivEntry]:
        """``query`` uses arXiv's syntax, e.g. ``all:"speculative decoding"`` or ``cat:cs.LG``."""
        response = self.get(
            "/api/query",
            params={
                "search_query": query,
                "start": start,
                "max_results": max_results,
                "sortBy": sort_by,
                "sortOrder": "descending",
            },
        )
        return parse_feed(response.text)

    def get_many(self, arxiv_ids: list[str]) -> list[ArxivEntry]:
        """Look up papers by arXiv ID; IDs arXiv doesn't know are left out."""
        response = self.get(
            "/api/query", params={"id_list": ",".join(arxiv_ids), "max_results": len(arxiv_ids)}
        )
        return parse_feed(response.text)


def parse_feed(xml_text: str) -> list[ArxivEntry]:
    root = ET.fromstring(xml_text)
    entries = []
    for entry in root.findall("a:entry", _NS):
        match = _ABS_ID.search(_text(entry, "a:id"))
        if match is None:  # error entries, e.g. for a malformed ID
            continue
        primary = entry.find("arxiv:primary_category", _NS)
        pdf = entry.find("a:link[@title='pdf']", _NS)
        entries.append(
            ArxivEntry(
                arxiv_id=match["id"],
                version=int(match["version"]),
                title=_text(entry, "a:title"),
                summary=_text(entry, "a:summary"),
                authors=[_text(a, "a:name") for a in entry.findall("a:author", _NS)],
                published=_text(entry, "a:published"),
                updated=_text(entry, "a:updated"),
                categories=[c.get("term", "") for c in entry.findall("a:category", _NS)],
                primary_category=primary.get("term") if primary is not None else None,
                doi=_text(entry, "arxiv:doi") or None,
                pdf_url=pdf.get("href") if pdf is not None else None,
            )
        )
    return entries


def _text(element: ET.Element, path: str) -> str:
    """Whitespace-normalised text of a child element, or "" if it is missing."""
    found = element.find(path, _NS)
    return " ".join((found.text or "").split()) if found is not None else ""
