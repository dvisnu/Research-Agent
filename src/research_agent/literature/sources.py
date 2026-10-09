"""The literature sources a run can use, built from settings."""

from dataclasses import dataclass
from typing import Any, Self

from research_agent.clients import Arxiv, OpenAlex, SemanticScholar, Unpaywall
from research_agent.clients.http import ApiClient
from research_agent.config import Settings

# arxiv.org's robots.txt sets Crawl-delay: 15 for /html and /pdf.
ARXIV_WEB_INTERVAL = 15.0
# Open-access PDFs come from many hosts; one request every 2 s across all of them.
DOWNLOAD_INTERVAL = 2.0
MAX_DOWNLOAD_BYTES = 50_000_000


class ArxivWeb(ApiClient):
    """arxiv.org itself, for full text (HTML and PDF), paced by its robots.txt."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("min_interval", ARXIV_WEB_INTERVAL)
        super().__init__(service="arxiv_web", base_url="https://arxiv.org", timeout=60.0, **kwargs)

    def html(self, arxiv_id: str) -> str | None:
        response = self.get(f"/html/{arxiv_id}", allow_404=True)
        return response.text if response else None

    def pdf(self, arxiv_id: str) -> bytes | None:
        response = self.get(f"/pdf/{arxiv_id}", allow_404=True)
        return response.content if response else None


class Downloader(ApiClient):
    """Fetches open-access PDFs from any host."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("min_interval", DOWNLOAD_INTERVAL)
        super().__init__(service="download", base_url="", timeout=60.0, max_attempts=3, **kwargs)

    def pdf(self, url: str) -> bytes | None:
        """The PDF at ``url``, or None if the URL does not serve a PDF."""
        response = self.get(url, allow_404=True)
        if response is None or len(response.content) > MAX_DOWNLOAD_BYTES:
            return None
        return response.content if response.content[:5] == b"%PDF-" else None


@dataclass
class Sources:
    arxiv: Arxiv
    arxiv_web: ArxivWeb
    openalex: OpenAlex
    downloader: Downloader
    s2: SemanticScholar | None  # only with an API key
    unpaywall: Unpaywall | None  # only with a contact email

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(
            arxiv=Arxiv(),
            arxiv_web=ArxivWeb(),
            openalex=OpenAlex(settings),
            downloader=Downloader(),
            s2=SemanticScholar(settings) if settings.s2_api_key else None,
            unpaywall=Unpaywall(settings) if settings.contact_email else None,
        )

    def close(self) -> None:
        for client in (self.arxiv, self.arxiv_web, self.openalex, self.downloader, self.s2, self.unpaywall):
            if client is not None:
                client.close()
