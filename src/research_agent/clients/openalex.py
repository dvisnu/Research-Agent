"""OpenAlex works API. Sending a contact email puts requests in OpenAlex's polite pool."""

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from research_agent.clients.http import ApiClient
from research_agent.config import Settings

# Works whose primary topic is in the Computer Science field.
CS_FILTER = "primary_topic.field.id:17"
# The most IDs one `openalex_id:` filter takes.
IDS_PER_REQUEST = 50


class OpenAlexWork(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: str
    doi: str | None = None
    title: str | None = Field(None, alias="display_name")
    publication_year: int | None = None
    ids: dict[str, Any] = {}
    open_access: dict[str, Any] | None = None
    best_oa_location: dict[str, Any] | None = None
    primary_location: dict[str, Any] | None = None
    authorships: list[dict[str, Any]] = []
    cited_by_count: int | None = None
    referenced_works: list[str] = []
    related_works: list[str] = []
    abstract: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _rebuild_abstract(cls, data: Any) -> Any:
        if isinstance(data, dict) and "abstract_inverted_index" in data:
            data = {**data, "abstract": abstract_from_inverted_index(data["abstract_inverted_index"])}
        return data

    @property
    def venue(self) -> str | None:
        return ((self.primary_location or {}).get("source") or {}).get("display_name")

    @property
    def author_names(self) -> list[str]:
        return [
            a["author"]["display_name"]
            for a in self.authorships
            if (a.get("author") or {}).get("display_name")
        ]


def abstract_from_inverted_index(index: dict[str, list[int]] | None) -> str | None:
    """OpenAlex ships abstracts as {word: [positions]}; put the words back in order."""
    if not index:
        return None
    positions = {pos: word for word, places in index.items() for pos in places}
    return " ".join(positions[pos] for pos in sorted(positions))


def short_id(work_id: str) -> str:
    """``https://openalex.org/W123`` -> ``W123``."""
    return work_id.rsplit("/", 1)[-1]


class OpenAlex(ApiClient):
    def __init__(self, settings: Settings, **kwargs: Any) -> None:
        super().__init__(
            service="openalex",
            base_url="https://api.openalex.org",
            min_interval=0.1,
            params={"mailto": settings.contact_email} if settings.contact_email else None,
            headers=(
                {"Authorization": f"Bearer {settings.openalex_api_key.get_secret_value()}"}
                if settings.openalex_api_key else None
            ),
            **kwargs,
        )

    def work(self, work_id: str) -> OpenAlexWork | None:
        """``work_id`` is an OpenAlex ID (``W...``) or ``doi:10....``."""
        response = self.get(f"/works/{work_id}", allow_404=True)
        return OpenAlexWork.model_validate(response.json()) if response else None

    def search(
        self, query: str, *, per_page: int = 25, filter: str | None = None
    ) -> list[OpenAlexWork]:
        # OpenAlex rejects some punctuation in search (e.g. "?" gives HTTP 400).
        query = " ".join(re.sub(r"[^\w\s-]", " ", query).split())
        params: dict[str, Any] = {"search": query, "per-page": per_page}
        if filter:
            params["filter"] = filter
        response = self.get("/works", params=params)
        return [OpenAlexWork.model_validate(w) for w in response.json().get("results") or []]

    def works(self, work_ids: list[str]) -> list[OpenAlexWork]:
        """Look up works by OpenAlex ID, ``IDS_PER_REQUEST`` per request."""
        found = []
        ids = [short_id(w) for w in work_ids]
        for start in range(0, len(ids), IDS_PER_REQUEST):
            chunk = ids[start : start + IDS_PER_REQUEST]
            response = self.get(
                "/works", params={"filter": "openalex_id:" + "|".join(chunk), "per-page": len(chunk)}
            )
            found += [OpenAlexWork.model_validate(w) for w in response.json().get("results") or []]
        return found

    def citing(
        self, work_id: str, *, per_page: int = 25, sort: str = "cited_by_count:desc"
    ) -> list[OpenAlexWork]:
        """Works that cite ``work_id``."""
        response = self.get(
            "/works",
            params={"filter": f"cites:{short_id(work_id)}", "sort": sort, "per-page": per_page},
        )
        return [OpenAlexWork.model_validate(w) for w in response.json().get("results") or []]
