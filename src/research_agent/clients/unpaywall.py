"""Unpaywall: open-access copies of a paper, by DOI. Requires a contact email."""

from typing import Any

from pydantic import BaseModel, ConfigDict

from research_agent.clients.http import ApiClient
from research_agent.config import Settings


class OaLocation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    url: str | None = None
    url_for_pdf: str | None = None
    host_type: str | None = None  # publisher or repository
    version: str | None = None  # submittedVersion, acceptedVersion, publishedVersion
    license: str | None = None


class UnpaywallRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    doi: str
    is_oa: bool
    best_oa_location: OaLocation | None = None
    oa_locations: list[OaLocation] = []


class Unpaywall(ApiClient):
    def __init__(self, settings: Settings, **kwargs: Any) -> None:
        if not settings.contact_email:
            raise ValueError("Unpaywall needs a contact email: set RA_CONTACT_EMAIL in .env.")
        super().__init__(
            service="unpaywall",
            base_url="https://api.unpaywall.org",
            min_interval=0.1,
            params={"email": settings.contact_email},
            **kwargs,
        )

    def lookup(self, doi: str) -> UnpaywallRecord | None:
        response = self.get(f"/v2/{doi}", allow_404=True)
        return UnpaywallRecord.model_validate(response.json()) if response else None
