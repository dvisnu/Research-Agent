"""Settings, read from `RA_*` environment variables and `.env`."""

from pathlib import Path
from typing import Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RA_", env_file=".env", env_ignore_empty=True, extra="ignore"
    )

    # Where the database, checkpoints and run reports go.
    data_dir: Path = Path("var")

    # LLM roles, each "provider:model", or several separated by commas: the role uses
    # the first until its quota runs out, then the next. Unset roles fail when first used.
    llm_extraction: str | None = None
    llm_reasoning: str | None = None
    llm_verifier: str | None = None

    # Text embeddings for linking claims across papers, "provider:model" (Phase 2).
    # Unset: claims are linked only through shared datasets and metrics.
    embedding: str | None = None
    embedding_dims: int = Field(768, ge=128, le=3072)

    # API keys for the LLM and embedding providers.
    google_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None

    # Semantic Scholar works without a key; requests then share a public rate limit.
    s2_api_key: SecretStr | None = None
    # Required by Unpaywall; also puts OpenAlex requests in its polite pool.
    contact_email: str | None = None
    # OpenAlex works without a key, but keyless requests share a small daily budget
    # with everyone on the same IP address; a free key has its own.
    openalex_api_key: SecretStr | None = None

    # Corpus size per run: the top papers read in full, then the next ones read by abstract.
    # Development sizes, chosen to fit the free LLM tiers; see END_TODO.md.
    full_read: int = Field(10, ge=0)
    abstract_only: int = Field(30, ge=0)

    # Development only: reuse full texts parsed by an earlier run instead of fetching
    # them again. Off by default, so a normal run fetches every text fresh.
    reuse_texts: bool = False

    # The search APIs' responses are kept and reused for this many days, so a rerun of a
    # question finds the same papers. 0 fetches fresh (and keeps what it fetched).
    response_cache_days: float = Field(30, ge=0)

    @field_validator("llm_extraction", "llm_reasoning", "llm_verifier")
    @classmethod
    def _check_model_spec(cls, value: str | None) -> str | None:
        if value is not None:
            for spec in model_specs(value) or [value]:
                provider, _, model = spec.partition(":")
                if not provider or not model:
                    # The value is not echoed: it may be an API key in the wrong variable.
                    raise ValueError("expected 'provider:model', or several separated by commas")
        return value

    @field_validator("embedding")
    @classmethod
    def _check_embedding_spec(cls, value: str | None) -> str | None:
        if value is not None:
            provider, _, model = value.strip().partition(":")
            if not provider or not model or "," in model:
                raise ValueError("expected 'provider:model'")
        return value

    @model_validator(mode="after")
    def _check_verifier_differs(self) -> Self:
        if set(model_specs(self.llm_verifier)) & set(model_specs(self.llm_reasoning)):
            raise ValueError(
                "the verifier role must use a different provider or model family from reasoning"
            )
        return self

    @property
    def db_path(self) -> Path:
        return self.data_dir / "agent.db"

    @property
    def checkpoint_path(self) -> Path:
        return self.data_dir / "checkpoints.db"

    @property
    def responses_path(self) -> Path:
        return self.data_dir / "responses.db"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"


def model_specs(value: str | None) -> list[str]:
    """The "provider:model" entries of a role setting, in fallback order."""
    return [spec.strip() for spec in (value or "").split(",") if spec.strip()]
