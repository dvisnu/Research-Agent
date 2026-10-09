"""One real call per literature API, and per configured LLM model. Run with `uv run pytest -m live`."""

import pytest

from research_agent.clients import Arxiv, OpenAlex, SemanticScholar, Unpaywall
from research_agent.config import Settings, model_specs
from research_agent.literature.brief import BriefDraft, brief_prompt
from research_agent.llm import LLM, Role

pytestmark = pytest.mark.live


@pytest.fixture
def env_settings() -> Settings:
    return Settings()  # reads .env, for the S2 key and contact email


def test_arxiv():
    with Arxiv() as client:
        [entry] = client.get_many(["2106.09685"])
    assert entry.title.startswith("LoRA")


def test_semantic_scholar(env_settings):
    with SemanticScholar(env_settings) as client:
        paper = client.paper("arXiv:2106.09685")
    assert paper is not None and paper.title.startswith("LoRA")


def test_openalex(env_settings):
    with OpenAlex(env_settings) as client:
        work = client.work("doi:10.1038/nature14539")
    assert work is not None and work.title == "Deep learning"


def test_unpaywall(env_settings):
    if not env_settings.contact_email:
        pytest.skip("set RA_CONTACT_EMAIL in .env")
    with Unpaywall(env_settings) as client:
        record = client.lookup("10.1038/nature14539")
    assert record is not None and record.doi == "10.1038/nature14539"


def _models(role: Role) -> list[tuple[Role, str]]:
    return [(role, spec) for spec in model_specs(getattr(Settings(), f"llm_{role}"))]


# Every model of each role, fallbacks included, so a dead fallback shows up before it is needed.
@pytest.mark.parametrize(
    ("role", "spec"), [*_models(Role.REASONING), *_models(Role.EXTRACTION), *_models(Role.VERIFIER)]
)
def test_llm_model_writes_a_valid_brief(env_settings, role, spec):
    provider = spec.partition(":")[0]
    if getattr(env_settings, f"{provider}_api_key", None) is None:
        pytest.skip(f"set RA_{provider.upper()}_API_KEY in .env")
    llm = LLM(env_settings.model_copy(update={f"llm_{role}": spec}))
    try:
        draft = llm.structured(role, brief_prompt("How can RAG systems handle conflicting retrieved documents?"), BriefDraft)
    finally:
        llm.close()
    assert draft.angles and all(angle.keywords for angle in draft.angles)

