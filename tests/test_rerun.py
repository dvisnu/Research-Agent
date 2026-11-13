"""Phase 6: a rerun of a question sees what its first run saw."""

import httpx
import pytest

from fakes import FakeModel, fake_services
from research_agent.clients.cache import ResponseCache
from research_agent.clients.http import ApiClient
from research_agent.costs import run_cost
from research_agent.pipeline import start_run
from research_agent.storage import Database
from test_literature_stages import QUESTION, _corpus, _router, small_corpus_settings  # noqa: F401  (a fixture)


def _client(handler, cache: ResponseCache, **kwargs) -> ApiClient:
    return ApiClient(service="test", base_url="https://api.example.org", transport=httpx.MockTransport(handler),
                     sleep=lambda seconds: None, cache=cache, **kwargs)


def test_a_kept_response_is_reused_whatever_the_key_and_traced_as_cached(recorder, settings):
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={"n": len(sent)})

    cache = ResponseCache(settings.responses_path, 30)
    with recorder.step("run_test", "search"):
        first = _client(handler, cache, params={"api_key": "secret-1"}).get("/search", params={"q": "lora"})
        again = _client(handler, cache, params={"api_key": "secret-2"}).get("/search", params={"q": "lora"})
        other = _client(handler, cache).get("/search", params={"q": "adapters"})

    assert first.json() == again.json() == {"n": 1} and other.json() == {"n": 2}
    assert len(sent) == 2  # the same search under another key was not sent again
    calls = [e for e in recorder.events("run_test") if e["activity_type"] == "api_call"]
    assert [c["detail"].get("cached", False) for c in calls] == [False, True, False]
    assert all("secret" not in url for (url,) in cache._conn.execute("SELECT url FROM responses"))


def test_only_successes_and_allowed_404s_are_kept_and_age_zero_fetches_fresh(settings):
    sent: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request.url.path)
        return httpx.Response({"/missing": 404, "/bad": 400}.get(request.url.path, 200), json={})

    client = _client(handler, ResponseCache(settings.responses_path, 30))
    assert client.get("/missing", allow_404=True) is None and client.get("/missing", allow_404=True) is None
    for _ in range(2):
        with pytest.raises(httpx.HTTPStatusError):
            client.get("/bad")
    assert sent == ["/missing", "/bad", "/bad"]

    fresh = _client(handler, ResponseCache(settings.responses_path, 0))
    fresh.get("/ok")
    fresh.get("/ok")
    assert sent[-2:] == ["/ok", "/ok"]
    assert client.get("/ok").status_code == 200 and sent[-1] == "/ok" and len(sent) == 5  # kept by the fresh fetches


def test_a_rerun_of_a_question_reads_the_same_papers_without_searching_again(small_corpus_settings, tmp_path):  # noqa: F811
    settings = small_corpus_settings
    router, model = _router(), FakeModel()
    services = fake_services(settings, tmp_path, router, model=model, cache=True)
    first = start_run(settings, QUESTION, services=services)
    searches = [r for r in router.requests if r.url.host != "arxiv.org"]
    asked = [c["output"] for c in model.calls]
    # Meanwhile the store learns more about the candidates, as when a later search fetches fresher
    # records: the rerun must still rank them by what its own (kept) searches said.
    with Database(settings.db_path) as db:
        ranked = [row[0] for row in db.query("SELECT paper_id FROM run_papers WHERE run_id = ? ORDER BY rank",
                                             (first.run_id,))]
        db.execute("UPDATE papers SET citation_count = 100000, abstract = ? WHERE paper_id = ?",
                   ("Low-rank adaptation for parameter-efficient fine-tuning of large language models.", ranked[-1]))
        db.execute("UPDATE papers SET citation_count = 0 WHERE paper_id = ?", (ranked[0],))
    second = start_run(settings, QUESTION, services=services)
    services.sources.close()

    assert [r for r in router.requests if r.url.host != "arxiv.org"] == searches  # nothing searched again
    corpus = [[(r["paper_id"], r["rank"], r["read_level"]) for r in _corpus(settings, run.run_id)[0]] for run in (first, second)]
    assert corpus[0] == corpus[1] and corpus[0]
    # The brief and the relevance labels were kept, so the rerun asked the models nothing for them.
    later = [c["output"] for c in model.calls[len(asked):]]
    assert "BriefDraft" in asked and "Judgements" in asked
    assert "BriefDraft" not in later and "Judgements" not in later
    # The full text was fetched again (texts aren't kept as responses) but, unchanged, not stored again.
    with Database(settings.db_path) as db:
        assert db.query("SELECT MAX(text_no) FROM paper_texts")[0][0] == 1
        before, after = run_cost(db, first.run_id), run_cost(db, second.run_id)

    # The cost of each run, from its trace: the rerun reused what the first run produced.
    assert before.reuse_of(("brief",)).asked == 1 and after.reuse_of(("brief",)).reused == 1
    assert after.reuse_of(("relevance_check",)).asked == 0 < after.reuse_of(("relevance_check",)).reused
    searched = {a.service: a for a in after.apis}
    assert searched["semantic_scholar"].requests == 0 < searched["semantic_scholar"].cached
    assert searched["arxiv_web"].requests > 0  # full texts are fetched, not kept as responses
    assert [s.stage for s in after.stages][:3] == ["brief", "search", "expand"]
    assert {m.model for m in before.models} >= {"fake:reasoner", "fake:extractor"}
    report = (settings.runs_dir / second.run_id / "report.md").read_text(encoding="utf-8")
    assert "## Run cost" in report and "kept from earlier" in report

    # Asked to refresh, a run searches again.
    fresh = settings.model_copy(update={"response_cache_days": 0})
    services = fake_services(fresh, tmp_path, router, model=model, cache=True)
    start_run(fresh, QUESTION, services=services)
    services.sources.close()
    again = [r for r in router.requests if r.url.host != "arxiv.org"][len(searches):]
    assert {str(r.url) for r in again} == {str(r.url) for r in searches}
