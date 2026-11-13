"""Phase 6: a run that stops partway through any stage (a used-up quota, say) resumes to
the result an uninterrupted run gets: no stage leaves duplicates or half-written state
behind when it runs again."""

import json
import re
import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest

from fakes import fake_services
from research_agent.config import Settings
from research_agent.literature.papers import PaperRecord, PaperStore
from research_agent.literature.parse import ParsedSection, ParsedText
from research_agent.literature.texts import abstract_text, store_text
from research_agent.llm import LLMUnavailable
from research_agent.pipeline import RunError, resume_run, start_run
from research_agent.storage import Database, ids
from research_agent.trace import TraceRecorder
from test_graph import ALPHA, BETA, GAMMA, PaperModel
from test_ideas import _router as novelty_router
from test_ideas import _s2

QUESTION = "Does low-rank adaptation match full fine-tuning on GLUE?"
LIBRARY = [  # the papers the literature search finds, as in test_graph's corpus
    (ALPHA, "s2alpha", [
        ParsedSection("Abstract", ["We propose LoRA, a low-rank adaptation method. LoRA reaches 90% accuracy on GLUE."]),
        ParsedSection("Introduction", ["Fine-tuning large models is expensive. LoRA freezes the pretrained weights."]),
        ParsedSection("Limitations", ["LoRA cannot merge several tasks at once."]),
        ParsedSection("Appendix A Extra results", ["This appendix is never read."]),
    ], "Low-rank adaptation of large language models."),
    (BETA, "s2beta", [
        ParsedSection("Abstract", ["We study Low-Rank Adaptation on the GLUE benchmark. "
                                   "Low-Rank Adaptation reaches 85% accuracy on GLUE."]),
    ], "Low-rank adaptation on GLUE."),
    (GAMMA, "s2gamma", None, "Adapters add small trainable layers. They match full fine-tuning."),
]


class FlakyModel:
    """PaperModel, except that its quota runs out once, on call number ``fail_at``."""

    def __init__(self, fail_at: int | None = None) -> None:
        self.inner, self.fail_at, self.count = PaperModel(), fail_at, 0

    def complete(self, **kwargs: Any) -> Any:
        self.count += 1
        if self.count == self.fail_at:
            raise LLMUnavailable("daily quota used up (test)", lasting=True)
        return self.inner.complete(**kwargs)


class _CountingIds:
    """IDs in creation order. Two runs in two databases then order what they sort by ID
    alike; real IDs are random per database (and fixed within it)."""

    def __init__(self) -> None:
        self.n = 0

    def uuid4(self) -> uuid.UUID:
        self.n += 1
        return uuid.UUID(int=self.n << 64)


def _router():
    """The literature search finds the library; the novelty and targeted searches find test_ideas' papers."""
    router = novelty_router()
    library = {"data": [_s2(s2_id, title, None, abstract) for title, s2_id, _, abstract in LIBRARY]}
    s2_ids = [s2_id for _, s2_id, _, _ in LIBRARY]

    def literature(request: httpx.Request) -> bool:
        asked = " ".join(request.url.params.values()).lower()
        return "adaptation" in asked or any(s2_id in request.url.path for s2_id in s2_ids)

    def either(path: str, found: Any, empty: Any):
        others = next(handler for host, prefix, handler in router.routes if prefix == path)
        router.add(host_of[path], path, lambda r: httpx.Response(200, json=found) if literature(r) else others(r)) if found \
            else router.add(host_of[path], path, lambda r: httpx.Response(200, json=empty) if literature(r) else others(r))

    host_of = {"/graph/v1/paper/search": "api.semanticscholar.org", "/graph/v1/snippet/search": "api.semanticscholar.org",
               "/recommendations/v1/papers/forpaper/": "api.semanticscholar.org", "/works": "api.openalex.org"}
    either("/graph/v1/paper/search", library, None)
    either("/graph/v1/snippet/search", None, {"data": []})
    either("/recommendations/v1/papers/forpaper/", None, {"recommendedPapers": []})
    either("/works", None, {"results": []})
    return router


def _settings(tmp_path: Path) -> Settings:
    settings = Settings(_env_file=None, data_dir=tmp_path / "var", full_read=3, abstract_only=0, reuse_texts=True)
    with Database(settings.db_path) as db:  # the library's texts are already stored, as from earlier runs
        for title, _, sections, abstract in LIBRARY:
            paper_id = PaperStore(db).upsert(PaperRecord(source="arxiv", title=title, abstract=abstract, year=2021))
            parsed = ParsedText(title, sections) if sections else abstract_text(title, abstract)
            store_text(db, paper_id, parsed, source="arxiv_html" if sections else "abstract", parser="test")
    return settings


_ID = re.compile(r"\b[a-z]+_[0-9a-f]{16}\b")
_COUNTS = {
    "papers": "SELECT read_level, COUNT(*) FROM run_papers WHERE run_id = ? GROUP BY read_level",
    "extractions": "SELECT COUNT(*), 'n' FROM run_extractions WHERE run_id = ?",
    "nodes": "SELECT node_type, COUNT(*) FROM nodes WHERE run_id = ? GROUP BY node_type",
    "edges": "SELECT edge_type, COUNT(*) FROM edges WHERE run_id = ? GROUP BY edge_type",
    "ideas": "SELECT status, COUNT(*) FROM ideas WHERE run_id = ? GROUP BY status",
    "candidates": "SELECT COUNT(*), 'n' FROM idea_candidates JOIN ideas USING (idea_id) WHERE run_id = ?",
    "comparisons": "SELECT COUNT(*), 'n' FROM comparisons JOIN ideas USING (idea_id) WHERE run_id = ?",
    "plan_papers": "SELECT role, COUNT(*) FROM plan_papers WHERE run_id = ? GROUP BY role",
    "plans": "SELECT status, COUNT(*) FROM plans WHERE run_id = ? GROUP BY status",
    "elements": "SELECT verdict, COUNT(*) FROM plan_elements JOIN plans USING (plan_id) WHERE run_id = ? GROUP BY verdict",
    "evidence": "SELECT owner_kind, COUNT(*) FROM evidence WHERE owner_id IN (SELECT node_id FROM nodes WHERE run_id = ?1"
                " UNION SELECT edge_id FROM edges WHERE run_id = ?1 UNION SELECT element_id FROM plan_elements"
                " JOIN plans USING (plan_id) WHERE run_id = ?1 UNION SELECT comparison_id FROM comparisons"
                " JOIN ideas USING (idea_id) WHERE run_id = ?1) GROUP BY owner_kind",
}


def _outcome(settings: Settings, run_id: str) -> dict[str, Any]:
    """What a run produced, without its IDs, timestamps and call counts (a resumed run reuses replies)."""
    report = json.loads((settings.runs_dir / run_id / "report.json").read_text(encoding="utf-8"))
    report.pop("generated_at")
    report.pop("cost")
    graph = report["knowledge_graph"]
    for key in ("extraction_calls", "extractions_reused"):
        graph.pop(key)
    graph["links"].pop("embedding", None)
    names: dict[str, str] = {}
    text = _ID.sub(lambda m: names.setdefault(m[0], f"ID{len(names)}"), json.dumps(report, sort_keys=True))
    with Database(settings.db_path) as db:
        counts = {name: sorted((str(a), str(b)) for a, b in db.query(sql.replace("?1", "?").replace("?", "?1"), (run_id,)))
                  for name, sql in _COUNTS.items()}
    return {"report": json.loads(text), "counts": counts}


def _run(tmp_path: Path, fail_at: int | None) -> tuple[Settings, str, list[str]]:
    """Run the pipeline, and resume it once if the quota ran out; returns the steps of its LLM calls."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ids, "uuid", _CountingIds())
        settings, model = _settings(tmp_path), FlakyModel(fail_at)
        services = fake_services(settings, tmp_path, _router(), model=model)
        try:
            run_id = start_run(settings, QUESTION, services=services).run_id
            assert fail_at is None or model.count < fail_at, "the failure did not happen"
        except RunError as exc:
            run_id = exc.run_id
            services.sources.close()
            services = fake_services(settings, tmp_path, _router(), model=model)  # a new session: the quota is back
            resume_run(settings, run_id, services)
        services.sources.close()
    with Database(settings.db_path) as db:
        steps = [e["step"] for e in TraceRecorder(db).events(run_id) if e["activity_type"] == "llm_call"]
    return settings, run_id, steps


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    settings, run_id, steps = _run(tmp_path_factory.mktemp("baseline"), None)
    return _outcome(settings, run_id), steps


def test_the_uninterrupted_run_reaches_a_verified_plan(baseline):
    outcome, steps = baseline
    counts = outcome["counts"]
    assert dict(counts["papers"]) == {"abstract": "1", "full": "2"}
    assert counts["nodes"] and dict(counts["nodes"]).get("gap") and counts["comparisons"][0][0] != "0"
    assert dict(counts["plans"]).get("final") == "1" and dict(counts["elements"]).get("supported")
    assert {"brief", "rank", "extract", "build_graph", "gaps", "ideate", "novelty_search", "novelty_assess",
            "plan", "critique", "verify"} <= set(steps)


def _failure_points(steps: list[str]) -> list[tuple[str, int]]:
    """For each stage that calls a model: its first call, and its last one."""
    points = []
    for step in dict.fromkeys(steps):
        calls = [n for n, s in enumerate(steps, start=1) if s == step]
        points += [(step, n) for n in dict.fromkeys((calls[0], calls[-1]))]
    return points


def test_a_run_stopped_at_any_stage_resumes_to_the_same_result(baseline, tmp_path):
    expected, steps = baseline
    different = []
    for step, call in _failure_points(steps):
        settings, run_id, _ = _run(tmp_path / f"call{call}", call)
        got = _outcome(settings, run_id)
        if got != expected:
            diff = [key for key in ("counts", "report") if got[key] != expected[key]]
            different.append(f"stopped at call {call} ({step}): {diff}")
    assert not different, "\n".join(different)
