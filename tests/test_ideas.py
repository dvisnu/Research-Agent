"""Phase 4: ideas, the novelty search, comparisons and novelty levels."""

import json
from types import SimpleNamespace

import pytest

from fakes import FakeTextEmbedder, Router, fake_services
from research_agent.graph.quotes import normalize
from research_agent.ideas.compare import check
from research_agent.ideas.ideas import IdeaDraft, IdeaStore, duplicates, select
from research_agent.ideas.levels import GapInfo, Verdict, assess
from research_agent.ideas.report import idea_reports
from research_agent.ideas.stages import MAX_ROUNDS, check_idea, run_phase4
from research_agent.literature.papers import PaperRecord, PaperStore
from research_agent.literature.parse import parse_arxiv_html
from research_agent.literature.stages import is_full_text
from research_agent.literature.texts import abstract_text, extend_text, store_text
from research_agent.pipeline.graph import RunOptions
from research_agent.report import RunReport, to_markdown
from test_gaps import _phase3
from test_graph import PaperModel, corpus  # noqa: F401  (corpus is a fixture)
from test_parse import HTML

CLOSE = "Merging low-rank adapters across tasks"


# The level rules --------------------------------------------------------------------------

def _v(position: int, purpose: str = "different", mechanism: str = "different", *, solves: str = "no",
       ok: bool = True, year: int = 2024) -> Verdict:
    return Verdict(f"p{position}", f"Paper {position}", year, position, ok, purpose, mechanism, "different", solves)


def _others(start: int = 2, count: int = 9) -> list[Verdict]:
    return [_v(n) for n in range(start, start + count)]


STATED = GapInfo("stated", papers=2, newest=2023)


def test_a_paper_with_the_same_purpose_and_mechanism_makes_a_duplicate():
    result = assess([_v(1), _v(2, "same", "same"), *_others(3)], STATED, [])
    assert result.level == 1 and result.deciding == ["p2"] and not result.inconclusive


@pytest.mark.parametrize(("purpose", "mechanism", "level"), [
    ("similar", "same", 2), ("same", "similar", 2), ("similar", "similar", 2),
    ("different", "same", 4),  # a known technique for a new purpose: no variation, but no new mechanism either
    ("unknown", "same", 4),
    ("same", "different", 5),
])
def test_close_purpose_and_mechanism_make_a_minor_variation(purpose, mechanism, level):
    assert assess([_v(1, purpose, mechanism), *_others()], STATED, []).level == level


def test_a_stated_gap_nobody_solves_gives_4_and_a_widely_raised_one_5():
    assert assess([_v(1), *_others()], STATED, []).level == 5
    assert assess([_v(1), *_others()], GapInfo("stated", papers=1), []).level == 4
    assert assess([_v(1, mechanism="unknown"), *_others()], STATED, []).level == 4  # unstated blocks 5
    assert assess([_v(1), *_others()], GapInfo("conflict", papers=2), []).level == 5


def test_an_unstated_gap_stops_at_3():
    assert assess([_v(1), *_others()], GapInfo("structural", papers=3), []).level == 3
    assert assess([_v(1), *_others()], GapInfo(None), []).level == 3


def test_a_paper_that_solves_the_gap_caps_at_3_and_is_named():
    result = assess([_v(1, solves="yes", year=2025), *_others()], STATED, [])
    assert result.level == 3 and result.addressed_by == ["p1"]
    assert "possibly addressed by Paper 1 (2025), newer than the gap's evidence (2023)" in result.reasons[-1]
    assert assess([_v(1, solves="partly"), *_others()], STATED, []).level == 5


def test_weak_coverage_and_failed_comparisons_cap_at_3_as_inconclusive():
    weak = assess([_v(1), *_others()], STATED, ["only 12 candidate papers found"])
    assert (weak.level, weak.inconclusive) == (3, True) and "only 12 candidate papers found" in weak.caps
    failed = assess([_v(1), _v(2, ok=False), *_others(3)], STATED, [])
    assert (failed.level, failed.inconclusive) == (3, True)
    late_failure = assess([_v(1), *_others(2, 3), _v(5, ok=False), *_others(6, 5)], STATED, [])
    assert (late_failure.level, late_failure.inconclusive) == (5, False)  # not one of the 3 closest
    few = assess([_v(1), *_others(2, 3), *[_v(n, ok=False) for n in range(5, 11)]], STATED, [])
    assert few.inconclusive and "only 4 closest papers could be compared" in few.caps


def test_a_close_match_stays_conclusive_whatever_the_coverage():
    result = assess([_v(1, "same", "same")], STATED, ["no Semantic Scholar key"])
    assert (result.level, result.inconclusive) == (1, False)


def test_comparisons_fail_closed():
    shown = {"p:s0:0": "We retrieve passages with dense retrievers.", "p:s0:1": "We test on HotpotQA."}
    reply = {
        "purpose": {"match": "different", "paper": "x", "evidence": [{"sentence": "0", "quote": "words it never wrote"}]},
        "mechanism": {"match": "same", "paper": "y", "evidence": []},
        "evaluation": {"match": "different", "paper": "z", "evidence": [{"sentence": "0", "quote": "We test on HotpotQA"}]},
        "solves_gap": "no", "gap_evidence": [], "difference": "d",
    }
    out = check(reply, shown, {"0": "p:s0:0", "1": "p:s0:1"}.get)
    assert out["purpose"]["effective"] == "similar"  # an unsupported "different" counts as similar
    assert out["mechanism"]["effective"] == "same" and not out["mechanism"]["verified"]
    assert out["evaluation"]["evidence"] == [{"sentence_id": "p:s0:1", "quote": "We test on HotpotQA"}]  # readdressed
    assert out["quote_check"] == {"quotes": 2, "passed": 1}


# Picking ideas -----------------------------------------------------------------------------

def _draft(gap: int, title: str) -> IdeaDraft:
    return IdeaDraft(gap=gap, kind="fix_limitation", title=title, purpose="p", mechanism="m", evaluation="e",
                     builds_on=[], feasibility="f")


def test_ideas_are_picked_one_per_gap_first():
    drafts = [_draft(1, "a"), _draft(1, "b"), _draft(2, "c"), _draft(9, "bad"), _draft(3, "d")]
    chosen, dropped = select(drafts, 3, [None] * 5)
    assert chosen == [0, 2, 4] and [d["draft"] for d in dropped] == [3]
    chosen, _ = select(drafts, 3, [None, None, None, None, 0])  # d repeats a: gap 1's second idea fills in
    assert chosen == [0, 2, 1]


def test_near_duplicates_are_found_among_new_ideas_and_rejected_ones(db):
    texts = ["low rank adapters for vision", "graph neural networks for molecules", "adapters for vision low rank"]
    found, _ = duplicates(db, FakeTextEmbedder(), texts, ["Graph neural networks for molecules!"])
    assert found == [None, 0, 1]  # the second repeats the rejected idea, the third the first new one


# Reading a closest paper in full ------------------------------------------------------------

def test_extending_an_abstract_keeps_its_sentences_and_their_evidence(db):
    abstract = "We study routing. It works well."
    paper_id = PaperStore(db).upsert(PaperRecord(source="arxiv", title="Routing", abstract=abstract, arxiv_id="2401.00001"))
    store_text(db, paper_id, abstract_text("Routing", abstract), source="abstract", parser="abstract")
    db.execute("INSERT INTO evidence (owner_kind, owner_id, sentence_id, quote) VALUES ('node', 'n1', ?, 'We study routing')",
               (f"{paper_id}:s0:0",))
    added = extend_text(db, paper_id, parse_arxiv_html(HTML), source="arxiv_html", parser="test")
    headings = [r["heading"] for r in db.query("SELECT heading FROM sections WHERE paper_id = ? ORDER BY idx", (paper_id,))]
    assert headings == ["Abstract", "1 Introduction", "1.1 Setup"] and added == 5
    assert db.query("SELECT COUNT(*) AS n FROM evidence WHERE owner_id = 'n1'")[0]["n"] == 1
    source = db.query("SELECT text_source FROM papers WHERE paper_id = ?", (paper_id,))[0]["text_source"]
    assert source == "abstract+arxiv_html" and is_full_text(source)


# Phase 4 end to end ---------------------------------------------------------------------------

def _s2(paper_id: str, title: str, arxiv: str | None, abstract: str) -> dict:
    return {"paperId": paper_id, "title": title, "abstract": abstract,
            "externalIds": {"ArXiv": arxiv} if arxiv else {}, "authors": [], "year": 2024}


def _router() -> Router:
    router = Router()
    router.add("api.semanticscholar.org", "/graph/v1/paper/search", {"data": [
        _s2("s2close", CLOSE, "2401.00001", "We merge several LoRA modules so one model serves many tasks. "
                                           "Merged adapters match per-task accuracy on GLUE."),
        _s2("s2sparse", "Sparse fine-tuning of language models", None,
            "Sparse updates fine-tune a small subset of weights. They work on GLUE."),
    ]})
    router.add("api.semanticscholar.org", "/graph/v1/snippet/search", {"data": [
        {"snippet": {"text": "Task vectors add up.", "section": "Method"}, "score": 0.9,
         "paper": {"corpusId": 7, "title": "Editing models with task arithmetic"}},
    ]})
    router.add("api.semanticscholar.org", "/graph/v1/paper/batch", [
        _s2("s2arith", "Editing models with task arithmetic", None,
            "Task vectors add and subtract fine-tuned weights. Adding task vectors builds multi-task models."),
    ])
    router.add("api.semanticscholar.org", "/recommendations/v1/papers/forpaper/", {"recommendedPapers": [
        _s2("s2fusion", "AdapterFusion: combining adapters for transfer", None, "AdapterFusion combines adapters. It helps transfer."),
    ]})
    router.add("api.openalex.org", "/works", {"results": [
        {"id": "https://openalex.org/W9", "display_name": "Model merging for multi-task learning", "publication_year": 2023},
    ]})
    router.add("arxiv.org", "/html/2401.00001", HTML)
    return router


def _ctx(db, recorder, settings, tmp_path, model, *, s2: bool = True, options: RunOptions | None = None):
    services = fake_services(settings, tmp_path, _router(), s2=s2, model=model)
    return SimpleNamespace(db=db, settings=settings, recorder=recorder, llm=services.llm, embedder=services.embedder,
                           text_embedder=services.text_embedder, sources=services.sources, options=options or RunOptions())


def _phase4(db, recorder, settings, tmp_path, model, **options):
    state = _phase3(db, recorder, settings, model)
    ctx = _ctx(db, recorder, settings, tmp_path, model, **options)
    return run_phase4(ctx, state), ctx


def _evidence_resolves(db) -> bool:
    rows = db.query("SELECT e.quote, s.text FROM evidence e LEFT JOIN sentences s USING (sentence_id) WHERE e.owner_kind = 'comparison'")
    return bool(rows) and all(r["text"] and normalize(r["quote"]).strip(" .\"'") in normalize(r["text"]) for r in rows)


def test_ideas_are_written_searched_compared_and_rated(db, recorder, settings, tmp_path, corpus):  # noqa: F811
    model = PaperModel()
    state, _ = _phase4(db, recorder, settings, tmp_path, model)

    ideas = IdeaStore(db).for_run("run_test")
    checked = [i for i in ideas if i.status != "unchecked"]
    assert checked and all(i.status == "assessed" and i.level == 3 for i in checked)  # few candidates: weak coverage
    assert all(i.assessment["inconclusive"] and i.assessment["coverage"]["weak"] for i in checked)
    assert {i.gap_node_id for i in checked} <= {n["node_id"] for n in db.query("SELECT node_id FROM nodes WHERE node_type = 'gap'")}
    assert state["novelty_round"] == 1 and not state["novelty_retry"]

    # The search went past the corpus, through every channel, and kept the closest papers.
    candidates = db.query("SELECT c.found_by, c.position, p.title FROM idea_candidates c JOIN papers p USING (paper_id)"
                          " WHERE c.idea_id = ?", (checked[0].idea_id,))
    channels = {f["channel"] for row in candidates for f in json.loads(row["found_by"])}
    assert {"s2_search", "s2_title", "s2_snippets", "openalex_search", "corpus", "s2_recommendations"} <= channels
    assert CLOSE in {row["title"] for row in candidates if row["position"]}

    # Every closest paper was compared, the 3 closest in full when their text could be had, and
    # every quote resolves to a real sentence.
    comparisons = db.query("SELECT c.depth, p.title FROM comparisons c JOIN papers p USING (paper_id) WHERE c.idea_id = ?",
                           (checked[0].idea_id,))
    assert len(comparisons) == len([r for r in candidates if r["position"]])
    assert ("full", CLOSE) in {(c["depth"], c["title"]) for c in comparisons}  # its body came from arXiv HTML
    assert _evidence_resolves(db)

    # The report: the best idea fills the Novel Evidence Report, every checked idea is listed.
    reports = idea_reports(db, "run_test")
    assert [r.rank for r in reports] == list(range(1, len(checked) + 1))
    report = RunReport(run_id="run_test", question="q", generated_at="now")
    report.add_phase4(reports)
    markdown = to_markdown(report)
    assert "Potential novelty: Moderate (level 3 of 5, Combination, inconclusive)" in markdown
    assert "## Ideas checked" in markdown and "Search coverage:" in markdown
    assert report.novel_evidence_report.closest_existing_work


def test_an_idea_rated_1_or_2_is_revised_once_per_round_up_to_the_limit(db, recorder, settings, tmp_path, corpus):  # noqa: F811
    model = PaperModel()
    original = model.extract

    def extract(prompt: str, output: str) -> dict | None:
        if output == "PaperComparison" and f"Paper: {CLOSE}" in prompt and "Title: Idea for gap" in prompt:
            facet = {"match": "same", "paper": "It merges LoRA modules.", "evidence": [{"sentence": "0", "quote": "We merge several LoRA modules"}]}
            return {"purpose": facet, "mechanism": facet, "evaluation": facet, "solves_gap": "yes",
                    "gap_evidence": [], "difference": "none"}
        return original(prompt, output)

    model.extract = extract
    state, _ = _phase4(db, recorder, settings, tmp_path, model)

    ideas = IdeaStore(db).for_run("run_test")
    first = [i for i in ideas if i.round == 1 and i.status != "unchecked"]
    second = [i for i in ideas if i.round == 2]
    assert first and all(i.status == "rejected" and i.level == 1 for i in first)
    assert len(second) == len(first) and all(i.origin == "revised" and i.level == 3 for i in second)
    parent = {i.idea_id: i for i in first}
    for idea in second:  # one facet changed, the other two kept
        assert idea.mechanism.startswith("Revised mechanism") and idea.purpose == parent[idea.parent_id].purpose
        assert idea.lineage == parent[idea.parent_id].lineage and idea.detail["changed"] == "mechanism"
    assert state["novelty_round"] == 2
    revision_prompt = next(c["prompt"] for c in model.calls if c["output"] == "Revisions")
    assert "It merges LoRA modules." in revision_prompt  # the paper that matched is shown

    reports = idea_reports(db, "run_test")
    assert all(r.title.startswith("Revised idea") for r in reports)
    assert all(r.history and "changed its mechanism" in r.history[0] for r in reports)


def test_the_loop_stops_after_the_last_round(db, recorder, settings, tmp_path, corpus):  # noqa: F811
    model = PaperModel()
    original = model.extract

    def extract(prompt: str, output: str) -> dict | None:
        if output == "PaperComparison" and f"Paper: {CLOSE}" in prompt:
            facet = {"match": "same", "paper": "x", "evidence": [{"sentence": "0", "quote": "We merge several LoRA modules"}]}
            return {"purpose": facet, "mechanism": facet, "evaluation": facet, "solves_gap": "no",
                    "gap_evidence": [], "difference": "none"}
        return original(prompt, output)

    model.extract = extract
    state, _ = _phase4(db, recorder, settings, tmp_path, model)
    assert state["novelty_round"] == MAX_ROUNDS and not state["novelty_retry"]
    ideas = IdeaStore(db).for_run("run_test")
    assert sum(1 for c in model.calls if c["output"] == "Revisions") == MAX_ROUNDS - 1
    # The fake model revises the same way twice: the last round's versions repeat rejected ones,
    # so they are rejected unchecked, and each lineage's best checked version is reported.
    last = [i for i in ideas if i.round == MAX_ROUNDS]
    assert last and all(i.status == "rejected" and i.level is None and i.detail["rejected_because"] for i in last)
    ranked = [i for i in ideas if i.final_rank]
    assert ranked and all(i.level == 1 and i.round == MAX_ROUNDS - 1 for i in ranked)


def test_a_given_idea_is_checked_on_its_own(db, recorder, settings, tmp_path):
    ctx = _ctx(db, recorder, settings, tmp_path, PaperModel(), s2=False)
    facets = {"title": "Merge LoRA modules for many tasks", "purpose": "Serve many tasks with one model.",
              "mechanism": "Merge low-rank adapters.", "evaluation": "GLUE accuracy."}
    idea = check_idea(ctx, "run_test", "planted", facets=facets, detail={"source": "test"})
    assert idea.status == "assessed" and idea.level == 3 and idea.final_rank == 1
    weak = idea.assessment["coverage"]["weak"]
    assert any("no Semantic Scholar key" in w for w in weak) and idea.assessment["inconclusive"]
    steps = [e["step"] for e in recorder.events("run_test") if e["activity_type"] == "step"]
    assert steps == ["ideate", "novelty_search", "novelty_assess"]
