"""Phase 5: the fact menu, the plan and its provenance, the critique loop, verification, PROV."""

import json
import re
from types import SimpleNamespace

import httpx
from typer.testing import CliRunner

from fakes import Router, fake_services
from research_agent.cli import app
from research_agent.graph.quotes import normalize
from research_agent.ideas.stages import run_phase4
from research_agent.pipeline.graph import RunOptions
from research_agent.plans.critique import code_checks
from research_agent.plans.facts import Fact, Menu, PlanPaper, build_menu, load_plan_papers
from research_agent.plans.prov import provenance_document, trace_chains
from research_agent.plans.report import plan_report
from research_agent.plans.stages import best_idea, run_phase5
from research_agent.plans.store import Element, Plan, PlanStore
from research_agent.plans.verify import coverage, verify_elements
from research_agent.report import RunReport, to_markdown
from test_gaps import _phase3
from test_graph import PaperModel, corpus  # noqa: F401  (corpus is a fixture)
from test_ideas import _ctx, _router, _s2
from test_parse import HTML

FRESH = "Routing tokens to experts on GLUE"


def _plan(db, recorder, settings, tmp_path, model, router: Router | None = None):
    """Phases 3, 4 and 5 on the test corpus."""
    state = _phase3(db, recorder, settings, model)
    services = fake_services(settings, tmp_path, router or _router(), model=model)
    ctx = SimpleNamespace(db=db, settings=settings, recorder=recorder, llm=services.llm, embedder=services.embedder,
                          text_embedder=services.text_embedder, sources=services.sources, options=RunOptions())
    state = run_phase4(ctx, state)
    return run_phase5(ctx, state), ctx


def _resolves(db, owner_kind: str) -> bool:
    rows = db.query("SELECT e.quote, s.text FROM evidence e LEFT JOIN sentences s USING (sentence_id) WHERE e.owner_kind = ?",
                    (owner_kind,))
    return bool(rows) and all(r["text"] and normalize(r["quote"]).strip(" .\"'") in normalize(r["text"]) for r in rows)


# The fact menu ----------------------------------------------------------------------------------

def test_the_menu_merges_names_ranks_what_the_idea_names_and_keeps_evidence(db, recorder, settings, tmp_path, corpus):  # noqa: F811
    _plan(db, recorder, settings, tmp_path, PaperModel())
    plan = PlanStore(db).latest("run_test")
    menu = Menu.from_json(plan.menu, load_plan_papers(db, "run_test"))
    kinds = [f.kind for f in menu.facts.values()]
    assert kinds == sorted(kinds, key=["gap", "approach", "dataset", "method", "metric", "result", "limitation", "finding"].index)
    assert menu.facts["F1"].kind == "gap" and menu.facts["F1"].evidence

    glue = [f for f in menu.of_kind("dataset") if f.text in ("GLUE", "GLUE benchmark")]
    assert len(glue) == 1 and glue[0].attrs["named"]  # one entity for both spellings; the idea names it
    assert len(glue[0].paper_ids) >= 2 and glue[0].attrs["keys"] == ["glue"]
    lora = next(f for f in menu.of_kind("method") if f.text == "LoRA")  # Low-Rank Adaptation merged into it
    assert lora.attrs["proposed_by"] and lora.attrs["named"] and menu.of_kind("method")[0] is lora
    assert any("90% accuracy on GLUE" in f.text and f.attrs["dataset"] for f in menu.of_kind("result"))
    assert any("cannot merge several tasks" in f.text for f in menu.of_kind("limitation"))
    assert all(f.evidence and f.origin.get("from") for f in menu.facts.values())
    # The idea's closest papers are plan papers, extracted for the plan when the run had not.
    roles = {role for p in menu.papers.values() for role in p.roles}
    assert {"closest", "gap", "related"} <= roles
    extracted = [p for p in menu.papers.values() if p.roles.get("closest", {}).get("extracted") == "for the plan"]
    assert extracted and any(p.extractions() for p in extracted)


def test_a_revision_keeps_every_fact_under_its_number(db, recorder, settings, tmp_path, corpus):  # noqa: F811
    _, ctx = _plan(db, recorder, settings, tmp_path, PaperModel())
    plan = PlanStore(db).latest("run_test")
    old = Menu.from_json(plan.menu, load_plan_papers(db, "run_test"))
    idea = best_idea(ctx, "run_test")
    again = build_menu(db, "run_test", idea, None, [], previous=old, round_=2)
    assert all(again.facts[n].text == f.text for n, f in old.facts.items())


# Code checks ----------------------------------------------------------------------------------------

def _fact(fact_id: str, kind: str, text: str, paper: str, **attrs) -> Fact:
    return Fact(fact_id, kind, f"{kind}:{text}", text, [paper], [{"sentence_id": f"{paper}:s0:0", "quote": text}],
                {"from": "test"}, attrs)


def _menu(*facts: Fact) -> Menu:
    papers = {p: PlanPaper(p, f"Paper {p}", 2024, {"closest": {"position": 1}}) for f in facts for p in f.paper_ids}
    return Menu({f.fact_id: f for f in facts}, papers)


def _element(field: str, cites: list[str], name: str | None = None, text: str = "x", position: int = 1) -> Element:
    return Element(field, position, text, name=name, cites=cites, derivation={"checks": []})


def test_code_checks_ask_for_the_closest_work_and_a_reference_result():
    menu = _menu(
        _fact("F1", "approach", "It merges adapters.", "p1", position=1, purpose="similar", mechanism="different"),
        _fact("F2", "approach", "It prunes weights.", "p2", position=2, purpose="different", mechanism="different"),
        _fact("F3", "method", "LoRA", "p3", proposed_by=["p3"], keys=["lora"]),
        _fact("F4", "dataset", "GLUE", "p3", keys=["glue"]),
        _fact("F5", "result", "Adapters reach 80 on GLUE.", "p4", method="Adapters", dataset="GLUE", metric="accuracy"),
    )
    plan = Plan("plan_x", "run_x", "i_x", 1, elements=[
        _element("baselines", ["F3"], "LoRA"), _element("datasets", ["F4"], "GLUE")])
    points = code_checks(plan, menu)
    assert [p["kind"] for p in points] == ["missing_baseline", "no_reference_result"]
    assert "Paper p1 (2024)" in points[0]["problem"] and not points[0]["search"]  # its approach is listed: no search
    assert points[1]["search"] == "LoRA GLUE results"

    menu.facts["F6"] = _fact("F6", "result", "LoRA reaches 90 on GLUE.", "p3", method="LoRA", dataset="GLUE")
    plan.elements.append(_element("baselines", ["F1"], "Adapter merging", position=2))
    assert code_checks(plan, menu) == []


def test_an_entity_not_from_the_literature_is_flagged_for_a_search():
    menu = _menu(_fact("F1", "dataset", "GLUE", "p1", keys=["glue"]))
    element = _element("datasets", [], "SuperGLUE")
    element.derivation["checks"] = ["cites no listed dataset: not from the literature", "cites no fact"]
    points = code_checks(Plan("plan_x", "run_x", "i_x", 1, elements=[element]), menu)
    assert points[0]["kind"] == "unsupported" and points[0]["search"] == "SuperGLUE"


# Verification ---------------------------------------------------------------------------------------

class _NoCalls:
    def models(self, role):
        return ["fake:verifier"]

    def answer(self, *args, **kwargs):
        raise AssertionError("no call expected")


def test_elements_without_a_source_are_flagged_without_a_call(db):
    elements = [_element("method", [], text="A new step."), _element("hypothesis", ["F9"], text="h")]
    stats = verify_elements(db, _NoCalls(), None, elements)
    assert stats["claims"] == 0 and [e.verdict for e in elements] == ["no_source", "no_source"]
    assert all(e.status == "flagged" for e in elements)


def test_coverage_counts_supported_elements_of_the_final_plan():
    elements = [_element("datasets", ["F1"], position=n) for n in range(1, 5)]
    for element, verdict, status in zip(elements, ("supported", "supported", "not_supported", "contradicted"),
                                        ("kept", "kept", "flagged", "removed")):
        element.verdict, element.status = verdict, status
    result = coverage(elements)
    assert (result["elements"], result["supported"], result["coverage"]) == (3, 2, 0.6667)
    assert (result["flagged"], result["removed"]) == (1, 1)


# End to end -------------------------------------------------------------------------------------------

def test_the_best_idea_gets_a_verified_plan_whose_every_element_traces_to_a_paper(db, recorder, settings, tmp_path, corpus):  # noqa: F811
    model = PaperModel()
    state, ctx = _plan(db, recorder, settings, tmp_path, model)
    versions = PlanStore(db).versions("run_test")
    assert len(versions) == 1 and versions[0].status == "final"  # the critique made no point: no revision
    plan = versions[0]
    assert plan.idea_id == best_idea(ctx, "run_test").idea_id
    assert state["plan"]["coverage"] == plan.detail["coverage"]["coverage"]

    # Each element has its three provenance records, and its evidence is real.
    for element in plan.elements:
        assert element.derivation["step"] == "plan" and element.derivation["model"] == "fake:reasoner"
        assert element.decision["reason"] == "fake decision"
        assert element.verdict in ("supported", "no_source")
        assert bool(element.evidence) == (element.verdict == "supported")
    datasets = plan.field_elements("datasets")
    assert datasets and datasets[0].decision["alternatives"] is not None
    assert all(f["origin"]["from"] for e in plan.elements for f in e.derivation["facts"])
    assert _resolves(db, "plan_element")
    c = plan.detail["coverage"]
    assert c["elements"] == len(plan.elements) and c["supported"] == len(plan.elements) - 1  # the new method step
    assert c["labels"] == {"supported": c["supported"], "no_source": 1}

    # The verifier saw each claim with its full sentences, and the critique came from the verifier role.
    verify = next(call["prompt"] for call in model.calls if call["output"] == "Verdicts")
    assert "Statement about the literature: The sources say:" in verify and "Cited sentences:" in verify
    assert re.search(r"Cited facts:\n- F\d+ \(\w+\) .+ \[from .+: closest paper \d", verify)
    assert any(call["model"] == "verifier" for call in model.calls if call["output"] == "PlanCritique")

    # The PROV document walks from every cited element through a fact and a sentence to a section and a paper.
    document = json.loads(provenance_document(db, "run_test").serialize(format="json"))
    walked = trace_chains(document, plan.plan_id)
    assert len(walked) == len(plan.elements) and all(w["reasoning"] for w in walked)
    assert all(w["traceable"] for w in walked if w["status"] != "removed" and w["chains"])
    assert sum(1 for w in walked if not w["traceable"]) == 1  # only the uncited step

    # The report links the plan to the Novel Evidence Report and lists its sources.
    report = RunReport(run_id="run_test", question="q", generated_at="now")
    report.add_phase5(plan_report(db, "run_test"))
    markdown = to_markdown(report)
    assert f"`{plan.idea_id}`" in markdown and "**Provenance coverage:**" in markdown
    assert "### Sources" in markdown and "*(flagged, no source" in markdown
    assert report.experiment_plan.baselines and report.experiment_plan.baselines[0].sources


def test_the_critique_sends_a_search_and_the_revision_uses_what_it_found(db, recorder, settings, tmp_path, corpus):  # noqa: F811
    model = PaperModel()
    original = model.extract
    critiques = []

    def extract(prompt: str, output: str) -> dict | None:
        if output == "PlanCritique":
            critiques.append(prompt)
            point = {"element": "expected_results 1", "kind": "unsupported", "problem": "No result of routing on GLUE.",
                     "fix": "Find one.", "search": "routing experts GLUE results"}
            return {"answers_gap": "partly", "points": [point] if len(critiques) == 1 else []}
        if output == "OpeningChunkExtraction" and f"Paper: {FRESH}" in prompt:
            reply = {k: [] for k in ("tasks", "methods", "datasets", "metrics", "claims", "limitations", "assumptions",
                                     "future_work")}
            evidence = [{"sentence": "s0:1", "quote": "Routing reaches 92% accuracy on GLUE"}]
            reply["results"] = [{"text": "Routing reaches 92% accuracy on GLUE.", "evidence": evidence, "method": "Routing",
                                 "dataset": "GLUE", "metric": "accuracy", "conditions": None}]
            reply["facets"] = {"purpose": None, "mechanism": None, "evaluation": None}
            return reply
        return original(prompt, output)

    model.extract = extract
    fresh = _s2("s2fresh", FRESH, "2402.00002", "We route tokens to experts. Routing reaches 92% accuracy on GLUE.")
    state, _ = _plan(db, recorder, settings, tmp_path, model, _plan_router(fresh))

    versions = PlanStore(db).versions("run_test")
    assert [v.round for v in versions] == [1, 2] and [v.status for v in versions] == ["superseded", "final"]
    assert state["critique_round"] == 2 and len(critiques) == 2
    first = versions[0].critique
    assert first["searches"] == [{"query": "routing experts GLUE results", "point": first["points"][-1]["number"]}]
    assert first["answers_gap"] == "partly"

    # The search found a new paper, read the chunk holding its snippet, and its result reached the menu.
    row = db.query("SELECT r.detail FROM plan_papers r JOIN papers p USING (paper_id) WHERE r.role = 'search' AND p.title = ?",
                   (FRESH,))
    detail = json.loads(row[0]["detail"])
    assert detail["snippet"] and detail["chunk"] == 0 and detail["extractions"]
    menu = Menu.from_json(versions[1].menu, load_plan_papers(db, "run_test"))
    new = [f for f in menu.facts.values() if f.added == 2]
    assert any("92% accuracy" in f.text for f in new)

    # The revision answered the point with an element citing the new result.
    added = [e for e in versions[1].elements if e.derivation["change"] == "added"]
    answer = next(e for e in added if e.field == "expected_results")
    assert answer.decision["answers"][0]["point"] == first["points"][-1]["number"]
    assert any(menu.get(c).added == 2 for c in answer.cites)
    assert versions[1].detail["answers"] == [{"point": 1, "action": "fixed", "note": "fake"}]
    assert all(e.derivation["change"] == "kept" for e in versions[1].elements if e not in added)

    # PROV: the second version is a revision of the first, made after its critique.
    document = json.loads(provenance_document(db, "run_test").serialize(format="json"))
    assert any("Revision" in str(r.get("prov:type")) for r in document["wasDerivedFrom"].values())
    assert any(r["prov:entity"] == f"ra:critique/{versions[0].plan_id}" for r in document["used"].values())
    assert all(w["traceable"] for w in trace_chains(document, versions[1].plan_id) if w["chains"])
    report = plan_report(db, "run_test")
    assert report.revisions and "fixed 1" in report.revisions[0] and not report.open_points


def _plan_router(fresh: dict) -> Router:
    """The novelty search's answers, and a fresh paper for the plan's targeted search (whose
    query, unlike the novelty search's, mentions routing)."""
    router = _router()
    novelty = {"data": [_s2("s2close", "Merging low-rank adapters across tasks", "2401.00001",
                            "We merge several LoRA modules so one model serves many tasks. "
                            "Merged adapters match per-task accuracy on GLUE."),
                        _s2("s2sparse", "Sparse fine-tuning of language models", None,
                            "Sparse updates fine-tune a small subset of weights. They work on GLUE.")]}

    def planning(request: httpx.Request) -> bool:
        return "routing" in request.url.params.get("query", "")

    router.add("api.semanticscholar.org", "/graph/v1/paper/search", lambda request: httpx.Response(
        200, json={"data": [fresh]} if planning(request) else novelty))
    snippet = {"data": [{"snippet": {"text": "We use eight experts.", "section": "1.1 Setup"}, "score": 0.9,
                         "paper": {"corpusId": 8, "title": FRESH}}]}
    task = {"data": [{"snippet": {"text": "Task vectors add up.", "section": "Method"}, "score": 0.9,
                      "paper": {"corpusId": 7, "title": "Editing models with task arithmetic"}}]}
    router.add("api.semanticscholar.org", "/graph/v1/snippet/search", lambda request: httpx.Response(
        200, json=snippet if planning(request) else task))
    arith = _s2("s2arith", "Editing models with task arithmetic", None,
                "Task vectors add and subtract fine-tuned weights. Adding task vectors builds multi-task models.")
    router.add("api.semanticscholar.org", "/graph/v1/paper/batch", lambda request: httpx.Response(
        200, json=[fresh if i == "CorpusId:8" else arith for i in json.loads(request.content)["ids"]]))
    router.add("arxiv.org", "/html/2402.00002", HTML)
    return router


def test_verification_removes_contradicted_elements_and_flags_unsupported_ones(db, recorder, settings, tmp_path, corpus):  # noqa: F811
    model = PaperModel()
    original = model.extract

    def extract(prompt: str, output: str) -> dict | None:
        if output == "Verdicts":
            verdicts = []
            for number, element in re.findall(r"^\[(\d+)\] Plan element \(([^)]*)\)", prompt, re.MULTILINE):
                label = {"ablations 1": "contradicted", "conditions 1": "not_supported"}.get(element, "supported")
                verdicts.append({"claim": int(number), "label": label, "reason": f"fake {label}"})
            return {"verdicts": verdicts}
        return original(prompt, output)

    model.extract = extract
    _plan(db, recorder, settings, tmp_path, model)
    plan = PlanStore(db).latest("run_test")
    status = {e.label: (e.verdict, e.status) for e in plan.elements}
    assert status["ablations 1"] == ("contradicted", "removed") and status["conditions 1"] == ("not_supported", "flagged")
    c = plan.detail["coverage"]
    assert c["removed"] == 1 and c["elements"] == len(plan.elements) - 1 and c["flagged"] == 2  # with the uncited step

    report = plan_report(db, "run_test")
    assert not report.ablations and report.removed[0].verdict_reason == "fake contradicted"
    markdown = "\n".join(report.markdown())
    assert "### Removed by verification" in markdown and "not supported by its sources: fake not_supported" in markdown

    # Contested elements show their provenance in full; supported ones only their source numbers.
    blocks = re.split(r"\n(?=- )", markdown)
    flagged = next(b for b in blocks if "fake not_supported" in b)
    removed = next(b for b in blocks if "fake contradicted" in b)
    supported = next(b for b in blocks if b.startswith("- ") and "flagged" not in b and "*Why:*" in b)
    for block in (flagged, removed):
        assert "*Rests on:*" in block and re.search(r'\*\[\d+\]\* .+: ".+" \(`p_', block) and "*Derived:* plan round" in block
    assert "*Rests on:*" not in supported and "*Derived:*" not in supported


def test_the_plan_command_lists_the_plan_and_traces_one_element(db, recorder, settings, corpus, tmp_path, monkeypatch):  # noqa: F811
    _plan(db, recorder, settings, tmp_path, PaperModel())
    monkeypatch.chdir(tmp_path)  # the settings' data_dir is tmp_path/var
    runner = CliRunner()
    listed = runner.invoke(app, ["plan", "run_test"])
    assert listed.exit_code == 0, listed.output
    assert "Provenance coverage" in listed.output and "baselines 1" in listed.output

    traced = runner.invoke(app, ["plan", "run_test", "--element", "baselines 1"])
    assert traced.exit_code == 0, traced.output
    assert "decision: fake decision" in traced.output and "paper: Alpha" in traced.output
    assert "section: Abstract" in traced.output and "sentence:" in traced.output
    assert runner.invoke(app, ["plan", "run_test", "--element", "nothing 9"]).exit_code == 1


def test_a_run_without_an_assessed_idea_has_no_plan(db, recorder, settings, tmp_path):
    ctx = _ctx(db, recorder, settings, tmp_path, PaperModel())
    state = run_phase5(ctx, {"run_id": "run_test", "question": "q"})
    assert state["plan"] is None and PlanStore(db).latest("run_test") is None
    assert plan_report(db, "run_test").note == "the run has no assessed idea to plan"
    steps = [e["step"] for e in recorder.events("run_test") if e["activity_type"] == "step"]
    assert steps == ["plan", "critique", "verify"]
