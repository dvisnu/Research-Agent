"""Phase 3: related work, the landscape grid and research gaps."""

import re
from types import SimpleNamespace

import networkx as nx

from fakes import FakeEmbedder, FakeModel, FakeTextEmbedder, fake_llm
from research_agent.gaps.gaps import Candidate, rank, write_up
from research_agent.gaps.landscape import build_landscape
from research_agent.gaps.related import related_papers
from research_agent.gaps.stages import gaps_stage, landscape_stage, related_work_stage
from research_agent.graph.quotes import normalize
from research_agent.graph.store import JudgmentStore
from research_agent.llm import Role
from research_agent.report import RunReport, to_markdown
from test_graph import PaperModel, _run_stages, corpus  # noqa: F401  (corpus is a fixture)


def _graph(nodes: dict[str, tuple[str, str, dict]], edges: list[tuple[str, str, str]]) -> nx.MultiDiGraph:
    """nodes: id -> (type, name, attrs); edges: (type, src, dst)."""
    graph = nx.MultiDiGraph()
    for node, (node_type, name, attrs) in nodes.items():
        graph.add_node(node, node_type=node_type, name=name, aliases=[], attrs=attrs)
    for n, (edge_type, src, dst) in enumerate(edges):
        graph.add_edge(src, dst, key=f"e{n}", edge_type=edge_type, attrs={})
    return graph


def _paper(paper_id: str) -> tuple[str, str, dict]:
    return ("paper", f"Paper {paper_id}", {"paper_id": paper_id, "year": 2024})


# Related work ------------------------------------------------------------------------

def test_graph_neighbours_of_the_closest_paper_rise_above_isolated_papers():
    graph = _graph(
        {"a": _paper("A"), "b": _paper("B"), "c": _paper("C"), "m": ("method", "LoRA", {}),
         "f": ("facet", "Make fine-tuning cheap.", {"facet": "purpose"})},
        [("proposes", "a", "m"), ("uses", "b", "m"), ("has_facet", "a", "f")],
    )
    graph.edges["a", "f", "e2"]["attrs"] = {"facet": "purpose"}
    related = related_papers(graph, {"A": 0.9, "B": 0.6, "C": 0.6})
    assert [p["paper_id"] for p in related] == ["A", "B", "C"]  # B shares a method with A; C shares nothing
    assert related[0]["shares"] == ["LoRA"] and related[2]["shares"] == []
    assert related[0]["purpose"] == "Make fine-tuning cheap."


# The landscape grid ---------------------------------------------------------------------

def test_the_grid_marks_results_shared_papers_and_empty_cells():
    graph = _graph(
        {"p1": _paper("1"), "p2": _paper("2"), "p3": _paper("3"),
         "lora": ("method", "LoRA", {}), "adapters": ("method", "Adapters", {}), "bitfit": ("method", "BitFit", {}),
         "glue": ("dataset", "GLUE", {}), "squad": ("dataset", "SQuAD", {}), "acc": ("metric", "accuracy", {}),
         "r1": ("result", "LoRA gets 90 on GLUE.", {})},
        [("proposes", "p1", "lora"), ("evaluates_on", "p1", "glue"), ("reports", "p1", "r1"),
         ("about", "r1", "lora"), ("evaluates_on", "r1", "glue"), ("measured_by", "r1", "acc"),
         ("uses", "p2", "lora"), ("uses", "p2", "adapters"), ("evaluates_on", "p2", "squad"),
         ("uses", "p3", "bitfit"), ("evaluates_on", "p3", "glue")],
    )
    landscape = build_landscape(graph, methods=2, datasets=2)
    assert [m["name"] for m in landscape["methods"]] == ["LoRA", "Adapters"]  # by papers, then name; BitFit cut
    assert [d["name"] for d in landscape["datasets"]] == ["GLUE", "SQuAD"]
    status = [[cell["status"] for cell in row] for row in landscape["cells"]]
    assert status == [["result", "same_paper"], ["empty", "same_paper"]]
    assert landscape["cells"][0][0]["metrics"] == ["accuracy"]


def test_benchmarks_and_metrics_typed_as_methods_stay_out_of_the_grid():
    graph = _graph(
        {"p1": _paper("1"), "p2": _paper("2"),
         "judge": ("method", "LLM-as-a-judge", {}), "mtb_m": ("method", "MT-Bench", {}), "bleu_m": ("method", "BLEU", {}),
         "mtb": ("dataset", "MT Bench", {}), "bleu_d": ("dataset", "BLEU", {}), "arena": ("dataset", "Chatbot Arena", {}),
         "bleu": ("metric", "BLEU", {}), "rouge_m": ("method", "ROUGE-L", {})},
        [("uses", "p1", "judge"), ("uses", "p1", "mtb_m"), ("uses", "p2", "mtb_m"), ("uses", "p2", "bleu_m"), ("uses", "p2", "rouge_m"),
         ("evaluates_on", "p1", "mtb"), ("evaluates_on", "p2", "bleu_d"), ("evaluates_on", "p2", "arena")],
    )
    landscape = build_landscape(graph)
    assert [m["name"] for m in landscape["methods"]] == ["LLM-as-a-judge"]
    assert [d["name"] for d in landscape["datasets"]] == ["Chatbot Arena", "MT Bench"]


# Ranking and the write-up -----------------------------------------------------------------

def _candidate(text: str, tier: int, relevance: float, *papers: str, kind: str = "stated") -> Candidate:
    evidence = [{"sentence": f"{p}:s0:0", "quote": text} for p in papers]
    return Candidate(kind, text, evidence, [f"n_{text}"], detail="limitation", relevance=relevance, tier=tier)


def test_gaps_rank_by_relevance_tier_then_independent_papers():
    ranked = rank([
        _candidate("close, one paper", 2, 0.80, "p1"),
        _candidate("close, three papers", 2, 0.78, "p1", "p2", "p3"),
        _candidate("far, five papers", 0, 0.70, "p1", "p2", "p3", "p4", "p5"),
        _candidate("closest of all", 3, 0.85, "p9"),
    ])
    assert [c.text for c in ranked] == ["closest of all", "close, three papers", "close, one paper", "far, five papers"]


def test_the_write_up_drops_non_gaps_and_folds_duplicates_into_the_earlier_gap(db, settings):
    def extract(prompt: str, output: str) -> dict | None:
        if output != "GapWriteups":
            return None
        return {"gaps": [
            {"candidate": 1, "gap": "Gap one.", "warrant": "W1.", "keep": True, "duplicate_of": None},
            {"candidate": 2, "gap": "Same as one.", "warrant": "W2.", "keep": False, "duplicate_of": 1},
            {"candidate": 3, "gap": "Not a gap.", "warrant": "W3.", "keep": False, "duplicate_of": None},
            {"candidate": 4, "gap": "Gap four.", "warrant": "W4.", "keep": True, "duplicate_of": None},
        ]}

    model = FakeModel(extract=extract)
    llm = fake_llm(settings, model)
    shortlist = [_candidate("one", 2, 0.8, "p1"), _candidate("two", 2, 0.79, "p2", "p3"),
                 _candidate("three", 1, 0.7, "p4"), _candidate("four", 2, 0.78, "p5", "p6")]
    judgments = JudgmentStore(db, llm.models(Role.REASONING))
    gaps, stats = write_up(llm, judgments, "q", shortlist, {})

    # One took two's papers (3 in all), so it still leads; three is dropped.
    assert [(g.gap, len(g.candidate.papers)) for g in gaps] == [("Gap one.", 3), ("Gap four.", 2)]
    assert gaps[0].warrant == "W1." and stats["merged"] == 1 and stats["dropped"] == 1
    assert [c["output"] for c in model.calls] == ["GapWriteups"] and model.calls[0]["model"] == "reasoner"

    write_up(llm, judgments, "q", [_candidate("one", 2, 0.8, "p1"), _candidate("two", 2, 0.79, "p2", "p3"),
                                   _candidate("three", 1, 0.7, "p4"), _candidate("four", 2, 0.78, "p5", "p6")], {})
    assert len(model.calls) == 1  # the same shortlist reuses the kept write-up


# The stages, end to end --------------------------------------------------------------------

def _phase3(db, recorder, settings, model):
    _run_stages(db, recorder, settings, model)
    ctx = SimpleNamespace(db=db, settings=settings, llm=fake_llm(settings, model),
                          text_embedder=FakeTextEmbedder(), embedder=FakeEmbedder())
    state = {"run_id": "run_test", "question": "Does low-rank adaptation match full fine-tuning on GLUE?"}
    for step, work in (("related_work", related_work_stage), ("landscape", landscape_stage), ("gaps", gaps_stage)):
        with recorder.step("run_test", step) as event:
            state |= work(state, ctx, event)
    return state


def test_the_corpus_yields_ranked_gaps_whose_evidence_is_real(db, recorder, settings, corpus):  # noqa: F811
    model = PaperModel()
    state = _phase3(db, recorder, settings, model)

    assert {p["title"][:5] for p in state["related_work"]} == {"Alpha", "Beta:", "Gamma"}
    assert [m["name"] for m in state["landscape"]["methods"]] == ["LoRA"]
    assert state["landscape"]["cells"] == [[{"status": "result", "results": 2, "papers": 2, "metrics": ["Accuracy"]}]]

    gaps = state["gaps"]
    kinds = {g["kind"] for g in gaps}
    assert kinds == {"stated", "conflict"}  # the limitation, and the two results that disagree
    assert [g["rank"] for g in gaps] == list(range(1, len(gaps) + 1)) and len(gaps) <= 5
    assert all(g["warrant"] == "fake warrant" and g["gap"].startswith("Gap: ") for g in gaps)
    conflict = next(g for g in gaps if g["kind"] == "conflict")
    assert conflict["papers"] == 2

    # Stored as graph nodes; every piece of their evidence is a real sentence containing its quote.
    nodes = db.query("SELECT node_id, attrs FROM nodes WHERE run_id = 'run_test' AND node_type = 'gap'")
    assert len(nodes) == len(gaps)
    rows = db.query(
        "SELECT e.quote, s.text FROM evidence e JOIN sentences s USING (sentence_id)"
        " WHERE e.owner_id IN (SELECT node_id FROM nodes WHERE node_type = 'gap')")
    assert rows and all(normalize(r["quote"]).strip(" .\"'") in normalize(r["text"]) for r in rows)
    rests_on = db.query("SELECT COUNT(*) AS n FROM edges WHERE edge_type = 'rests_on' AND run_id = 'run_test'")[0]["n"]
    assert rests_on >= len(gaps)

    # The reasoning model wrote them up once; a second pass reuses that and replaces the stored gaps.
    calls = len(model.calls)
    again = _phase3(db, recorder, settings, model)
    assert len(model.calls) == calls and [g["gap"] for g in again["gaps"]] == [g["gap"] for g in gaps]
    assert db.query("SELECT COUNT(*) AS n FROM nodes WHERE node_type = 'gap'")[0]["n"] == len(gaps)


def test_the_report_shows_gaps_related_work_and_the_grid(db, recorder, settings, corpus):  # noqa: F811
    state = _phase3(db, recorder, settings, PaperModel())
    report = RunReport(run_id="run_test", question="q", generated_at="now")
    report.add_phase3(state["related_work"], state["landscape"], state["gaps"])
    markdown = to_markdown(report)
    assert "## Research gaps" in markdown and "**Warrant:** fake warrant" in markdown
    assert "## Related work" in markdown and "## Research landscape" in markdown
    assert re.search(r"\| LoRA \| R2 \|", markdown)
    assert report.research_gaps[0].evidence and report.research_gaps[0].evidence[0].paper


def test_the_graph_command_lists_the_gaps(db, recorder, settings, corpus, tmp_path, monkeypatch):  # noqa: F811
    from typer.testing import CliRunner

    from research_agent.cli import app

    _phase3(db, recorder, settings, PaperModel())
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["graph", "run_test", "--gaps"])
    assert result.exit_code == 0, result.output
    assert "Research gaps" in result.output and "fake warrant" in result.output


def test_same_kind_candidates_that_say_the_same_thing_merge_into_the_most_relevant():
    import numpy as np

    from research_agent.gaps.gaps import merge_duplicates

    a = _candidate("retrieval noise hurts", 2, 0.80, "p1")
    b = _candidate("noisy retrieval degrades output", 2, 0.75, "p2")
    c = _candidate("noise in retrieved text is harmful", 2, 0.70, "p3")  # close to b, not to a: no chaining
    d = _candidate("findings disagree", 2, 0.78, "p4", "p5", kind="conflict")  # same vector as a, other kind
    unit = lambda *v: np.array(v) / np.linalg.norm(v)  # noqa: E731
    vectors = {a.text: unit(1, 0, 0), b.text: unit(0.95, 0.31, 0), c.text: unit(0.70, 0.71, 0), d.text: unit(1, 0, 0)}
    kept, merged = merge_duplicates([c, b, a, d], vectors)
    assert merged == 1
    assert [k.text for k in kept] == [a.text, d.text, c.text]  # by relevance; b folded into a
    assert kept[0].papers == {"p1", "p2"} and len(kept[0].evidence) == 2


def test_relevance_falls_back_to_specter2_when_a_vector_is_missing():
    import numpy as np

    from research_agent.gaps.gaps import score_relevance

    candidates = [_candidate("low rank adaptation", 0, 0, "p1"), _candidate("prompt tuning", 0, 0, "p2")]
    question = "low rank adaptation of language models"
    vectors = {question: np.array([1.0, 0]), "low rank adaptation": np.array([1.0, 0])}  # "prompt tuning" missing
    assert score_relevance(candidates, FakeEmbedder(), question, vectors) == "specter2"
    vectors["prompt tuning"] = np.array([0, 1.0])
    assert score_relevance(candidates, FakeEmbedder(), question, vectors) == "text embeddings"
    assert candidates[0].relevance == 1.0 and candidates[1].relevance == 0.0
    assert candidates[0].tier > candidates[1].tier


def test_gaps_rank_with_specter2_when_no_embedding_model_is_set(db, recorder, settings, corpus):  # noqa: F811
    _run_stages(db, recorder, settings, PaperModel())
    ctx = SimpleNamespace(db=db, settings=settings, llm=fake_llm(settings, PaperModel()), text_embedder=None,
                          embedder=FakeEmbedder())
    with recorder.step("run_test", "gaps") as event:
        state = gaps_stage({"run_id": "run_test", "question": "q about LoRA"}, ctx, event)
    assert state["gaps"] and event.detail["gaps"]["relevance_by"] == "specter2"
    assert "RA_EMBEDDING" in event.detail["gaps"]["embedding"]["status"]


def test_a_report_rebuilt_from_storage_matches_the_runs_results(db, recorder, settings, corpus, tmp_path, monkeypatch):  # noqa: F811
    from typer.testing import CliRunner

    from research_agent.cli import app
    from research_agent.pipeline.stored_report import stored_report

    state = _phase3(db, recorder, settings, PaperModel())
    rebuilt = stored_report(db, "run_test")
    assert [g.gap for g in rebuilt.research_gaps] == [g["gap"] for g in state["gaps"]]
    assert [g.evidence[0].sentence_id for g in rebuilt.research_gaps] == [
        sorted(e["sentence_id"] for e in g["evidence"])[0] for g in state["gaps"]]
    assert [p.title for p in rebuilt.related_work] == [p["title"] for p in state["related_work"]]
    assert rebuilt.landscape.methods == ["LoRA"] and rebuilt.knowledge_graph.papers_extracted == 3

    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["report", "run_test"])
    assert result.exit_code == 0, result.output
    markdown = (tmp_path / "var" / "runs" / "run_test" / "report.md").read_text(encoding="utf-8")
    assert "## Research gaps" in markdown and "## Knowledge graph" in markdown


def test_a_pair_some_paper_names_together_is_not_a_structural_gap(db, recorder, settings, corpus):  # noqa: F811
    from research_agent.gaps.gaps import papers_mentioning_both

    _run_stages(db, recorder, settings, PaperModel())
    # Alpha names both LoRA and GLUE; no paper names BitFit; "QA"-length names are ignored by the caller.
    assert papers_mentioning_both(db, "run_test", {"LoRA", "Low-Rank Adaptation"}, {"GLUE"}) == sorted(
        [corpus["alpha"], corpus["beta"]])
    assert papers_mentioning_both(db, "run_test", {"BitFit"}, {"GLUE"}) == []
    assert papers_mentioning_both(db, "run_test", set(), {"GLUE"}) == []
