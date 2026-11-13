"""Phase 2: extraction, the quote check, name merging, cross-paper links and the graph."""

import re
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

from fakes import FakeModel, FakeTextEmbedder, fake_llm
from research_agent.cli import app
from research_agent.graph import queries
from research_agent.graph.build import load_graph
from research_agent.graph.extraction import (
    Line,
    Section,
    abstract_item,
    abstract_resolver,
    chunk_sections,
    full_resolver,
    load_sections,
    resolve_addresses,
)
from research_agent.graph.links import Finding, candidate_pairs
from research_agent.graph.names import merge_names, name_key, vague_name
from research_agent.graph.quotes import QuoteChecker, normalize
from research_agent.graph.stages import build_graph_stage, extract_stage
from research_agent.graph.store import JudgmentStore
from research_agent.literature.corpus import Corpus
from research_agent.literature.papers import PaperRecord, PaperStore
from research_agent.literature.parse import ParsedSection, ParsedText
from research_agent.literature.texts import abstract_text, store_text
from research_agent.llm import Role

# The quote check -------------------------------------------------------------

SENTENCES = {
    "p:s0:0": "We propose LoRA – a low‑rank adaptation method.",
    "p:s0:1": "LoRA reaches 90%  accuracy on GLUE.",
    "p:s1:0": "Fine-tuning large models is expensive.",
}


def _item(sentence: str, quote: str, **fields) -> dict:
    return {**fields, "evidence": [{"sentence": sentence, "quote": quote}]}


def test_quotes_pass_despite_dashes_spacing_case_and_end_punctuation():
    result = QuoteChecker(SENTENCES).check({
        "methods": [_item("p:s0:0", '"we propose LoRA - a low-rank adaptation method."', name="LoRA")],
        "results": [_item("p:s0:1", "LoRA reaches 90% accuracy", text="r")],
    })
    assert result.stats.exact == 2 and result.stats.failed == 0
    assert [len(result.output[key]) for key in ("methods", "results")] == [1, 1]


def test_a_quote_cited_to_the_wrong_sentence_is_readdressed():
    result = QuoteChecker(SENTENCES).check({"claims": [_item("p:s0:0", "is expensive", text="c")]})
    assert result.stats.readdressed == 1
    assert result.output["claims"][0]["evidence"] == [
        {"sentence": "p:s1:0", "quote": "is expensive", "check": "readdressed"}]


def test_made_up_quotes_are_dropped_with_their_items():
    item = {"text": "c", "evidence": [
        {"sentence": "p:s0:1", "quote": "reaches 90% accuracy"},
        {"sentence": "p:s0:1", "quote": "reaches 95% accuracy"},
    ]}
    result = QuoteChecker(SENTENCES).check({
        "results": [item],
        "claims": [_item("p:s1:0", "LoRA trains faster than anything", text="invented")],
        "facets": {"purpose": _item("p:s1:0", "nothing like this", text="p"), "mechanism": None},
    })
    assert len(result.output["results"][0]["evidence"]) == 1  # the good quote stays
    assert result.output["claims"] == []
    assert result.output["facets"] == {"purpose": None, "mechanism": None}
    assert result.stats.failed == 3 and result.stats.items == 3 and result.stats.items_kept == 1
    assert any("invented" in d["reason"] for d in result.dropped)


def test_an_ellipsis_may_skip_words_but_not_reorder_them():
    checker = QuoteChecker(SENTENCES)
    assert checker.locate("p:s0:1", "LoRA reaches ... on GLUE")[0] == "exact"
    assert checker.locate("p:s0:1", "on GLUE ... LoRA reaches")[0] == "failed"
    assert checker.locate("p:s0:1", "on")[0] == "failed"  # too short to prove anything


def test_latex_backslashes_and_escape_debris_do_not_fail_a_true_quote():
    # Sentences as some papers' text has them, and quotes as the model sent them back (seen live).
    checker = QuoteChecker({
        "p:s0:0": r"LoRA parameterizes \Delta as a low-rank matrix.",
        "p:s0:1": r"Adapters of size 64 (2\% parameters) attain a best F1 of 90.4%.",
        "p:s0:2": r"Fine-tuning requires 9\times the total number of BERT parameters.",
        "p:s0:3": r"We find an extremely strong fit(\rho=0.97).",
        "p:s0:4": "SetFit [51] 66.9% GPT-3 [4] 62.7%",
    })
    assert checker.locate("p:s0:0", r"parameterizes \\Delta as a low-rank matrix")[0] == "exact"  # doubled
    assert checker.locate("p:s0:1", r"(2\textbackslash% parameters) attain a best F1")[0] == "exact"
    assert checker.locate("p:s0:2", "requires 9\times the total")[0] == "exact"  # "\t" arrived as a tab
    assert checker.locate("p:s0:3", "strong fit(\rho=0.97)")[0] == "exact"  # "\r" arrived as a carriage return
    assert checker.locate("p:s0:1", "of size 64 (2% parameters)")[0] == "exact"  # backslash dropped
    # Real copying errors still fail: a wrong number (behind a zero-width space), a changed word.
    assert checker.locate("p:s0:4", "SetFit [51] 6​2.7%")[0] == "failed"
    assert checker.locate("p:s0:4", "SetFit [51] 6​6.9%")[0] == "exact"
    assert checker.locate("p:s0:2", "Fine-tuning needs 9 times the total")[0] == "failed"


def test_normalize_folds_unicode_variants():
    assert normalize("Low‑Rank  “Adaptation”") == normalize('low-rank "adaptation"')


# Chunks and addresses ----------------------------------------------------------

def _section(idx: int, heading: str, *sizes: int) -> Section:
    return Section(idx, heading, [Line(f"p:s{idx}:{n}", "x" * size) for n, size in enumerate(sizes)])


def test_chunks_keep_sections_whole_and_leave_out_appendices():
    sections = [
        _section(0, "Abstract", 30, 30),
        _section(1, "Introduction", 50),
        _section(2, "Method", 40, 40, 40),  # longer than a chunk: split between sentences
        _section(3, "Appendix A Proofs", 10),
        _section(4, "A.1 More proofs", 10),
    ]
    chunks = chunk_sections(sections, limit=100)
    assert [chunk.sections for chunk in chunks] == [[0], [1], [2], [2]]
    assert [[b.continued for b in chunk.blocks] for chunk in chunks] == [[False], [False], [False], [True]]
    assert [len(chunk.lines) for chunk in chunks] == [2, 1, 2, 1]


def test_labels_resolve_to_full_addresses():
    full = full_resolver("p_1")
    assert full("s3:12") == full("[S3:12]") == full("3:12") == "p_1:s3:12"
    assert full("sentence 4") is None
    item = abstract_item("p_2", "T", None, [Section(0, "Abstract", [Line("p_2:s0:0", "a"), Line("p_2:s0:1", "b")])])
    abstract = abstract_resolver(2, item)
    assert abstract("2:1") == "p_2:s0:1"
    assert abstract("3:1") is None and abstract("2:5") is None
    reply = {"claims": [{"text": "t", "evidence": [{"sentence": "2:0", "quote": "a"}, {"sentence": "x", "quote": "b"}]}]}
    assert resolve_addresses(reply, abstract)["claims"][0]["evidence"] == [
        {"sentence": "p_2:s0:0", "quote": "a"}, {"sentence": "x", "quote": "b"}]


# Names --------------------------------------------------------------------------

def test_name_keys_ignore_spelling_but_keep_versions_apart():
    assert name_key("ResNet-50") == name_key("ResNet50") == name_key("the ResNet 50 model")
    assert name_key("GLUE benchmark") == name_key("GLUE")
    assert name_key("F1 score") == name_key("F1")
    assert name_key("GPT-3") != name_key("GPT-4")


class GroupingModel(FakeModel):
    """Groups LoRA with Low-Rank Adaptation; answers with a bad number too."""

    def __init__(self) -> None:
        def extract(prompt: str, output: str) -> dict | None:
            if output != "NameGroups":
                return None
            numbers = {name: int(n) for n, name in re.findall(r"^\[(\d+)\] (.*)$", prompt, re.MULTILINE)}
            if "LoRA" not in numbers:
                return {"groups": []}
            return {"groups": [{"names": [numbers["LoRA"], numbers["Low-Rank Adaptation"], 99], "canonical": numbers["LoRA"]}]}

        super().__init__(extract=extract)


def test_names_merge_by_key_then_by_the_model_and_the_grouping_is_kept(db, settings):
    model = GroupingModel()
    llm = fake_llm(settings, model)
    mentions = Counter({"LoRA": 3, "Low-Rank Adaptation": 1, "low-rank adaptation": 1, "Adapters": 2})
    entities, stats = merge_names(llm, JudgmentStore(db, llm.models(Role.EXTRACTION)), "method", mentions)

    lora = entities[name_key("LoRA")]
    assert lora is entities[name_key("Low-Rank Adaptation")]
    assert lora.name == "LoRA" and set(lora.aliases) == {"LoRA", "Low-Rank Adaptation", "low-rank adaptation"}
    assert entities[name_key("Adapters")] is not lora
    assert stats == {"calls": 1, "reused": 0, "by_key": 1, "by_model": 1}

    _, again = merge_names(llm, JudgmentStore(db, llm.models(Role.EXTRACTION)), "method", mentions)
    assert again["calls"] == 0 and again["reused"] == 1
    assert len([c for c in model.calls if c["output"] == "NameGroups"]) == 1


# Candidate pairs ------------------------------------------------------------------

def _finding(node: str, paper: str, text: str, **fields) -> Finding:
    return Finding(node, paper, f"title {paper}", fields.pop("kind", "claim"), text, text, **fields)


def test_candidate_pairs_join_different_papers_by_similar_text_or_shared_setting():
    findings = [
        _finding("a", "p1", "same words here"),
        _finding("b", "p2", "same words here too"),
        _finding("c", "p1", "same words here again"),  # same paper as a: never paired with it
        _finding("d", "p3", "unrelated", kind="result", dataset="glue", metric="acc"),
        _finding("e", "p4", "different", kind="result", dataset="glue", metric="acc"),
        _finding("f", "p5", "other", kind="result", dataset="glue", metric="f1"),
    ]
    vectors = {"same words here": np.array([1.0, 0, 0]), "same words here too": np.array([0.96, 0.28, 0]),
               "same words here again": np.array([1.0, 0, 0]), "unrelated": np.array([0, 1.0, 0]),
               "different": np.array([0, 0, 1.0]), "other": np.array([0.5, 0.5, 0.7071])}
    pairs = {(p.a.node_id, p.b.node_id): p.found_by for p in candidate_pairs(findings, vectors)}
    assert pairs == {("a", "b"): ["similar_text"], ("b", "c"): ["similar_text"], ("d", "e"): ["shared_setting"]}


# The stages, end to end ----------------------------------------------------------------

ALPHA = "Alpha: low-rank adaptation of large language models"
BETA = "Beta: a study of low-rank adaptation on the GLUE benchmark"
GAMMA = "Gamma: adapters for parameter-efficient transfer learning"

ALPHA_REPLY = {
    "tasks": [_item("s1:0", "Fine-tuning large models is expensive", name="parameter-efficient fine-tuning")],
    "methods": [_item("s0:0", "We propose LoRA", name="LoRA", role="proposed")],
    "datasets": [_item("s0:1", "accuracy on GLUE", name="GLUE")],
    "metrics": [_item("s0:0", "90% accuracy", name="accuracy")],  # cites the wrong sentence
    "results": [_item("s0:1", "LoRA reaches 90% accuracy on GLUE", text="LoRA reaches 90% accuracy on GLUE.",
                      method="LoRA", dataset="GLUE", metric="accuracy", conditions=None)],
    "claims": [_item("s1:1", "LoRA trains faster than anything", text="LoRA trains faster than anything.")],  # made up
    "limitations": [_item("s2:0", "cannot merge several tasks", text="LoRA cannot merge several tasks at once.")],
    "assumptions": [],
    "future_work": [],
    "facets": {"purpose": _item("s1:0", "Fine-tuning large models is expensive.", text="Cheaper fine-tuning."),
               "mechanism": _item("s1:1", "freezes the pretrained weights", text="LoRA freezes the weights."),
               "evaluation": None},
}
BETA_REPLY = {
    "tasks": [], "claims": [], "limitations": [], "assumptions": [], "future_work": [],
    "methods": [_item("s0:0", "We study Low-Rank Adaptation", name="Low-Rank Adaptation", role="used")],
    "datasets": [_item("s0:0", "the GLUE benchmark", name="GLUE benchmark")],
    "metrics": [_item("s0:1", "85% accuracy", name="Accuracy")],
    "results": [_item("s0:1", "reaches 85% accuracy on GLUE", text="Low-Rank Adaptation reaches 85% accuracy on GLUE.",
                      method="Low-Rank Adaptation", dataset="GLUE benchmark", metric="Accuracy", conditions=None)],
    "facets": {"purpose": None, "mechanism": None, "evaluation": None},
}


class PaperModel(GroupingModel):
    """Extracts Alpha and Beta as above; calls their results contradictory."""

    def __init__(self, relation: str = "contradicts", same_setting: bool = True) -> None:
        super().__init__()
        grouping = self.extract

        def extract(prompt: str, output: str) -> dict | None:
            if output == "OpeningChunkExtraction" and f"Paper: {ALPHA}" in prompt:
                return ALPHA_REPLY
            if output == "OpeningChunkExtraction" and f"Paper: {BETA}" in prompt:
                return BETA_REPLY
            if output == "PairLabels":
                return {"labels": [
                    {"pair": int(n), "same_setting": same_setting, "reason": "different numbers",
                     "relation": relation if "85% accuracy" in block and "90% accuracy" in block else "unrelated"}
                    for n, block in re.findall(r"^\[(\d+)\]\n(.*?)(?=^\[\d+\]$|\Z)", prompt, re.MULTILINE | re.DOTALL)
                ]}
            return grouping(prompt, output)

        self.extract = extract


def _add_paper(db, title: str, sections: list[ParsedSection] | None, abstract: str, rank: int, level: str) -> str:
    paper_id = PaperStore(db).upsert(PaperRecord(source="arxiv", title=title, abstract=abstract, year=2021))
    parsed = ParsedText(title, sections) if sections else abstract_text(title, abstract)
    store_text(db, paper_id, parsed, source="arxiv_html" if sections else "abstract", parser="test")
    corpus = Corpus(db, "run_test")
    corpus.add(paper_id, {"channel": "test", "rank": rank})
    corpus.set_rank(paper_id, score=1.0, rank=rank, read_level=level, reason="test")
    return paper_id


@pytest.fixture
def corpus(db, recorder):
    return {
        "alpha": _add_paper(db, ALPHA, [
            ParsedSection("Abstract", ["We propose LoRA, a low-rank adaptation method. LoRA reaches 90% accuracy on GLUE."]),
            ParsedSection("Introduction", ["Fine-tuning large models is expensive. LoRA freezes the pretrained weights."]),
            ParsedSection("Limitations", ["LoRA cannot merge several tasks at once."]),
            ParsedSection("Appendix A Extra results", ["This appendix is never read."]),
        ], "unused", 1, "full"),
        "beta": _add_paper(db, BETA, [
            ParsedSection("Abstract", ["We study Low-Rank Adaptation on the GLUE benchmark. "
                                       "Low-Rank Adaptation reaches 85% accuracy on GLUE."]),
        ], "unused", 2, "full"),
        "gamma": _add_paper(db, GAMMA, None, "Adapters add small trainable layers. They match full fine-tuning.", 3, "abstract"),
    }


def _run_stages(db, recorder, settings, model, embedder=None):
    ctx = SimpleNamespace(db=db, settings=settings, llm=fake_llm(settings, model), text_embedder=embedder or FakeTextEmbedder())
    state = {"run_id": "run_test"}
    with recorder.step("run_test", "extract") as event:
        state |= extract_stage(state, ctx, event)
    with recorder.step("run_test", "build_graph") as event:
        state |= build_graph_stage(state, ctx, event)
    return state


def _nodes(db, node_type):
    return {row["name"]: row for row in db.query(
        "SELECT * FROM nodes WHERE run_id = 'run_test' AND node_type = ?", (node_type,))}


def _edges(db, edge_type):
    return db.query("SELECT * FROM edges WHERE run_id = 'run_test' AND edge_type = ?", (edge_type,))


def test_the_corpus_becomes_an_evidence_backed_graph(db, recorder, settings, corpus):
    model = PaperModel()
    state = _run_stages(db, recorder, settings, model)

    # One call per full paper (each fits one chunk) and one for the abstract; the appendix stays unread.
    extraction_calls = [c for c in model.calls if c["output"] in ("OpeningChunkExtraction", "AbstractExtractions")]
    assert len(extraction_calls) == 3
    assert not any("never read" in c["prompt"] for c in model.calls)
    assert state["extraction"] == {"full_papers": 2, "chunks": 2, "abstracts": 1, "calls": 3, "reused": 0, "failed": 0}

    # The quote check: Alpha's wrongly cited metric is readdressed, its made-up claim dropped.
    quotes = state["graph"]["quote_check"]
    assert (quotes["readdressed"], quotes["failed"]) == (1, 1)
    assert "LoRA trains faster than anything." not in _nodes(db, "claim")

    # Names merged across papers: by key (GLUE, accuracy) and by the model (LoRA).
    methods, datasets = _nodes(db, "method"), _nodes(db, "dataset")
    assert list(methods) == ["LoRA"] and list(datasets) == ["GLUE"]
    aliases = {row["alias"] for row in db.query("SELECT alias FROM node_aliases WHERE node_id = ?", (methods["LoRA"]["node_id"],))}
    assert aliases == {"LoRA", "Low-Rank Adaptation"}
    assert len(_nodes(db, "metric")) == 1
    proposes, uses = _edges(db, "proposes"), _edges(db, "uses")
    assert len(proposes) == len(uses) == 1 and proposes[0]["dst_id"] == uses[0]["dst_id"] == methods["LoRA"]["node_id"]

    # Results point at their method, dataset and metric; the abstract paper reports its first sentence.
    results = _nodes(db, "result")
    assert set(results) == {"LoRA reaches 90% accuracy on GLUE.", "Low-Rank Adaptation reaches 85% accuracy on GLUE."}
    about = {(e["src_id"], e["dst_id"]) for e in _edges(db, "about")}
    assert about == {(r["node_id"], methods["LoRA"]["node_id"]) for r in results.values()}
    assert "Adapters add small trainable layers." in _nodes(db, "claim")
    assert set(_nodes(db, "facet")) == {"Cheaper fine-tuning.", "LoRA freezes the weights."}
    assert list(_nodes(db, "limitation")) == ["LoRA cannot merge several tasks at once."]

    # The two results share a setting and disagree.
    contradicts = _edges(db, "contradicts")
    assert len(contradicts) == 1
    assert {contradicts[0]["src_id"], contradicts[0]["dst_id"]} == {r["node_id"] for r in results.values()}
    links = state["graph"]["links"]
    assert links["shared_setting"] == 1 and links["contradicts"] == 1

    # Every piece of evidence is a real sentence that contains its quote.
    rows = db.query("SELECT e.quote, s.text FROM evidence e JOIN sentences s USING (sentence_id)")
    assert len(rows) == db.query("SELECT COUNT(*) AS n FROM evidence")[0]["n"] > 10
    assert all(normalize(row["quote"]).strip(" .\"'") in normalize(row["text"]) for row in rows)
    edge_evidence = db.query("SELECT COUNT(*) AS n FROM evidence WHERE owner_id = ?", (contradicts[0]["edge_id"],))
    assert edge_evidence[0]["n"] == 2  # one sentence from each result


def test_a_second_build_reuses_every_llm_answer_and_replaces_the_graph(db, recorder, settings, corpus):
    model = PaperModel()
    embedder = FakeTextEmbedder()
    _run_stages(db, recorder, settings, model, embedder)
    counts = Counter(row["node_type"] for row in db.query("SELECT node_type FROM nodes"))
    calls, sent = len(model.calls), len(embedder.sent)

    state = _run_stages(db, recorder, settings, model, embedder)

    assert len(model.calls) == calls and len(embedder.sent) == sent
    assert state["extraction"]["reused"] == 3 and state["extraction"]["calls"] == 0
    assert Counter(row["node_type"] for row in db.query("SELECT node_type FROM nodes")) == counts


def test_a_contradiction_in_different_settings_is_not_kept(db, recorder, settings, corpus):
    state = _run_stages(db, recorder, settings, PaperModel(same_setting=False))
    assert _edges(db, "contradicts") == []
    assert state["graph"]["links"]["contradicts_other_setting"] == 1
    build = [e for e in recorder.events("run_test") if e["step"] == "build_graph" and e["activity_type"] == "step"][0]
    assert any(d["reason"].startswith("contradicts, but not in the same setting") for d in build["dropped"])


def test_without_an_embedding_model_pairs_come_from_shared_settings_only(db, recorder, settings, corpus):
    ctx = SimpleNamespace(db=db, settings=settings, llm=fake_llm(settings, PaperModel()), text_embedder=None)
    with recorder.step("run_test", "extract") as event:
        extract_stage({"run_id": "run_test"}, ctx, event)
    with recorder.step("run_test", "build_graph") as event:
        state = build_graph_stage({"run_id": "run_test"}, ctx, event)
    links = state["graph"]["links"]
    assert "RA_EMBEDDING" in links["embedding"]["status"]
    assert links["candidates"] == links["shared_setting"] == 1


def test_queries_answer_from_the_graph_with_evidence(db, recorder, settings, corpus):
    _run_stages(db, recorder, settings, PaperModel())
    graph = load_graph(db, "run_test")

    methods = queries.methods_on_dataset(graph, db, "glue")
    assert {a.subject for a in methods} == {"LoRA"} and all(a.evidence for a in methods)
    assert queries.methods_on_dataset(graph, db, "ImageNet") == []

    by_alias = queries.findings_about_method(graph, db, "low-rank adaptation")
    assert [a.subject for a in by_alias].count("LoRA") == 2
    assert any(a.subject.strip() == "contradicts" for a in by_alias)

    limitations = queries.limitations_for_task(graph, db, "parameter-efficient fine-tuning")
    assert [a.detail for a in limitations] == ["LoRA cannot merge several tasks at once."]
    assert len(queries.conflicts(graph, db)) == 1

    sentence = f"{corpus['alpha']}:s0:1"
    resting = queries.resting_on(graph, db, sentence)
    # The result, and the paper's links to the dataset and metric it names (entities carry no evidence
    # themselves: each paper's link to them does).
    assert {"result", "reports", "evaluates_on", "measured_by"} <= {a.subject for a in resting}


def test_reparsing_a_paper_keeps_the_evidence_on_its_old_sentences(db, recorder, settings, corpus):
    _run_stages(db, recorder, settings, PaperModel())
    alpha = corpus["alpha"]
    count = "SELECT COUNT(*) AS n FROM evidence e JOIN sentences s USING (sentence_id) WHERE e.sentence_id LIKE ?"
    before = db.query(count, (f"{alpha}:%",))[0]["n"]
    assert before > 0
    store_text(db, alpha, ParsedText(ALPHA, [ParsedSection("Abstract", ["A new version."])]), source="arxiv_html", parser="test")
    assert db.query(count, (f"{alpha}:%",))[0]["n"] == before  # still resolving to the sentences they quoted
    # A later extraction reads the new text only.
    assert [line.text for s in load_sections(db, alpha) for line in s.lines] == ["A new version."]


def test_the_graph_command_shows_the_graph_and_answers_one_question(db, recorder, settings, corpus, tmp_path, monkeypatch):
    _run_stages(db, recorder, settings, PaperModel())
    monkeypatch.chdir(tmp_path)  # the settings' data_dir is tmp_path/var
    runner = CliRunner()

    summary = runner.invoke(app, ["graph", "run_test"])
    assert summary.exit_code == 0, summary.output
    assert "method" in summary.output and "contradicts" in summary.output

    answer = runner.invoke(app, ["graph", "run_test", "--methods-on", "GLUE"])
    assert answer.exit_code == 0, answer.output
    assert "LoRA" in answer.output and "accuracy" in answer.output

    assert runner.invoke(app, ["graph", "run_test", "--conflicts", "--about", "LoRA"]).exit_code == 2
    assert runner.invoke(app, ["graph", "run_missing"]).exit_code == 1


def test_storing_the_same_parse_again_keeps_the_evidence(db, recorder, settings, corpus):
    _run_stages(db, recorder, settings, PaperModel())
    beta = corpus["beta"]
    before = db.query("SELECT COUNT(*) AS n FROM evidence WHERE sentence_id LIKE ?", (f"{beta}:%",))[0]["n"]
    same = ParsedText(BETA, [ParsedSection("Abstract", ["We study Low-Rank Adaptation on the GLUE benchmark. "
                                                        "Low-Rank Adaptation reaches 85% accuracy on GLUE."])])
    assert store_text(db, beta, same, source="pdf", parser="test") == 2
    assert db.query("SELECT COUNT(*) AS n FROM evidence WHERE sentence_id LIKE ?", (f"{beta}:%",))[0]["n"] == before > 0
    assert db.query("SELECT text_source FROM papers WHERE paper_id = ?", (beta,))[0]["text_source"] == "pdf"


@pytest.mark.parametrize(("kind", "name"), [
    ("dataset", "several benchmarks"), ("dataset", "technical domains"), ("metric", "all the evaluation metrics"),
    ("task", "understanding"), ("method", "Large language models (LLMs)"),
])
def test_vague_names_get_no_node(kind, name):
    assert vague_name(kind, name)


@pytest.mark.parametrize(("kind", "name"), [
    ("dataset", "GLUE"), ("dataset", "seven Atari 2600 games"), ("metric", "accuracy"),
    ("task", "open-domain question answering"), ("method", "Retrieval-Augmented Generation (RAG)"),
])
def test_real_names_are_kept(kind, name):
    assert not vague_name(kind, name)
