import json
from types import SimpleNamespace

from fakes import FakeEmbedder, FakeModel, fake_llm
from research_agent.graph.store import JudgmentStore
from research_agent.literature.brief import write_brief
from research_agent.literature.corpus import Corpus
from research_agent.literature.papers import PaperRecord, PaperStore
from research_agent.literature.stages import rank_stage
from research_agent.llm import LLM, Completion, Role
from research_agent.storage import Database

QUESTION = "How can retrieval-augmented generation systems reduce hallucinations when documents conflict?"


class DraftModel:
    """Replies with one fixed brief draft."""

    def __init__(self, draft: dict) -> None:
        self.draft = draft

    def complete(self, *, model, system, messages, json_schema):
        return Completion(json.dumps(self.draft))


def _brief(settings, draft: dict):
    llm = LLM(settings.model_copy(update={"llm_reasoning": "fake:m"}), providers={"fake": DraftModel(draft)})
    with Database(settings.db_path) as db:
        return write_brief(llm, QUESTION, JudgmentStore(db, llm.models(Role.REASONING)))


def test_brief_turns_each_angle_into_queries(settings):
    brief = _brief(settings, {
        "input_kind": "question",
        "problem": "Retrieval-augmented LLMs hallucinate when retrieved evidence conflicts.",
        "angles": [
            {"label": "conflicts", "keywords": ["knowledge conflicts", "retrieval-augmented generation", "hallucination", "LLM"],
             "passage": "We study how  language models resolve\nconflicting retrieved evidence."},
            {"label": "methods", "keywords": ['"faithfulness"', "context: attribution", "Faithfulness"],
             "passage": "We propose a decoding method."},
        ],
    })
    own, first, second = brief.angles
    # The question as written always comes first: its content words.
    assert own.keywords == ["retrieval-augmented", "generation", "hallucinations", "documents", "conflict"]
    assert own.arxiv_query == 'all:"retrieval augmented" AND all:generation AND all:hallucinations AND all:documents'
    assert own.passage_query == QUESTION
    assert first.keyword_query == "knowledge conflicts retrieval-augmented generation hallucination LLM"
    # arXiv ANDs every term, so only the first two of the model's phrases are used, quoted.
    assert first.arxiv_query == 'all:"knowledge conflicts" AND all:"retrieval augmented generation"'
    assert first.passage_query == "We study how language models resolve conflicting retrieved evidence."
    # Quotes and colons would break arXiv's syntax; duplicates are dropped.
    assert second.keywords == ["faithfulness", "context attribution"]
    assert brief.rank_query == QUESTION and brief.facets is None


def test_brief_keeps_an_ideas_facets(settings):
    facets = {"purpose": "better retrieval", "mechanism": "citation-aware embeddings", "evaluation": "recall@k"}
    brief = _brief(settings, {
        "input_kind": "idea", "problem": "p", "facets": facets,
        "angles": [{"label": "a", "keywords": ["scientific retrieval"], "passage": "s"}],
    })
    assert brief.input_kind == "idea" and brief.facets.model_dump() == facets


def test_search_leaders_favour_papers_found_by_several_channels(db, recorder):
    store, corpus = PaperStore(db), Corpus(db, "run_test")
    a = store.upsert(PaperRecord(source="s2", title="Paper A with a long enough title"))
    b = store.upsert(PaperRecord(source="s2", title="Paper B with a long enough title"))
    c = store.upsert(PaperRecord(source="s2", title="Paper C with a long enough title"))
    corpus.add(a, {"channel": "s2_search", "rank": 1})
    corpus.add(b, {"channel": "s2_search", "rank": 2})
    corpus.add(b, {"channel": "openalex_search", "rank": 3})
    corpus.add(c, {"channel": "s2_references", "via": a})  # expansion hits carry no rank
    assert [row["paper_id"] for row in corpus.search_leaders(2)] == [b, a]


def _rank(db, recorder, settings, papers: dict[str, PaperRecord], model: FakeModel, full: int, abstract: int):
    store, corpus = PaperStore(db), Corpus(db, "run_test")
    ids = {name: store.upsert(record) for name, record in papers.items()}
    for paper_id in ids.values():
        corpus.add(paper_id, {"channel": "s2_search", "rank": 1})
    llm = fake_llm(settings, model)
    ctx = SimpleNamespace(
        db=db, embedder=FakeEmbedder(), llm=llm,
        settings=settings.model_copy(update={"full_read": full, "abstract_only": abstract}),
    )
    brief = write_brief(llm, "low-rank adaptation of language models", JudgmentStore(db, llm.models(Role.REASONING)))
    with recorder.step("run_test", "rank") as event:
        rank_stage({"run_id": "run_test", "brief": brief.model_dump()}, ctx, event)
    ranked = {row["paper_id"]: row for row in corpus.papers()}
    return {name: ranked[paper_id] for name, paper_id in ids.items()}


def test_rank_breaks_relevance_ties_by_citation_rate(db, recorder, settings):
    same = "Low-rank adaptation of language models"
    rows = _rank(db, recorder, settings, {
        "cited": PaperRecord(source="s2", title=same + " A", abstract=same, year=2021, citation_count=5000),
        "uncited": PaperRecord(source="s2", title=same + " B", abstract=same, year=2021, citation_count=3),
        "unknown": PaperRecord(source="arxiv", title=same + " C", abstract=same, year=2021),
    }, FakeModel(), full=1, abstract=1)

    assert sorted(rows, key=lambda name: rows[name]["rank"]) == ["cited", "unknown", "uncited"]  # unknown = median
    assert rows["cited"]["read_level"] == "full"
    assert "5000 citations" in rows["cited"]["reason"]
    assert rows["uncited"]["read_level"] == "excluded"


def test_relevance_check_drops_off_topic_papers_and_fills_the_quota_from_below(db, recorder, settings):
    same = "Low-rank adaptation of language models"
    model = FakeModel(off_topic=("Famous",))
    rows = _rank(db, recorder, settings, {
        "famous": PaperRecord(source="s2", title="Famous Vision Paper", abstract=same, year=2021, citation_count=90000),
        "good": PaperRecord(source="s2", title=same + " A", abstract=same, year=2021, citation_count=500),
        "also_good": PaperRecord(source="s2", title=same + " B", abstract=same, year=2021, citation_count=5),
    }, model, full=1, abstract=1)

    # It outranks a paper that is kept: without the check it would have taken a place.
    assert rows["famous"]["rank"] < rows["also_good"]["rank"]
    assert rows["famous"]["read_level"] == "excluded"
    assert rows["famous"]["relevance"] == "off_topic"
    assert "judged off-topic by the relevance check" in rows["famous"]["reason"]
    assert (rows["good"]["read_level"], rows["also_good"]["read_level"]) == ("full", "abstract")
    assert rows["good"]["relevance"] == "core" and "core: fake" in rows["good"]["reason"]
    screening = [call for call in model.calls if call["output"] == "Judgements"]
    assert len(screening) == 1 and "] Famous Vision Paper (2021)\nAbstract: Low-rank" in screening[0]["prompt"]
