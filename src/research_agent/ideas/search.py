"""The novelty search: a fresh search for work close to one idea, never only the
corpus that inspired it (after the Idea Novelty Checker, Shahid et al. 2025).

Round 1 gathers candidates through several channels: keyword queries and
likely paper titles that the extraction model writes for the idea (Semantic
Scholar, OpenAlex, arXiv), Semantic Scholar snippet search on the whole idea
(passages inside papers, which catches work that words the idea differently),
and the run's own corpus. An embedding filter keeps the FILTER_KEEP most
similar, and the extraction model reranks those by facet match: papers that
may already be the idea first, then same purpose and domain, then one shared
facet. In the Novelty Checker's ablation, ranking by plain relevance instead
dropped duplicate detection from 90% to 14%.

Round 2 adds recommendations for the RECOMMEND_SEEDS closest papers (Semantic
Scholar, or OpenAlex related works without a key), then filters and reranks
again down to the CLOSEST papers. How many round-2 papers reach the closest
work says whether the search had settled; that, the candidate count and the
channels that answered make up the search's coverage, which the novelty level
takes into account.

Two filters: SPECTER2 (local, paper-level) and Gemini Embedding 2 (each paper
text is a request on the free tier, kept in the embedding cache). The pipeline
uses FILTER, SPECTER2, chosen on the planted-duplicate test (2026-10-11): it kept
all 21 source papers in its top FILTER_KEEP, ranking 17 of them first, and on the
4 it didn't rank first its median rank for the source was 3 against Gemini's 15.
Gemini would also need about 170 embedding requests per idea checked.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from research_agent.clients.openalex import CS_FILTER
from research_agent.embeddings import EmbeddingCache
from research_agent.graph.store import JudgmentStore, digest
from research_agent.ideas.ideas import Idea
from research_agent.literature.brief import AngleDraft, make_angle, question_angle
from research_agent.literature.corpus import Corpus
from research_agent.literature.papers import PaperRecord, PaperStore, from_arxiv, from_openalex, from_s2
from research_agent.literature.stages import openalex_key, s2_key
from research_agent.llm import LLMOutputError, Prompt, Role
from research_agent.trace import Event, ref

if TYPE_CHECKING:
    from research_agent.pipeline.graph import PipelineContext

KEYWORD_RESULTS = 20  # per keyword query, on Semantic Scholar and on OpenAlex
TITLE_RESULTS = 10
ARXIV_RESULTS = 15
SNIPPET_RESULTS = 20
SNIPPET_CHARS = 1000  # of the idea, sent as the snippet query
RECOMMEND_SEEDS = 3
RECOMMENDATIONS = 20
FILTER_KEEP = 30
CLOSEST = 10
ABSTRACT_CHARS = 1200  # of each candidate's abstract, shown to the reranker
# Coverage: fewer candidates than this, or this many round-2 papers among the closest
# work, means the search may have missed close work.
MIN_POOL = 30
UNSETTLED = 3

Filter = Literal["specter2", "gemini"]
FILTER: Filter = "specter2"  # see the planted-duplicate test (scripts/duplicate_test.py)
MATCH_ORDER = ("all_facets", "purpose_and_domain", "one_facet", "partial")

Found = list[tuple[PaperRecord | None, dict[str, Any]]]


# Queries --------------------------------------------------------------------------------

class QueryTerms(BaseModel):
    terms: list[str] = Field(
        min_length=1, max_length=5,
        description="2 to 4 search terms, most specific first; each 1 to 3 words, worded as paper titles and abstracts word it",
    )


class NoveltyQueries(BaseModel):
    """The extraction model's reply."""

    keywords: list[QueryTerms] = Field(min_length=1, max_length=4, description="3 keyword queries")
    titles: list[str] = Field(max_length=3, description="2 titles that a paper presenting this idea might have")


_QUERIES_SYSTEM = """You write literature searches that look for existing work which may already describe a research idea in computer science or machine learning.

Given the idea, write:
- keywords: 3 keyword queries. The first names the idea's mechanism applied to its purpose; the second names its purpose, using the field's usual name for the problem; the third names the mechanism alone, under its established name.
- titles: 2 titles that a paper presenting this idea, or something very close to it, might have.

Use the established names of tasks, methods and benchmarks, as paper titles and abstracts use them. Do not invent results."""


def queries_prompt(idea: Idea) -> Prompt:
    return Prompt("novelty_queries", 1, _QUERIES_SYSTEM, "\n".join(idea.block()))


def write_queries(ctx: "PipelineContext", judgments: JudgmentStore, idea: Idea) -> tuple[NoveltyQueries, dict[str, Any]]:
    """The model's queries; the idea title's own words if it gives no valid reply."""
    prompt = queries_prompt(idea)
    input_hash = digest(prompt.id, prompt.system, prompt.user)
    kept = judgments.find("novelty_queries", input_hash)
    if kept is not None:
        return NoveltyQueries.model_validate(kept), {"reused": True}
    try:
        answer = ctx.llm.answer(Role.EXTRACTION, prompt, NoveltyQueries)
    except LLMOutputError as exc:
        fallback = question_angle(idea.title).keywords[:4] or [idea.title]
        return NoveltyQueries(keywords=[QueryTerms(terms=fallback)], titles=[idea.title]), {"error": str(exc)}
    judgments.add("novelty_queries", input_hash, answer.model, answer.output.model_dump())
    return answer.output, {"reused": False}


# The candidate pool -----------------------------------------------------------------------

@dataclass
class Candidate:
    paper_id: str
    title: str
    abstract: str | None
    year: int | None
    found_by: list[dict[str, Any]] = field(default_factory=list)
    similarity: float | None = None
    position: int | None = None
    facet_match: str | None = None

    def rounds(self) -> set[int]:
        return {f["round"] for f in self.found_by}

    def channels(self) -> set[str]:
        return {f["channel"] for f in self.found_by}


@dataclass
class Search:
    """One idea's novelty search: the pool, the channel log, and the SPECTER2 vectors so far."""

    idea: Idea
    filter: Filter
    pool: dict[str, Candidate] = field(default_factory=dict)
    log: list[dict[str, Any]] = field(default_factory=list)
    vectors: dict[str, np.ndarray] = field(default_factory=dict)
    filter_info: list[dict[str, Any]] = field(default_factory=list)
    rerank_info: list[dict[str, Any]] = field(default_factory=list)
    closest: list[Candidate] = field(default_factory=list)


def _snippets(ctx: "PipelineContext", query: str) -> Found:
    s2 = ctx.sources.s2
    snippets = [s for s in s2.snippets(query, limit=SNIPPET_RESULTS) if s.corpus_id]
    papers = s2.batch([f"CorpusId:{s.corpus_id}" for s in snippets]) if snippets else []
    return [(from_s2(paper) if paper else None, {"section": snippet.section})
            for snippet, paper in zip(snippets, papers)]


def _gather(ctx: "PipelineContext", search: Search, round_: int,
            channels: list[tuple[str, str, Callable[[str], Found]]]) -> None:
    """Run each channel and add what it finds to the pool. A failing channel is logged and skipped."""
    store = PaperStore(ctx.db)
    for name, query, fetch in channels:
        if not query:
            continue
        try:
            results = fetch(query)
        except Exception as exc:
            search.log.append({"round": round_, "channel": name, "query": query, "error": f"{type(exc).__name__}: {exc}"[:300]})
            continue
        ids = []
        for rank, (record, extra) in enumerate(results, start=1):
            if record is None:
                continue
            paper_id = store.upsert(record)
            _add(ctx, search, paper_id, {"channel": name.split(":")[0], "round": round_, "rank": rank, **extra})
            ids.append(paper_id)
        search.log.append({"round": round_, "channel": name, "query": query, "results": len(ids)})


def _add(ctx: "PipelineContext", search: Search, paper_id: str, found: dict[str, Any]) -> None:
    if paper_id not in search.pool:
        row = ctx.db.query("SELECT title, abstract, year FROM papers WHERE paper_id = ?", (paper_id,))[0]
        search.pool[paper_id] = Candidate(paper_id, row["title"], row["abstract"], row["year"])
    search.pool[paper_id].found_by.append(found)


def round_one(ctx: "PipelineContext", search: Search, queries: NoveltyQueries, run_id: str | None) -> None:
    sources, idea = ctx.sources, search.idea
    channels: list[tuple[str, str, Callable[[str], Found]]] = []
    for i, query in enumerate(queries.keywords):
        angle = make_angle(AngleDraft(label=f"query {i}", keywords=query.terms, passage=idea.title))
        if not angle.keywords:
            continue
        channels += [
            (f"openalex_search:{i}", angle.keyword_query,
             lambda q: [(from_openalex(w), {}) for w in sources.openalex.search(q, per_page=KEYWORD_RESULTS, filter=CS_FILTER)]),
            (f"arxiv_search:{i}", angle.arxiv_query,
             lambda q: [(from_arxiv(e), {}) for e in sources.arxiv.search(q, max_results=ARXIV_RESULTS)]),
        ]
        if sources.s2:
            channels.append((f"s2_search:{i}", angle.keyword_query,
                             lambda q: [(from_s2(p), {}) for p in sources.s2.search(q, limit=KEYWORD_RESULTS)]))
    for j, title in enumerate(queries.titles):
        if sources.s2:
            channels.append((f"s2_title:{j}", title,
                             lambda q: [(from_s2(p), {}) for p in sources.s2.search(q, limit=TITLE_RESULTS)]))
        else:
            channels.append((f"openalex_title:{j}", title,
                             lambda q: [(from_openalex(w), {}) for w in sources.openalex.search(q, per_page=TITLE_RESULTS, filter=CS_FILTER)]))
    if sources.s2:
        channels.append(("s2_snippets:0", idea.text()[:SNIPPET_CHARS], lambda q: _snippets(ctx, q)))
    _gather(ctx, search, 1, channels)
    if run_id:
        corpus = [row for row in Corpus(ctx.db, run_id).papers() if row["read_level"] in ("full", "abstract")]
        for row in corpus:
            _add(ctx, search, row["paper_id"], {"channel": "corpus", "round": 1, "rank": row["rank"]})
        search.log.append({"round": 1, "channel": "corpus", "query": run_id, "results": len(corpus)})


def round_two(ctx: "PipelineContext", search: Search, seeds: list[Candidate]) -> None:
    sources = ctx.sources
    channels: list[tuple[str, str, Callable[[str], Found]]] = []
    for seed in seeds:
        row = ctx.db.query("SELECT * FROM papers WHERE paper_id = ?", (seed.paper_id,))[0]
        if sources.s2:
            channels.append((f"s2_recommendations:{seed.paper_id}", s2_key(row) or "",
                             lambda k: [(from_s2(p), {}) for p in sources.s2.recommendations(k, limit=RECOMMENDATIONS)]))
        else:
            channels.append((f"openalex_related:{seed.paper_id}", openalex_key(row), lambda k: _openalex_related(ctx, k)))
    _gather(ctx, search, 2, channels)


def _openalex_related(ctx: "PipelineContext", key: str) -> Found:
    work = ctx.sources.openalex.work(key)
    if work is None:
        return []
    return [(from_openalex(w), {}) for w in ctx.sources.openalex.works(work.related_works[:RECOMMENDATIONS])]


# Filtering ---------------------------------------------------------------------------------

def paper_text(candidate: Candidate) -> str:
    return f"{candidate.title}. {candidate.abstract or ''}".strip()


def similarities(ctx: "PipelineContext", search: Search, method: Filter) -> dict[str, Any]:
    """Score every pool paper against the idea; returns what was used. Gemini falls back
    to SPECTER2 when any vector is missing (its quota used up), so scores never mix."""
    candidates = list(search.pool.values())
    info: dict[str, Any] = {"method": method, "papers": len(candidates)}
    if method == "gemini" and ctx.text_embedder is not None:
        texts = [search.idea.text(), *(paper_text(c) for c in candidates)]
        vectors, report = EmbeddingCache(ctx.db, ctx.text_embedder).vectors(texts)
        info["embedding"] = report
        if all(text in vectors for text in texts):
            query = vectors[texts[0]]
            for candidate, text in zip(candidates, texts[1:]):
                candidate.similarity = float(vectors[text] @ query)
            return info
        info["fallback"] = "specter2: some vectors are missing"
    elif method == "gemini":
        info["fallback"] = "specter2: no embedding model set (RA_EMBEDDING)"
    new = [c for c in candidates if c.paper_id not in search.vectors]
    if new:
        vectors = ctx.embedder.papers([(c.title, c.abstract) for c in new])
        search.vectors.update({c.paper_id: v for c, v in zip(new, vectors)})
    query = ctx.embedder.papers([(search.idea.title, " ".join((search.idea.purpose, search.idea.mechanism, search.idea.evaluation)))])[0]
    for candidate in candidates:
        candidate.similarity = float(search.vectors[candidate.paper_id] @ query)
    info["method_used"] = "specter2"
    return info


def filtered(search: Search) -> list[Candidate]:
    return sorted(search.pool.values(), key=lambda c: (-(c.similarity or 0.0), c.paper_id))[:FILTER_KEEP]


# Facet reranking ----------------------------------------------------------------------------

class CloseMatch(BaseModel):
    paper: int = Field(description="The candidate's number")
    match: Literal["all_facets", "purpose_and_domain", "one_facet", "partial"]


class CloseWork(BaseModel):
    """The extraction model's reply."""

    papers: list[CloseMatch] = Field(description=f"The {CLOSEST} closest candidates, closest first")


_RERANK_SYSTEM = f"""You rank candidate papers by how closely they match a proposed research idea in computer science or machine learning, to find existing work that may already do it.

The idea has a purpose (the problem it addresses), a mechanism (how it works), an evaluation (how it is tested) and an application domain. Rank by this order of priority:
1. all_facets: the paper addresses the same purpose with the same or a very similar mechanism; it may already be the idea.
2. purpose_and_domain: the same purpose in the same application domain, with a different mechanism.
3. one_facet: the paper shares the purpose, the mechanism or the evaluation.
4. partial: related only loosely.
Return the {CLOSEST} closest candidates, closest first, each with its number and match level. Judge only from the titles and abstracts given."""


def rerank_prompt(idea: Idea, candidates: list[Candidate]) -> Prompt:
    lines = ["Idea:", *idea.block(), "", "Candidates:"]
    for number, c in enumerate(candidates, start=1):
        year = f" ({c.year})" if c.year else ""
        abstract = (c.abstract or "no abstract available")[:ABSTRACT_CHARS]
        lines += ["", f"[{number}] {c.title}{year}", f"Abstract: {abstract}"]
    return Prompt("novelty_rerank", 1, _RERANK_SYSTEM, "\n".join(lines))


def rerank(ctx: "PipelineContext", judgments: JudgmentStore, idea: Idea, candidates: list[Candidate]) -> tuple[list[Candidate], dict[str, Any]]:
    """The CLOSEST candidates by facet match (stable within a match level: the model's order).
    Missing places are filled in filter order, without a match level."""
    if not candidates:
        return [], {"candidates": 0}
    prompt = rerank_prompt(idea, candidates)
    input_hash = digest(prompt.id, prompt.system, prompt.user)
    kept = judgments.find("novelty_rerank", input_hash)
    info: dict[str, Any] = {"candidates": len(candidates), "reused": kept is not None}
    if kept is None:
        try:
            answer = ctx.llm.answer(Role.EXTRACTION, prompt, CloseWork)
            kept = answer.output.model_dump()["papers"]
            judgments.add("novelty_rerank", input_hash, answer.model, kept)
        except LLMOutputError as exc:
            kept, info["error"] = [], str(exc)
    picked: list[tuple[int, Candidate, str]] = []
    seen: set[int] = set()
    for order, item in enumerate(kept):
        number = item["paper"]
        if 1 <= number <= len(candidates) and number not in seen and len(picked) < CLOSEST:
            seen.add(number)
            picked.append((order, candidates[number - 1], item["match"]))
    picked.sort(key=lambda p: (MATCH_ORDER.index(p[2]), p[0]))
    closest = [c for _, c, _ in picked]
    for candidate, (_, _, match) in zip(closest, picked):
        candidate.facet_match = match
    filled = [c for c in candidates if c not in closest][: CLOSEST - len(closest)]
    for candidate in filled:
        candidate.facet_match = None
    info.update(ranked=len(picked), filled=len(filled))
    return closest + filled, info


# The whole search ------------------------------------------------------------------------------

def run_search(ctx: "PipelineContext", idea: Idea, run_id: str | None, method: Filter, event: Event) -> Search:
    judgments = JudgmentStore(ctx.db, ctx.llm.models(Role.EXTRACTION))
    search = Search(idea, method)
    queries, info = write_queries(ctx, judgments, idea)
    search.log.append({"round": 1, "channel": "queries", "query": json.dumps(queries.model_dump()), **info})
    round_one(ctx, search, queries, run_id)
    search.filter_info.append(similarities(ctx, search, method))
    first, rerank_info = rerank(ctx, judgments, idea, filtered(search))
    search.rerank_info.append(rerank_info)
    round_two(ctx, search, first[:RECOMMEND_SEEDS])
    search.filter_info.append(similarities(ctx, search, method))
    search.closest, rerank_info = rerank(ctx, judgments, idea, filtered(search))
    search.rerank_info.append(rerank_info)
    for position, candidate in enumerate(search.closest, start=1):
        candidate.position = position
    store_candidates(ctx, search)
    kept = {c.paper_id for c in filtered(search)}
    for candidate in search.pool.values():
        if candidate.position is None:
            reason = "not among the closest after facet reranking" if candidate.paper_id in kept else (
                f"below the top {FILTER_KEEP} by similarity ({candidate.similarity:.3f})")
            event.drop(ref("paper", candidate.paper_id), f"idea {idea.idea_id}: {reason}")
    event.generated += [ref("paper", c.paper_id) for c in search.closest]
    return search


def store_candidates(ctx: "PipelineContext", search: Search) -> None:
    with ctx.db.transaction() as conn:
        conn.execute("DELETE FROM idea_candidates WHERE idea_id = ?", (search.idea.idea_id,))
        conn.executemany(
            "INSERT INTO idea_candidates (idea_id, paper_id, found_by, similarity, position, facet_match)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            [(search.idea.idea_id, c.paper_id, json.dumps(c.found_by), c.similarity, c.position, c.facet_match)
             for c in search.pool.values()],
        )


def coverage(search: Search, s2: bool) -> dict[str, Any]:
    """How well the search covered the literature, and what makes it weak."""
    answered = sorted({entry["channel"].split(":")[0] for entry in search.log if entry.get("results")})
    closest = search.closest
    round_two_only = [c for c in closest if c.rounds() == {2}]
    weak = []
    if not s2:
        weak.append("no Semantic Scholar key: no snippet search or recommendations")
    if len(search.pool) < MIN_POOL:
        weak.append(f"only {len(search.pool)} candidate papers found")
    if len(round_two_only) >= UNSETTLED:
        weak.append(f"the recommendations round still brought {len(round_two_only)} new papers into the closest {CLOSEST}")
    errors = [entry["channel"] for entry in search.log if "error" in entry and entry["channel"] != "queries"]
    return {
        "s2": s2,
        "filter": search.filter_info[-1].get("method_used", search.filter_info[-1]["method"]) if search.filter_info else None,
        "candidates": len(search.pool),
        "round_one_candidates": sum(1 for c in search.pool.values() if 1 in c.rounds()),
        "channels_answered": answered,
        "channel_errors": errors,
        "closest_found_by_several_channels": sum(1 for c in closest if len(c.channels()) > 1),
        "closest_from_recommendations_only": len(round_two_only),
        "weak": weak,
    }
