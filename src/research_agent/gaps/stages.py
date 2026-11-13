"""The Phase 3 stages: ``related_work``, ``landscape`` and ``gaps``."""

from typing import TYPE_CHECKING, Any

from research_agent.embeddings import EmbeddingCache
from research_agent.gaps.gaps import (
    SHORTLIST,
    TOP_GAPS,
    conflict_candidates,
    merge_duplicates,
    rank,
    score_relevance,
    stated_candidates,
    store_gaps,
    structural_candidates,
    write_up,
)
from research_agent.gaps.landscape import build_landscape
from research_agent.gaps.related import related_papers
from research_agent.graph.build import load_graph
from research_agent.graph.store import JudgmentStore
from research_agent.literature.corpus import Corpus
from research_agent.llm import Role
from research_agent.storage import parse_sentence_id
from research_agent.trace import Event, ref

if TYPE_CHECKING:
    from research_agent.pipeline.graph import PipelineContext, PipelineState


def _question(state: "PipelineState") -> str:
    """What papers and gaps are compared with: the question as written, as in ranking."""
    brief = state.get("brief")
    return brief["rank_query"] if brief else " ".join(state["question"].split())


def related_work_stage(state: "PipelineState", ctx: "PipelineContext", event: Event) -> dict[str, Any]:
    """The corpus papers closest to the question, by similarity and graph neighbourhood."""
    rows = [row for row in Corpus(ctx.db, state["run_id"]).papers() if row["read_level"] in ("full", "abstract")]
    if not rows:
        return {"related_work": []}
    query = ctx.embedder.query(_question(state))
    vectors = ctx.embedder.papers([(row["title"], row["abstract"]) for row in rows])
    similarity = {row["paper_id"]: float(score) for row, score in zip(rows, vectors @ query)}
    related = related_papers(load_graph(ctx.db, state["run_id"]), similarity)
    event.used += [ref("paper", row["paper_id"]) for row in rows]
    event.generated += [ref("paper", paper["paper_id"]) for paper in related]
    event.detail["related_work"] = related
    return {"related_work": related}


def landscape_stage(state: "PipelineState", ctx: "PipelineContext", event: Event) -> dict[str, Any]:
    """The most-studied methods x datasets, and what the corpus holds for each pair."""
    landscape = build_landscape(load_graph(ctx.db, state["run_id"]))
    event.used.append(ref("graph", state["run_id"]))
    counts: dict[str, int] = {}
    for row in landscape["cells"]:
        for cell in row:
            counts[cell["status"]] = counts.get(cell["status"], 0) + 1
    event.detail["landscape"] = {"methods": len(landscape["methods"]), "datasets": len(landscape["datasets"]), "cells": counts}
    return {"landscape": landscape}


def gaps_stage(state: "PipelineState", ctx: "PipelineContext", event: Event) -> dict[str, Any]:
    """Collect gap candidates, rank them, have the reasoning model write up the best, store the top ones."""
    run_id, question = state["run_id"], _question(state)
    graph = load_graph(ctx.db, run_id)
    judgments = JudgmentStore(ctx.db, ctx.llm.models(Role.REASONING))
    landscape = state.get("landscape") or build_landscape(graph)

    stated = stated_candidates(graph, ctx.db)
    structural, cells = structural_candidates(ctx.llm, judgments, ctx.db, run_id, graph, landscape, question)
    conflicts = conflict_candidates(graph, ctx.db)
    candidates = [c for c in stated + structural + conflicts if c.evidence]
    vectors: dict[str, Any] = {}
    embedding: dict[str, Any] = {"status": "no embedding model set (RA_EMBEDDING): SPECTER2 used, no merging"}
    if ctx.text_embedder is not None and candidates:
        vectors, embedding = EmbeddingCache(ctx.db, ctx.text_embedder).vectors([question, *(c.text for c in candidates)])
    relevance_by = score_relevance(candidates, ctx.embedder, question, vectors)
    merged = 0
    if relevance_by == "text embeddings":
        candidates, merged = merge_duplicates(candidates, vectors)
    ranked = rank(candidates)
    shortlist = ranked[:SHORTLIST]
    for candidate in ranked[SHORTLIST:]:
        event.drop(f"gap_candidate:{candidate.sources[0]}", f"ranked below the top {SHORTLIST}: {candidate.text[:150]}")

    papers = Corpus(ctx.db, run_id).papers()
    titles = {row["paper_id"]: row["title"] for row in papers}
    shown = {row["paper_id"]: f"{row['title']}, {row['year']}" if row["year"] else row["title"] for row in papers}
    gaps, writeup = write_up(ctx.llm, judgments, question, shortlist, shown)
    top = gaps[:TOP_GAPS]
    node_ids = store_gaps(ctx.db, run_id, top)

    event.used.append(ref("graph", run_id))
    event.generated += [ref("node", node_id) for node_id in node_ids]
    event.detail["gaps"] = {
        "candidates": {"stated": len(stated), "structural": len(structural), "conflict": len(conflicts)},
        "relevance_by": relevance_by,
        "merged_by_similarity": merged,
        "embedding": embedding,
        "structural_cells": cells,
        "writeup": writeup,
        "kept": len(gaps),
    }
    records = []
    for rank_, (gap, node_id) in enumerate(zip(top, node_ids), start=1):
        c = gap.candidate
        records.append({
            "rank": rank_, "node_id": node_id, "kind": c.kind, "gap": gap.gap, "warrant": gap.warrant,
            "relevance": round(c.relevance, 4), "papers": len(c.papers),
            "evidence": [
                {"sentence_id": e["sentence"], "quote": e["quote"],
                 "paper": titles.get(parse_sentence_id(e["sentence"])[0], "")}
                for e in c.evidence
            ],
        })
    return {"gaps": records}
