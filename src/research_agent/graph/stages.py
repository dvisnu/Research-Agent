"""The Phase 2 stages: ``extract`` and ``build_graph``."""

import json
from typing import TYPE_CHECKING, Any

from research_agent.graph.build import build_run_graph
from research_agent.graph.extraction import (
    ABSTRACT_BATCH,
    AbstractExtractions,
    AbstractItem,
    ChunkExtraction,
    OpeningChunkExtraction,
    abstract_block,
    abstract_item,
    abstract_prompt,
    abstract_resolver,
    chunk_prompt,
    chunk_sections,
    full_resolver,
    load_sections,
    resolve_addresses,
)
from research_agent.graph.store import ExtractionStore, digest
from research_agent.literature.corpus import Corpus
from research_agent.llm import LLMOutputError, Role
from research_agent.trace import Event, ref

if TYPE_CHECKING:
    from research_agent.pipeline.graph import PipelineContext, PipelineState


def extract_stage(state: "PipelineState", ctx: "PipelineContext", event: Event) -> dict[str, Any]:
    """Extract every corpus paper: full-read papers chunk by chunk, abstract-read
    papers several to a call. Extractions kept from earlier runs are reused."""
    run_id = state["run_id"]
    corpus = Corpus(ctx.db, run_id)
    ctx.db.execute("DELETE FROM run_extractions WHERE run_id = ?", (run_id,))
    found, counts = extract_papers(ctx, corpus.papers("full"), corpus.papers("abstract"), event)
    used = [extraction_id for ids in found.values() for extraction_id in ids]
    with ctx.db.transaction() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO run_extractions (run_id, extraction_id) VALUES (?, ?)",
            [(run_id, extraction_id) for extraction_id in used],
        )
    event.generated += [ref("extraction", extraction_id) for extraction_id in used]
    event.detail["extraction"] = counts
    return {"extraction": counts}


def extract_papers(
    ctx: "PipelineContext", full: list[Any], abstract: list[Any], event: Event,
    chunks: dict[str, set[int]] | None = None,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Extract papers (rows with paper_id, title and year): ``full`` ones chunk by chunk,
    only the chunks ``chunks`` names for a paper listed there, and ``abstract`` ones
    several to a call. Extractions kept from earlier are reused. Returns each paper's
    extraction IDs, in reading order, and what was done."""
    store = ExtractionStore(ctx.db, ctx.llm.models(Role.EXTRACTION))
    counts = {"full_papers": 0, "chunks": 0, "abstracts": 0, "calls": 0, "reused": 0, "failed": 0}
    found: dict[str, list[str]] = {}

    def keep(paper_id: str, extraction_id: str) -> None:
        found.setdefault(paper_id, []).append(extraction_id)

    for row in full:
        paper_id = row["paper_id"]
        event.used.append(ref("paper", paper_id))
        all_chunks = chunk_sections(load_sections(ctx.db, paper_id))
        if not all_chunks:
            event.drop(ref("paper", paper_id), "no text to extract from")
            continue
        counts["full_papers"] += 1
        wanted = (chunks or {}).get(paper_id)
        for chunk in all_chunks:
            if wanted is not None and chunk.index not in wanted:
                continue
            counts["chunks"] += 1
            prompt = chunk_prompt(row["title"], chunk, len(all_chunks))
            input_hash = digest(prompt.id, prompt.user)
            kept = store.find(input_hash)
            if kept is not None:
                counts["reused"] += 1
                keep(paper_id, kept["extraction_id"])
                continue
            output_type = OpeningChunkExtraction if chunk.index == 0 else ChunkExtraction
            try:
                answer = ctx.llm.answer(Role.EXTRACTION, prompt, output_type)
            except LLMOutputError as exc:
                counts["failed"] += 1
                event.drop(ref("chunk", f"{paper_id}#{chunk.index}"), f"no valid reply: {exc}")
                continue
            counts["calls"] += 1
            output = resolve_addresses(answer.output.model_dump(), full_resolver(paper_id))
            keep(paper_id, store.add(
                paper_id=paper_id, kind="full", chunk=chunk.index, sections=chunk.sections, input_hash=input_hash,
                prompt_version=prompt.id, model=answer.model, output=output,
            ))

    pending: list[tuple[AbstractItem, str]] = []
    for row in abstract:
        paper_id = row["paper_id"]
        event.used.append(ref("paper", paper_id))
        item = abstract_item(paper_id, row["title"], row["year"], load_sections(ctx.db, paper_id))
        if item is None:
            event.drop(ref("paper", paper_id), "no abstract to extract from")
            continue
        counts["abstracts"] += 1
        input_hash = digest(abstract_prompt([]).id, *abstract_block(0, item))
        kept = store.find(input_hash)
        if kept is not None:
            counts["reused"] += 1
            keep(paper_id, kept["extraction_id"])
        else:
            pending.append((item, input_hash))

    for start in range(0, len(pending), ABSTRACT_BATCH):
        batch = pending[start : start + ABSTRACT_BATCH]
        prompt = abstract_prompt([item for item, _ in batch])
        try:
            answer = ctx.llm.answer(Role.EXTRACTION, prompt, AbstractExtractions)
        except LLMOutputError as exc:
            counts["failed"] += len(batch)
            for item, _ in batch:
                event.drop(ref("paper", item.paper_id), f"no valid abstract extraction: {exc}")
            continue
        counts["calls"] += 1
        replies = {reply.paper: reply for reply in answer.output.papers}
        for number, (item, input_hash) in enumerate(batch, start=1):
            reply = replies.get(number)
            if reply is None:
                counts["failed"] += 1
                event.drop(ref("paper", item.paper_id), "the abstract extraction left this paper out")
                continue
            output = resolve_addresses(reply.model_dump(exclude={"paper"}), abstract_resolver(number, item))
            keep(item.paper_id, store.add(
                paper_id=item.paper_id, kind="abstract", chunk=0, sections=[item.section], input_hash=input_hash,
                prompt_version=prompt.id, model=answer.model, output=output,
            ))
    return found, counts


def build_graph_stage(state: "PipelineState", ctx: "PipelineContext", event: Event) -> dict[str, Any]:
    """Quote-check the extractions, merge names, link findings across papers, store the graph."""
    summary = build_run_graph(ctx.db, ctx.llm, ctx.text_embedder, state["run_id"], event)
    event.detail["graph"] = json.loads(json.dumps(summary, default=str))
    return {"graph": summary}
