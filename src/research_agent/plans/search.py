"""The targeted search: literature for what the critique found missing.

Each query (at most MAX_SEARCHES a round, see ``critique``) goes to Semantic Scholar
keyword search and snippet search (passages inside papers, which is where reported
results are), or without a key to OpenAlex and arXiv. Papers not yet among the plan
papers are ranked, snippet hits first since their passage matched the query, then by
SPECTER2 similarity to it, and the best KEEP_PER_QUERY become plan papers (role
"search"). Each is extracted by its abstract; a paper whose snippet came from its body
is also read in full and the chunk holding the snippet extracted, one call, since its
results are there rather than in the abstract.
"""

import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from research_agent.clients.openalex import CS_FILTER
from research_agent.graph.extraction import chunk_sections, load_sections
from research_agent.graph.stages import extract_papers
from research_agent.ideas.compare import ensure_abstract, read_in_full
from research_agent.literature.brief import question_angle
from research_agent.literature.papers import PaperRecord, PaperStore, from_arxiv, from_openalex, from_s2
from research_agent.literature.stages import is_full_text
from research_agent.plans.facts import add_plan_paper, load_plan_papers
from research_agent.trace import Event, ref

if TYPE_CHECKING:
    from research_agent.pipeline.graph import PipelineContext

SEARCH_RESULTS = 8  # per channel and query
KEEP_PER_QUERY = 3
_NOT_BODY = {"", "abstract", "title"}

Found = list[tuple[PaperRecord | None, dict[str, Any]]]


def _snippets(ctx: "PipelineContext", query: str) -> Found:
    s2 = ctx.sources.s2
    snippets = [s for s in s2.snippets(query, limit=SEARCH_RESULTS) if s.corpus_id]
    papers = s2.batch([f"CorpusId:{s.corpus_id}" for s in snippets]) if snippets else []
    return [(from_s2(paper) if paper else None, {"snippet": snippet.text, "section": snippet.section})
            for snippet, paper in zip(snippets, papers)]


def _channels(ctx: "PipelineContext", query: str) -> list[tuple[str, Callable[[], Found]]]:
    sources = ctx.sources
    if sources.s2:
        return [("s2_search", lambda: [(from_s2(p), {}) for p in sources.s2.search(query, limit=SEARCH_RESULTS)]),
                ("s2_snippets", lambda: _snippets(ctx, query))]
    angle = question_angle(query)
    return [("openalex_search", lambda: [(from_openalex(w), {}) for w in
                                         sources.openalex.search(angle.keyword_query, per_page=SEARCH_RESULTS, filter=CS_FILTER)]),
            ("arxiv_search", lambda: [(from_arxiv(e), {}) for e in sources.arxiv.search(angle.arxiv_query, max_results=SEARCH_RESULTS)])]


def _norm(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def snippet_chunk(ctx: "PipelineContext", paper_id: str, snippet: str) -> int | None:
    """The chunk of the paper's stored text that holds the snippet: the most shared runs of 4 words."""
    words = _norm(snippet).split()
    grams = {" ".join(words[i : i + 4]) for i in range(len(words) - 3)}
    best, best_score = None, 0
    for chunk in chunk_sections(load_sections(ctx.db, paper_id)):
        text = f" {_norm(' '.join(line.text for line in chunk.lines))} "
        score = sum(f" {gram} " in text for gram in grams)
        if score > best_score:
            best, best_score = chunk.index, score
    return best


def run_searches(
    ctx: "PipelineContext", run_id: str, searches: list[dict[str, Any]], round_: int, event: Event
) -> list[dict[str, Any]]:
    """Run the critique's searches and add the papers they find to the plan papers, extracted."""
    store = PaperStore(ctx.db)
    known = set(load_plan_papers(ctx.db, run_id))
    position = 1 + max((info.get("position") or 0 for paper in load_plan_papers(ctx.db, run_id).values()
                        for role, info in paper.roles.items() if role == "search"), default=0)
    log: list[dict[str, Any]] = []
    added: dict[str, dict[str, Any]] = {}
    for search in searches:
        query = search["query"]
        entry: dict[str, Any] = {"query": query, "point": search.get("point"), "channels": {}, "kept": []}
        found: dict[str, dict[str, Any]] = {}
        for channel, fetch in _channels(ctx, query):
            try:
                results = fetch()
            except Exception as exc:  # one failing channel must not stop the search
                entry["channels"][channel] = f"{type(exc).__name__}: {exc}"[:200]
                continue
            entry["channels"][channel] = sum(1 for record, _ in results if record is not None)
            for rank, (record, extra) in enumerate(results, start=1):
                if record is None:
                    continue
                paper_id = store.upsert(record)
                hit = found.setdefault(paper_id, {"channels": [], "rank": rank})
                hit["channels"].append(channel)
                if extra.get("snippet") and "snippet" not in hit:
                    hit |= {"snippet": extra["snippet"], "section": extra.get("section")}
        fresh = [p for p in found if p not in known and p not in added]
        if fresh:
            rows = {r["paper_id"]: r for r in ctx.db.query(
                f"SELECT paper_id, title, abstract FROM papers WHERE paper_id IN ({','.join('?' * len(fresh))})", tuple(fresh))}
            vectors = ctx.embedder.papers([(rows[p]["title"], rows[p]["abstract"]) for p in fresh])
            similarity = dict(zip(fresh, (vectors @ ctx.embedder.query(query)).tolist()))
            ranked = sorted(fresh, key=lambda p: ("snippet" not in found[p], -similarity[p], p))
            for paper_id in ranked[:KEEP_PER_QUERY]:
                hit = found[paper_id]
                added[paper_id] = {"query": query, "round": round_, "point": search.get("point"),
                                   "channel": ", ".join(dict.fromkeys(hit["channels"])),
                                   "similarity": round(similarity[paper_id], 4),
                                   **({"snippet": hit["snippet"][:600], "section": hit.get("section")} if "snippet" in hit else {}),
                                   "position": position}
                position += 1
                entry["kept"].append({"paper": paper_id, "title": rows[paper_id]["title"], "snippet": "snippet" in hit})
            for paper_id in ranked[KEEP_PER_QUERY:]:
                event.drop(ref("paper", paper_id), f"targeted search {query!r}: below the best {KEEP_PER_QUERY}")
        entry["found"] = len(found)
        log.append(entry)

    if not added:
        return log
    chunks: dict[str, set[int]] = {}
    for paper_id, detail in added.items():
        ensure_abstract(ctx, paper_id)
        section = (detail.get("section") or "").strip().casefold()
        if detail.get("snippet") and section not in _NOT_BODY:
            detail["read"] = read_in_full(ctx, paper_id)
            source = ctx.db.query("SELECT text_source FROM papers WHERE paper_id = ?", (paper_id,))[0]["text_source"]
            if is_full_text(source) and (chunk := snippet_chunk(ctx, paper_id, detail["snippet"])) is not None:
                chunks[paper_id] = {chunk}
                detail["chunk"] = chunk
    rows = ctx.db.query(f"SELECT * FROM papers WHERE paper_id IN ({','.join('?' * len(added))})", tuple(added))
    full = [row for row in rows if row["paper_id"] in chunks]
    extracted, counts = extract_papers(ctx, full, rows, event, chunks=chunks)
    for paper_id, detail in added.items():
        detail["extractions"] = extracted.get(paper_id, [])
        add_plan_paper(ctx.db, run_id, paper_id, "search", detail.pop("position"), detail)
        event.generated.append(ref("paper", paper_id))
    log.append({"extraction": counts})
    return log
