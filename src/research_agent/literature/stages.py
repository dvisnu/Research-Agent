"""Phase 1 pipeline stages: brief, search, expand, rank, fulltext.

Each stage records in its trace event what it used and produced, and what it
considered and dropped with the reason, so the retrieval log is the trace.
"""

import json
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pymupdf

from research_agent.clients.openalex import CS_FILTER
from research_agent.literature.brief import ResearchBrief, write_brief
from research_agent.literature.corpus import Corpus
from research_agent.literature.papers import PaperRecord, PaperStore, from_arxiv, from_openalex, from_s2
from research_agent.literature.parse import ParsedText, parse_arxiv_html, parse_pdf
from research_agent.literature.relevance import screen
from research_agent.literature.texts import abstract_text, store_text
from research_agent.trace import Event, ref

if TYPE_CHECKING:
    from research_agent.pipeline.graph import PipelineContext, PipelineState

# Results per search channel.
ARXIV_RESULTS = 20
S2_RESULTS = 30
SNIPPET_RESULTS = 20
OPENALEX_RESULTS = 25
# Expansion: how many seed papers, and how much to take from each.
SEEDS = 5
REFERENCES = 100  # papers cite 50-100 others; the foundational ones can be anywhere in the list
CITATIONS = 30
RECOMMENDATIONS = 20
# Ranking weights against relevance (all z-scored), chosen on the dev topics; see rank_stage.
CITATION_WEIGHT = 1.0
LINK_WEIGHT = 0.5
# The relevance check screens at most this many times the corpus size.
SCREEN_LIMIT = 2

FULL_TEXT_SOURCES = ("arxiv_html", "arxiv_pdf", "pdf", "upload")
PDF_PARSER = f"pymupdf-{pymupdf.VersionBind}"
HTML_PARSER = "arxiv-latexml-html"

Found = list[tuple[PaperRecord | None, dict[str, Any]]]


class AllChannelsFailed(RuntimeError):
    pass


class UploadError(RuntimeError):
    pass


def _brief(state: "PipelineState") -> ResearchBrief:
    return ResearchBrief.model_validate(state["brief"])


def brief_stage(state: "PipelineState", ctx: "PipelineContext", event: Event) -> dict[str, Any]:
    brief = write_brief(ctx.llm, state["question"])
    event.detail["brief"] = brief.model_dump()
    event.generated.append(ref("brief", state["run_id"]))
    return {"brief": brief.model_dump()}


def search_stage(state: "PipelineState", ctx: "PipelineContext", event: Event) -> dict[str, Any]:
    brief, sources = _brief(state), ctx.sources
    event.used.append(ref("brief", state["run_id"]))
    if state.get("uploads"):
        event.detail["uploads"] = _add_uploads(ctx, state["run_id"], state["uploads"], event)
    # Every angle of the brief runs on every channel; a channel's name carries its angle.
    channels: list[tuple[str, str, Callable[[str], Found]]] = []
    for i, angle in enumerate(brief.angles):
        channels += [
            (f"arxiv_search:{i}", angle.arxiv_query,
             lambda q: [(from_arxiv(e), {}) for e in sources.arxiv.search(q, max_results=ARXIV_RESULTS)]),
            (f"openalex_search:{i}", angle.keyword_query,
             lambda q: [(from_openalex(w), {}) for w in sources.openalex.search(q, per_page=OPENALEX_RESULTS, filter=CS_FILTER)]),
        ]
        if sources.s2:
            channels += [
                (f"s2_search:{i}", angle.keyword_query,
                 lambda q: [(from_s2(p), {}) for p in sources.s2.search(q, limit=S2_RESULTS)]),
                (f"s2_snippets:{i}", angle.passage_query, lambda q: _snippet_search(ctx, q)),
            ]
    if not sources.s2:
        event.detail["skipped"] = ["s2_search", "s2_snippets: no Semantic Scholar key"]

    def describe(name: str, rank: int) -> dict[str, Any]:
        channel, angle = name.split(":", 1)
        return {"channel": channel, "angle": int(angle), "rank": rank}

    log, found = _collect(ctx, state["run_id"], channels, describe)
    event.detail["queries"] = log
    event.generated += [ref("paper", paper_id) for paper_id in found]
    return {}


def _add_uploads(ctx: "PipelineContext", run_id: str, paths: list[str], event: Event) -> list[dict[str, Any]]:
    """Parse the user's PDFs into the corpus, tagged user-provided. An unreadable one fails the stage."""
    store, corpus = PaperStore(ctx.db), Corpus(ctx.db, run_id)
    log = []
    for path in map(Path, paths):
        event.used.append(ref("file", str(path)))
        try:
            parsed = parse_pdf(path.read_bytes())
        except (OSError, ValueError) as exc:
            raise UploadError(f"could not read uploaded file {path}: {exc}") from exc
        record = PaperRecord(source="upload", title=parsed.title or path.stem, arxiv_id=parsed.arxiv_id)
        paper_id = store.upsert(record)
        sentences = store_text(ctx.db, paper_id, parsed, source="upload", parser=PDF_PARSER)
        corpus.add(paper_id, {"channel": "upload", "file": path.name})
        event.generated += [ref("paper", paper_id), ref("text", paper_id)]
        log.append({"file": path.name, "paper": paper_id, "title": record.title, "sentences": sentences})
    return log


def _snippet_search(ctx: "PipelineContext", query: str) -> Found:
    """Snippet search returns passages; look up their papers in one batch request."""
    s2 = ctx.sources.s2
    snippets = [s for s in s2.snippets(query, limit=SNIPPET_RESULTS) if s.corpus_id]
    papers = s2.batch([f"CorpusId:{s.corpus_id}" for s in snippets]) if snippets else []
    return [
        (from_s2(paper) if paper else None, {"section": snippet.section, "snippet": snippet.text})
        for snippet, paper in zip(snippets, papers)
    ]


def _collect(
    ctx: "PipelineContext",
    run_id: str,
    channels: list[tuple[str, str, Callable[[str], Found]]],
    describe: Callable[[str, int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Run each channel, store what it finds, and log every query and result.

    A channel with no query is skipped. A failing channel is logged and skipped;
    if every channel that ran failed, the stage fails.
    """
    store, corpus = PaperStore(ctx.db), Corpus(ctx.db, run_id)
    log: list[dict[str, Any]] = []
    found: list[str] = []
    for name, query, fetch in channels:
        if not query:
            continue
        try:
            results = fetch(query)
        except Exception as exc:
            log.append({"channel": name, "query": query, "error": f"{type(exc).__name__}: {exc}"})
            continue
        paper_ids = []
        for rank, (record, extra) in enumerate(results, start=1):
            if record is None:
                continue
            paper_id = store.upsert(record)
            corpus.add(paper_id, {**describe(name, rank), **extra})
            paper_ids.append(paper_id)
            if paper_id not in found:
                found.append(paper_id)
        log.append({"channel": name, "query": query, "results": paper_ids})
    if log and all("error" in entry for entry in log):
        raise AllChannelsFailed("; ".join(f"{e['channel']}: {e['error']}" for e in log))
    return log, found


def expand_stage(state: "PipelineState", ctx: "PipelineContext", event: Event) -> dict[str, Any]:
    """Follow references, citations and recommendations from the best search hits."""
    run_id, sources = state["run_id"], ctx.sources
    seeds = Corpus(ctx.db, run_id).search_leaders(SEEDS)
    event.used += [ref("paper", seed["paper_id"]) for seed in seeds]

    channels: list[tuple[str, str, Callable[[str], Found]]] = []
    for seed in seeds:
        if sources.s2:
            key = _s2_key(seed)
            if key:
                channels += [
                    (f"s2_references:{seed['paper_id']}", key,
                     lambda k: [(from_s2(p), {}) for p in sources.s2.references(k, limit=REFERENCES)]),
                    (f"s2_citations:{seed['paper_id']}", key,
                     lambda k: [(from_s2(p), {}) for p in sources.s2.citations(k, limit=CITATIONS)]),
                    (f"s2_recommendations:{seed['paper_id']}", key,
                     lambda k: [(from_s2(p), {}) for p in sources.s2.recommendations(k, limit=RECOMMENDATIONS)]),
                ]
        else:
            channels.append((f"openalex_links:{seed['paper_id']}", _openalex_key(seed), lambda k: _openalex_links(ctx, k)))

    def describe(name: str, rank: int) -> dict[str, Any]:
        channel, seed_id = name.split(":", 1)
        return {"channel": channel, "via": seed_id}

    log, found = _collect(ctx, run_id, channels, describe)
    event.detail["expansions"] = log
    event.generated += [ref("paper", paper_id) for paper_id in found]
    return {}


def _s2_key(seed: Any) -> str | None:
    if seed["s2_id"]:
        return seed["s2_id"]
    if seed["arxiv_id"]:
        return f"arXiv:{seed['arxiv_id']}"
    return f"DOI:{seed['doi']}" if seed["doi"] else None


def _openalex_key(seed: Any) -> str:
    if seed["openalex_id"]:
        return seed["openalex_id"]
    if seed["arxiv_id"]:
        return f"doi:10.48550/arXiv.{seed['arxiv_id']}"
    return f"doi:{seed['doi']}" if seed["doi"] else ""


def _openalex_links(ctx: "PipelineContext", key: str) -> Found:
    openalex = ctx.sources.openalex
    work = openalex.work(key)
    if work is None:
        return []
    linked = [
        *[(w, "openalex_references") for w in openalex.works(work.referenced_works[:REFERENCES])],
        *[(w, "openalex_citations") for w in openalex.citing(work.id, per_page=CITATIONS)],
        *[(w, "openalex_related") for w in openalex.works(work.related_works[:RECOMMENDATIONS])],
    ]
    return [(from_openalex(w), {"link": link}) for w, link in linked]


def rank_stage(state: "PipelineState", ctx: "PipelineContext", event: Event) -> dict[str, Any]:
    """Score every candidate, screen the best for relevance, then split full / abstract / excluded.

    Score = z(relevance) + CITATION_WEIGHT * z(log citations per year)
            + LINK_WEIGHT * z(links), all z-scored within the run.

    Relevance alone (SPECTER2 similarity to the question) favours recent papers
    whose titles echo the question and buries the field's foundations; the
    citation-rate prior brings those back. Links count the search channels and
    seed papers that independently led to a paper; they keep recent, well-linked
    work from being crowded out by old, highly cited papers. A missing citation
    count counts as the median.

    The LLM relevance check then reads the best-scored candidates in order and
    drops the off-topic ones; the corpus is the first papers it keeps.
    """
    corpus = Corpus(ctx.db, state["run_id"])
    rows = corpus.papers()
    event.detail["candidates"] = len(rows)
    if not rows:
        return {}
    brief = _brief(state)
    query = ctx.embedder.query(brief.rank_query)
    vectors = ctx.embedder.papers([(row["title"], row["abstract"]) for row in rows])
    relevance = vectors @ query
    citations = _citation_rates(rows)
    links = np.array([_links(row) for row in rows], dtype=float)
    scores = _zscores(relevance) + CITATION_WEIGHT * _zscores(citations) + LINK_WEIGHT * _zscores(links)
    full, kept = ctx.settings.full_read, ctx.settings.full_read + ctx.settings.abstract_only
    event.detail.update(
        without_abstract=sum(1 for row in rows if not row["abstract"]),
        without_citation_count=sum(1 for row in rows if row["citation_count"] is None),
        citation_weight=CITATION_WEIGHT,
        link_weight=LINK_WEIGHT,
    )

    order = [int(index) for index in scores.argsort()[::-1]]
    uploaded = {row["paper_id"] for row in rows if _uploaded(row)}
    candidates = [rows[index] for index in order if rows[index]["paper_id"] not in uploaded]
    judged, screened = (
        screen(ctx.llm, brief, candidates, want=kept, limit=SCREEN_LIMIT * kept) if kept else ({}, 0)
    )
    screened_ids = {row["paper_id"] for row in candidates[:screened]}
    event.detail["relevance_check"] = {
        "screened": screened,
        "judgements": [{"paper": paper_id, **j.model_dump(exclude={"paper"})} for paper_id, j in judged.items()],
    }

    accepted = 0  # papers kept so far, uploads aside: uploads don't use up the quotas
    for rank, index in enumerate(order, start=1):
        paper_id = rows[index]["paper_id"]
        cited = rows[index]["citation_count"]
        judgement = judged.get(paper_id)
        why = f"relevance {relevance[index]:.3f}, {'?' if cited is None else cited} citations"
        if judgement:
            why += f"; {judgement.label.replace('_', '-')}: {judgement.reason}"
        if paper_id in uploaded:
            level, reason = "full", f"rank {rank} ({why}): provided by the user, always read in full"
        elif judgement and judgement.label == "off_topic":
            level, reason = "excluded", f"rank {rank} ({why}): judged off-topic by the relevance check"
        elif paper_id in screened_ids and accepted < kept:
            accepted += 1
            if accepted <= full:
                level, reason = "full", f"rank {rank} ({why}): among the first {full} kept, read in full"
            else:
                level, reason = "abstract", f"rank {rank} ({why}): among the first {kept} kept, read by abstract"
        else:
            level, reason = "excluded", f"rank {rank} ({why}): below the cutoff of {kept}"
        corpus.set_rank(
            paper_id, score=float(scores[index]), rank=rank, read_level=level, reason=reason,
            relevance=judgement.label if judgement else None,
        )
        if level == "excluded":
            event.drop(ref("paper", paper_id), reason)
        else:
            event.generated.append(ref("paper", paper_id))
    return {}


def _citation_rates(rows: list[Any]) -> np.ndarray:
    """log(1 + citations per year since publication); a missing count is the median."""
    this_year = date.today().year
    known = [row["citation_count"] for row in rows if row["citation_count"] is not None]
    median = float(np.median(known)) if known else 0.0
    return np.array([
        np.log1p((median if row["citation_count"] is None else row["citation_count"])
                 / max(1, this_year - (row["year"] or this_year) + 1))
        for row in rows
    ])


def _links(row: Any) -> int:
    """How many search channels and seed papers independently led to this paper."""
    found = json.loads(row["found_by"])
    return len({f["channel"] for f in found if "rank" in f}) + len({f["via"] for f in found if "via" in f})


def _zscores(values: np.ndarray) -> np.ndarray:
    spread = values.std()
    return (values - values.mean()) / spread if spread else np.zeros_like(values)


def fulltext_stage(state: "PipelineState", ctx: "PipelineContext", event: Event) -> dict[str, Any]:
    """Parse full-read papers into sentences, and abstract-read papers' abstracts."""
    corpus = Corpus(ctx.db, state["run_id"])
    log: list[dict[str, Any]] = []
    for row in corpus.papers("full"):
        paper_id = row["paper_id"]
        event.used.append(ref("paper", paper_id))
        if row["text_source"] in FULL_TEXT_SOURCES:
            if _uploaded(row):
                log.append({"paper": paper_id, "result": "uploaded by the user"})
                event.generated.append(ref("text", paper_id))
                continue
            if ctx.settings.reuse_texts:
                log.append({"paper": paper_id, "result": f"reused {row['text_source']} text from an earlier run"})
                event.generated.append(ref("text", paper_id))
                continue
        parsed, source, parser, tried = _fetch_full_text(ctx, row)
        sentences = None
        if parsed is not None:
            try:
                sentences = store_text(ctx.db, paper_id, parsed, source=source, parser=parser)
            except Exception as exc:  # one unreadable paper must not fail the run
                tried.append({"via": f"store {source}", "result": f"{type(exc).__name__}: {exc}"[:200]})
        if sentences is None:
            reason = "no usable full text: " + "; ".join(f"{t['via']}: {t['result']}" for t in tried)
            corpus.set_level(paper_id, "abstract", reason)
            log.append({"paper": paper_id, "result": "demoted to abstract", "tried": tried})
            _store_abstract(ctx, row)
            continue
        log.append({"paper": paper_id, "result": f"parsed {source}", "sentences": sentences, "tried": tried})
        event.generated.append(ref("text", paper_id))

    for row in corpus.papers("abstract"):
        if row["text_source"] is None and _store_abstract(ctx, row):
            event.generated.append(ref("text", row["paper_id"]))
    event.detail["full_text"] = log
    return {}


def _uploaded(row: Any) -> bool:
    """Whether the user uploaded this paper in this run, so its stored text is this run's."""
    return any(f["channel"] == "upload" for f in json.loads(row["found_by"]))


def _store_abstract(ctx: "PipelineContext", row: Any) -> bool:
    if row["text_source"] is not None:
        return False
    parsed = abstract_text(row["title"], row["abstract"])
    if parsed is None:
        return False
    store_text(ctx.db, row["paper_id"], parsed, source="abstract", parser="abstract")
    return True


def _fetch_full_text(
    ctx: "PipelineContext", row: Any
) -> tuple[ParsedText | None, str, str, list[dict[str, str]]]:
    """Try arXiv HTML, then arXiv PDF, then any open-access PDF. Returns what worked and every try."""
    sources = ctx.sources
    tried: list[dict[str, str]] = []

    def attempt(via: str, fetch: Callable[[], Any], parse: Callable[[Any], ParsedText]) -> ParsedText | None:
        try:
            data = fetch()
            if data is None:
                tried.append({"via": via, "result": "not available"})
                return None
            parsed = parse(data)
        except Exception as exc:  # network errors and ParseError alike: note it, try the next way
            tried.append({"via": via, "result": f"{type(exc).__name__}: {exc}"[:200]})
            return None
        tried.append({"via": via, "result": "ok"})
        return parsed

    if row["arxiv_id"]:
        parsed = attempt("arxiv_html", lambda: sources.arxiv_web.html(row["arxiv_id"]), parse_arxiv_html)
        if parsed:
            return parsed, "arxiv_html", HTML_PARSER, tried
        parsed = attempt("arxiv_pdf", lambda: sources.arxiv_web.pdf(row["arxiv_id"]), parse_pdf)
        if parsed:
            return parsed, "arxiv_pdf", PDF_PARSER, tried

    urls = [row["pdf_url"]] if row["pdf_url"] and "arxiv.org" not in row["pdf_url"] else []
    if row["doi"] and sources.unpaywall:
        try:
            record = sources.unpaywall.lookup(row["doi"])
            location = record.best_oa_location if record else None
            if location and location.url_for_pdf and location.url_for_pdf not in urls:
                urls.append(location.url_for_pdf)
        except Exception as exc:
            tried.append({"via": "unpaywall", "result": f"{type(exc).__name__}: {exc}"[:200]})
    for url in urls:
        parsed = attempt(f"pdf {url}", lambda: sources.downloader.pdf(url), parse_pdf)
        if parsed:
            return parsed, "pdf", PDF_PARSER, tried
    if not tried:
        tried.append({"via": "none", "result": "no arXiv ID, open-access PDF or DOI"})
    return None, "", "", tried
