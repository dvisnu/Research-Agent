"""Comparing an idea with each of its closest papers, one paper per call.

For each facet (purpose, mechanism, evaluation) the extraction model says what
the paper's own version is, quoting the paper, and whether it is the same as
the idea's, similar (a minor difference such as another dataset, model or
setting), different, or not stated in the text shown. It also says whether the
paper already solves the gap the idea answers, and how the idea differs from it.

Every quote goes through the Phase 2 quote check. The verdict that counts
("effective") fails closed: a "different" whose quotes all fail the check counts
as "similar", since nothing shows the paper really differs, while a "same" or
"similar" counts even without a passing quote, flagged unverified. "Not stated"
counts as unknown.

Papers are first compared by abstract. Before an idea is rated 3 or higher, its
FULL_READS closest papers are read in full: arXiv HTML, else a PDF, parsed as
in Phase 1. A paper whose stored text is only its abstract has the body added
after it (``extend_text``), so evidence quoting the abstract stays valid.
"""

import json
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from research_agent.graph.extraction import (
    Evidence,
    abstract_item,
    chunk_sections,
    full_resolver,
    load_sections,
    short_label,
)
from research_agent.graph.quotes import QuoteChecker
from research_agent.graph.store import JudgmentStore, digest
from research_agent.ideas.ideas import FACETS, Idea
from research_agent.literature.stages import fetch_full_text, is_full_text
from research_agent.literature.texts import abstract_text, extend_text, store_text
from research_agent.llm import LLMOutputError, Prompt, Role
from research_agent.storage import new_id, utc_now

if TYPE_CHECKING:
    from research_agent.pipeline.graph import PipelineContext

FULL_READS = 3
FULL_CHARS = 60_000  # of a paper's text shown in a full-text comparison: about 15k tokens


class FacetVerdict(BaseModel):
    match: Literal["same", "similar", "different", "not_stated"]
    paper: str = Field(description="One sentence: the paper's own version of this facet; empty when not stated")
    evidence: list[Evidence] = Field(description="Quotes showing the paper's version; empty only when not stated")


class PaperComparison(BaseModel):
    """The extraction model's reply."""

    purpose: FacetVerdict
    mechanism: FacetVerdict
    evaluation: FacetVerdict
    solves_gap: Literal["yes", "partly", "no"] = Field(description="Whether the paper already solves the research gap")
    gap_evidence: list[Evidence] = Field(description="Quotes showing how the paper addresses the gap; empty when no")
    difference: str = Field(description='One sentence: how the idea differs from this paper, or "none"')


_SYSTEM = """You compare a proposed research idea in computer science or machine learning with one existing paper, to judge whether the paper already does the idea.

The idea has three facets: purpose (the problem it addresses), mechanism (how it works) and evaluation (how it is tested). For each facet, say in one sentence what the paper's own version is, quote the paper to show it, and say how it matches the idea's:
- same: essentially the idea's (for the mechanism, the same technique even if named differently);
- similar: close, differing only in something minor such as the dataset, model or setting;
- different: clearly different;
- not_stated: the text given does not say.
Then say whether the paper already solves the research gap the idea answers (yes, partly or no), quoting it if yes or partly, and in one sentence how the idea differs from the paper, or "none".

{addresses} Quote words copied exactly from one sentence, with that sentence's address. Judge only from the text given, not from what you know about the paper."""

_ABSTRACT_ADDRESSES = "Each sentence of the paper follows its number in brackets, e.g. [3]."
_FULL_ADDRESSES = "Each sentence of the paper follows its address in brackets, e.g. [s3:12]."
NO_GAP = "none given; answer no"


def _idea_lines(idea: Idea, gap: str | None) -> list[str]:
    return ["Idea:", *idea.block(), f"Research gap it answers: {gap or NO_GAP}", ""]


def abstract_prompt(idea: Idea, gap: str | None, title: str, year: int | None, lines: list[str]) -> Prompt:
    header = f"Paper: {title}" + (f" ({year})" if year else "")
    body = [header, *(f"[{n}] {text}" for n, text in enumerate(lines))]
    return Prompt("novelty_compare", 1, _SYSTEM.format(addresses=_ABSTRACT_ADDRESSES), "\n".join(_idea_lines(idea, gap) + body))


def full_prompt(idea: Idea, gap: str | None, title: str, year: int | None, blocks: list[tuple[str | None, list[tuple[str, str]]]]) -> Prompt:
    header = f"Paper: {title}" + (f" ({year})" if year else "")
    body = [header]
    for heading, lines in blocks:
        body += ["", f"## {heading or 'Untitled section'}", *(f"[{short_label(sid)}] {text}" for sid, text in lines)]
    return Prompt("novelty_compare_full", 1, _SYSTEM.format(addresses=_FULL_ADDRESSES), "\n".join(_idea_lines(idea, gap) + body))


def ensure_abstract(ctx: "PipelineContext", paper_id: str) -> None:
    """Store the paper's abstract as its text if it has none yet, so quotes get addresses."""
    row = ctx.db.query("SELECT title, abstract, text_source FROM papers WHERE paper_id = ?", (paper_id,))[0]
    if row["text_source"] is None and (parsed := abstract_text(row["title"], row["abstract"])) is not None:
        store_text(ctx.db, paper_id, parsed, source="abstract", parser="abstract")


def read_in_full(ctx: "PipelineContext", paper_id: str) -> dict[str, Any]:
    """Make sure the paper's body is stored; says what happened."""
    row = ctx.db.query("SELECT * FROM papers WHERE paper_id = ?", (paper_id,))[0]
    if is_full_text(row["text_source"]):
        return {"result": f"stored {row['text_source']} text"}
    parsed, source, parser, tried = fetch_full_text(ctx, row)
    if parsed is None:
        return {"result": "no full text", "tried": tried}
    try:
        if row["text_source"] == "abstract":
            added = extend_text(ctx.db, paper_id, parsed, source=source, parser=parser)
        else:
            added = store_text(ctx.db, paper_id, parsed, source=source, parser=parser)
    except Exception as exc:  # one unreadable paper must not fail the check
        return {"result": f"{type(exc).__name__}: {exc}"[:200], "tried": tried}
    return {"result": f"parsed {source}", "sentences": added}


def compare(
    ctx: "PipelineContext", judgments: JudgmentStore, idea: Idea, gap: str | None, paper_id: str, depth: str
) -> dict[str, Any]:
    """Compare the idea with one paper, by abstract or full text; returns the checked verdict
    (``status`` "ok") or why there is none (``status`` "failed")."""
    row = ctx.db.query("SELECT title, year FROM papers WHERE paper_id = ?", (paper_id,))[0]
    sections = load_sections(ctx.db, paper_id)
    shown: dict[str, str] = {}
    if depth == "full":
        chunks = chunk_sections(sections, limit=FULL_CHARS)
        if not chunks:
            return {"status": "failed", "depth": depth, "error": "no text to compare with"}
        blocks = [(block.heading, [(line.sentence_id, line.text) for line in block.lines]) for block in chunks[0].blocks]
        shown = {sid: text for _, lines in blocks for sid, text in lines}
        prompt = full_prompt(idea, gap, row["title"], row["year"], blocks)
        resolve = full_resolver(paper_id)
    else:
        item = abstract_item(paper_id, row["title"], row["year"], sections)
        if item is None:
            return {"status": "failed", "depth": depth, "error": "no abstract or text to compare with"}
        ids = [line.sentence_id for line in item.lines]
        shown = {line.sentence_id: line.text for line in item.lines}
        prompt = abstract_prompt(idea, gap, row["title"], row["year"], [line.text for line in item.lines])

        def resolve(label: str) -> str | None:
            digits = label.strip().strip("[]").strip()
            return ids[int(digits)] if digits.isdigit() and int(digits) < len(ids) else None

    input_hash = digest(prompt.id, prompt.system, prompt.user)
    reply = judgments.find("novelty_compare", input_hash)
    verdict: dict[str, Any] = {"depth": depth, "prompt": prompt.id, "reused": reply is not None}
    if reply is None:
        try:
            answer = ctx.llm.answer(Role.EXTRACTION, prompt, PaperComparison)
        except LLMOutputError as exc:
            return {**verdict, "status": "failed", "error": str(exc)}
        reply = answer.output.model_dump()
        judgments.add("novelty_compare", input_hash, answer.model, reply)
        verdict["model"] = answer.model
    return {**verdict, **check(reply, shown, resolve), "status": "ok"}


def check(reply: dict[str, Any], shown: dict[str, str], resolve: Any) -> dict[str, Any]:
    """The reply with its quotes checked and each facet's effective match (fail closed)."""
    checker = QuoteChecker(shown)
    quotes = {"quotes": 0, "passed": 0}

    def evidence(items: list[dict[str, str]]) -> list[dict[str, str]]:
        kept = []
        for item in items:
            quotes["quotes"] += 1
            sentence = resolve(item["sentence"]) or item["sentence"]
            outcome, sentence = checker.locate(sentence, item["quote"])
            if outcome != "failed" and all(k["sentence_id"] != sentence for k in kept):
                quotes["passed"] += 1
                kept.append({"sentence_id": sentence, "quote": item["quote"]})
        return kept

    out: dict[str, Any] = {}
    for facet in FACETS:
        given = reply[facet]
        kept = evidence(given["evidence"])
        match = given["match"]
        if match == "not_stated":
            effective = "unknown"
        elif match == "different" and not kept:
            effective = "similar"  # nothing shows the paper really differs
        else:
            effective = match
        out[facet] = {"match": match, "effective": effective, "paper": given["paper"], "evidence": kept,
                      "verified": bool(kept)}
    out["solves_gap"] = reply["solves_gap"]
    out["gap_evidence"] = evidence(reply["gap_evidence"])
    out["difference"] = reply["difference"]
    out["quote_check"] = quotes
    return out


def store_comparison(ctx: "PipelineContext", idea_id: str, paper_id: str, verdict: dict[str, Any]) -> None:
    """Keep the verdict and its evidence, replacing an earlier comparison of the same pair."""
    with ctx.db.transaction() as conn:
        old = conn.execute("SELECT comparison_id FROM comparisons WHERE idea_id = ? AND paper_id = ?", (idea_id, paper_id)).fetchone()
        if old:
            conn.execute("DELETE FROM evidence WHERE owner_kind = 'comparison' AND owner_id = ?", (old["comparison_id"],))
            conn.execute("DELETE FROM comparisons WHERE comparison_id = ?", (old["comparison_id"],))
        comparison_id = new_id("c")
        conn.execute(
            "INSERT INTO comparisons (comparison_id, idea_id, paper_id, depth, verdict, model, prompt_version, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (comparison_id, idea_id, paper_id, verdict["depth"], json.dumps(verdict, ensure_ascii=False),
             verdict.get("model"), verdict.get("prompt"), utc_now()),
        )
        quoted = [e for facet in FACETS for e in verdict.get(facet, {}).get("evidence", [])] + verdict.get("gap_evidence", [])
        conn.executemany(
            "INSERT OR IGNORE INTO evidence (owner_kind, owner_id, sentence_id, quote) VALUES ('comparison', ?, ?, ?)",
            [(comparison_id, e["sentence_id"], e["quote"]) for e in quoted],
        )
