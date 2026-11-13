"""Verification: the verifier model checks every cited claim of the final plan.

Each element that cites facts makes a claim about the literature: its basis (what the
cited facts say that it rests on). The claim, with the element itself, is checked
against the full sentences its facts quote, with their papers. The facts are listed
with the claim, each with its papers and why the plan uses them (the writer refers to
facts by number, as in "F7 reports ..."). The verifier model, of
a different family from the plan writer, labels each one:

- supported: the sentences state or directly imply it, with the same names, numbers
  and settings; the element is kept;
- not supported: they don't say it, say only part of it, or the element misstates
  them; the element is kept but flagged;
- contradicted: they say something incompatible with it; the element is removed from
  the plan (and kept in the record, with the reason).

An element with no source (it cites nothing, or nothing it cites has evidence) is
flagged without a call ("no_source"), and so is one whose check failed ("unverified").
A verifier that is unavailable fails the stage instead, so the run can be resumed once
it is back: a plan is not passed on half-checked.

Provenance Coverage is the share of the final plan's elements (those not removed) whose
claim the verifier found supported: supported decisions over all decisions.
"""

from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, Field

from research_agent.graph.store import JudgmentStore, digest
from research_agent.llm import LLM, LLMOutputError, Prompt, Role
from research_agent.plans.store import FIELDS, Element
from research_agent.storage import Database

VERIFY_BATCH = 6  # claims per call
# Groq's free tier allows 8k tokens a minute and counts the 4k reply budget in: a batch's
# claims stay under BATCH_CHARS, about 2.5k tokens.
BATCH_CHARS = 9_000
SENTENCE_CHARS = 400
FACT_CHARS = 160
FACT_PAPERS = 3
STATUS = {"supported": "kept", "not_supported": "flagged", "contradicted": "removed",
          "no_source": "flagged", "unverified": "flagged"}


class ClaimVerdict(BaseModel):
    claim: int = Field(description="The claim's number")
    label: Literal["supported", "not_supported", "contradicted"]
    reason: str = Field(description="One sentence")


class Verdicts(BaseModel):
    """The verifier model's reply."""

    verdicts: list[ClaimVerdict] = Field(description="One verdict per claim, in order")


_SYSTEM = """You check whether an experiment plan cites the literature faithfully. Each numbered claim gives a plan element, the statement it makes about the literature (what it rests on), the facts it cites and the sentences they quote from the papers. Fact numbers such as F7 in the statement refer to the cited facts; each fact names the papers it comes from and why the plan uses them (such as the idea's closest papers). The element is part of a plan: its own design choices and predictions are new, and they are not what you judge. Judge the statement, and anything the element says the papers report:
- supported: the sentences state the statement or directly imply it, with the same names, numbers and settings, and the element attributes nothing to the papers that they do not say;
- not_supported: the sentences do not say the statement, or only part of it; or the statement adds claims about the papers that they do not make; or the element attributes to the papers something they do not say, such as a number, a dataset or a setting; or the statement gives no grounds for the element;
- contradicted: the sentences say something incompatible with the statement.
Judge what the papers say only from the sentences given, not from what you know about them. Give a one-sentence reason for each."""


def verify_prompt(claims: list[dict[str, Any]]) -> Prompt:
    lines: list[str] = []
    for number, claim in enumerate(claims, start=1):
        if lines:
            lines.append("")
        lines += [f"[{number}] Plan element ({claim['label']}): {claim['element']}",
                  f"Statement about the literature: {claim['basis']}",
                  "Cited facts:", *claim["facts"],
                  "Cited sentences:", *(f'- ({e["paper"]}) "{e["text"]}"' for e in claim["evidence"])]
    return Prompt("plan_verify", 3, _SYSTEM, "\n".join(lines))


def _label(row: Any) -> str:
    return f"{row['title']}, {row['year']}" if row["year"] else row["title"]


def _sentences(db: Database, ids: list[str]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for start in range(0, len(ids), 500):
        part = ids[start : start + 500]
        for row in db.query(
            "SELECT t.sentence_id, t.text, p.title, p.year FROM sentences t JOIN sections s ON s.section_id = t.section_id"
            f" JOIN papers p ON p.paper_id = s.paper_id WHERE t.sentence_id IN ({','.join('?' * len(part))})", tuple(part),
        ):
            found[row["sentence_id"]] = {"paper": _label(row), "text": row["text"][:SENTENCE_CHARS]}
    return found


def _papers(db: Database, ids: list[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for start in range(0, len(ids), 500):
        part = ids[start : start + 500]
        for row in db.query(f"SELECT paper_id, title, year FROM papers WHERE paper_id IN ({','.join('?' * len(part))})",
                            tuple(part)):
            found[row["paper_id"]] = _label(row)
    return found


def _fact_lines(element: Element, papers: dict[str, str]) -> list[str]:
    """What each fact the element cites is, with its papers and why the plan uses them."""
    lines = []
    for fact in element.derivation.get("facts", []):
        text = fact["text"] if len(fact["text"]) <= FACT_CHARS else fact["text"][: FACT_CHARS - 3] + "..."
        sources = [f"{papers.get(p, p)}: {(info.get('why') or ['a paper of the run'])[0]}"  # the first reason: closest first
                   for p, info in list((fact.get("origin") or {}).get("papers", {}).items())[:FACT_PAPERS]]
        lines.append(f"- {fact['fact']} ({fact['kind']}) {text}" + (f" [from {' | '.join(sources)}]" if sources else ""))
    return lines


def verify_elements(db: Database, llm: LLM, judgments: JudgmentStore, elements: list[Element]) -> dict[str, Any]:
    """Label every element's claim and set its status; returns what the calls did."""
    sentences = _sentences(db, list(dict.fromkeys(e["sentence_id"] for el in elements for e in el.evidence)))
    papers = _papers(db, list(dict.fromkeys(
        p for el in elements for f in el.derivation.get("facts", []) for p in (f.get("origin") or {}).get("papers", {}))))
    claims: list[tuple[Element, dict[str, Any]]] = []
    for element in elements:
        evidence = [sentences[e["sentence_id"]] for e in element.evidence if e["sentence_id"] in sentences]
        if not element.cites or not evidence:
            _set(element, "no_source", "It cites nothing with evidence." if element.cites else "It cites nothing.", None)
            continue
        claims.append((element, {"label": element.label, "element": element.text, "basis": element.basis or element.text,
                                 "facts": _fact_lines(element, papers), "evidence": evidence}))
    stats = {"claims": len(claims), "calls": 0, "reused": 0, "failed_batches": 0}
    for batch in _batches(claims):
        prompt = verify_prompt([claim for _, claim in batch])
        input_hash = digest(prompt.id, prompt.system, prompt.user)
        kept = judgments.find("plan_verify", input_hash)
        if kept is not None:
            stats["reused"] += 1
        else:
            try:
                answer = llm.answer(Role.VERIFIER, prompt, Verdicts)
            except LLMOutputError as exc:
                stats["failed_batches"] += 1
                for element, _ in batch:
                    _set(element, "unverified", f"The verifier gave no valid reply: {exc}"[:300], None)
                continue
            stats["calls"] += 1
            kept = {"model": answer.model, "reply": answer.output.model_dump()}
            judgments.add("plan_verify", input_hash, answer.model, kept)
        verdicts = {v["claim"]: v for v in kept["reply"]["verdicts"]}
        for number, (element, _) in enumerate(batch, start=1):
            verdict = verdicts.get(number)
            if verdict is None:
                _set(element, "unverified", "The verifier's reply left this claim out.", kept["model"])
            else:
                _set(element, verdict["label"], verdict["reason"], kept["model"], prompt.id)
    return stats


def _batches(claims: list[tuple[Element, dict[str, Any]]]) -> list[list[tuple[Element, dict[str, Any]]]]:
    """Claims in order, at most VERIFY_BATCH and about BATCH_CHARS a batch."""
    batches: list[list[tuple[Element, dict[str, Any]]]] = []
    size = 0
    for claim in claims:
        chars = len(verify_prompt([claim[1]]).user)
        if not batches or len(batches[-1]) >= VERIFY_BATCH or size + chars > BATCH_CHARS:
            batches.append([])
            size = 0
        batches[-1].append(claim)
        size += chars
    return batches


def _set(element: Element, verdict: str, reason: str, model: str | None, prompt: str | None = None) -> None:
    element.verdict = verdict
    element.status = STATUS[verdict]
    element.verification = {"reason": reason, "model": model, "prompt": prompt}


def coverage(elements: list[Element]) -> dict[str, Any]:
    """Provenance Coverage of the final plan, with the labels behind it."""
    final = [e for e in elements if e.status != "removed"]
    supported = sum(1 for e in final if e.verdict == "supported")
    fields = {}
    for name in FIELDS:
        mine = [e for e in final if e.field == name]
        if mine:
            fields[name] = {"elements": len(mine), "supported": sum(1 for e in mine if e.verdict == "supported")}
    return {
        "elements": len(final), "supported": supported,
        "coverage": round(supported / len(final), 4) if final else None,
        "removed": sum(1 for e in elements if e.status == "removed"),
        "flagged": sum(1 for e in final if e.status == "flagged"),
        "labels": dict(Counter(e.verdict or "unchecked" for e in elements)),
        "fields": fields,
    }
