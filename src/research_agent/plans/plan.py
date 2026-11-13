"""Writing the experiment plan for the run's best idea, and revising it after a critique.

The reasoning model sees the idea, its novelty check and the numbered fact menu
(``facts``), and writes every element of the plan with the facts it rests on (cited by
number), a basis (what those facts say that the element rests on) and a decision (why
this choice, over which alternatives). Datasets, baselines and metrics are picked from
the menu; an expected result is stated against a reported one. Code then turns each
element's citations into its three provenance records:

- source: the cited facts' evidence, sentence addresses with the exact quotes;
- derivation: the step, round, model and prompt that wrote it, and each cited fact's
  origin (the extraction or graph query it came from, and why its papers were read);
- decision: the model's reason, the listed alternatives of the same kind it was chosen
  over, and the critique points it answers.

Citations of facts that do not exist are dropped and recorded; a dataset, baseline or
metric citing no listed one of its kind is marked "not from the literature" for the
critique, as is an expected result citing no reported result.
"""

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from research_agent.graph.store import JudgmentStore, digest
from research_agent.ideas.ideas import Idea
from research_agent.llm import LLM, LLMOutputError, Prompt, Role
from research_agent.plans.facts import Fact, Menu
from research_agent.plans.store import ENTITY_FIELDS, FIELDS, SINGLE, Element, Plan

MAX_ELEMENT_EVIDENCE = 10  # one sentence per cited fact first: the verifier must see every fact an element cites
ALTERNATIVES = 5
FIELD_TITLES = {
    "hypothesis": "Hypothesis", "research_question": "Research question", "method": "Method",
    "datasets": "Datasets", "baselines": "Baselines", "metrics": "Metrics", "conditions": "Experimental conditions",
    "ablations": "Ablation studies", "expected_results": "Expected results", "risks": "Risks / limitations",
}


class PlanItem(BaseModel):
    name: str = Field(description="For a dataset, baseline or metric: its name as listed; otherwise empty")
    text: str = Field(description="The element itself, specific, in one or two sentences")
    cites: list[str] = Field(description='The numbers of the facts it rests on, e.g. ["F7", "F35"]')
    basis: str = Field(description="One sentence: what the cited facts state that the element rests on; empty when it cites nothing")
    decision: str = Field(description="One sentence: why this choice, and over which listed alternatives")
    issues: list[int] = Field(description="When revising: the numbers of the critique points it answers; otherwise empty")


class PlanDraft(BaseModel):
    """The reasoning model's reply."""

    hypothesis: PlanItem
    research_question: PlanItem
    method: list[PlanItem] = Field(description="2 to 6 steps")
    datasets: list[PlanItem] = Field(description="1 to 4")
    baselines: list[PlanItem] = Field(description="2 to 5")
    metrics: list[PlanItem] = Field(description="1 to 4")
    conditions: list[PlanItem] = Field(description="1 to 4")
    ablations: list[PlanItem] = Field(description="1 to 4")
    expected_results: list[PlanItem] = Field(description="1 to 4")
    risks: list[PlanItem] = Field(description="1 to 4")


class PointAnswer(BaseModel):
    point: int = Field(description="The critique point's number")
    action: Literal["fixed", "declined"]
    note: str = Field(description="One sentence: what changed, or why the point is declined")


class PlanRevision(PlanDraft):
    """The reasoning model's reply to a critique."""

    answers: list[PointAnswer] = Field(description="One answer per critique point, in order")


_SYSTEM = """You write experiment plans for research ideas in computer science and machine learning. Every element of the plan must be traceable to the literature: it cites, by number, the facts it rests on, from the list given. The facts were extracted from the papers closest to the idea, each backed by quoted sentences.

Write the plan:
- hypothesis: one testable sentence predicting what the idea's mechanism achieves, and compared with what.
- research_question: the question the experiments answer.
- method: 2 to 6 concrete steps that implement the idea's mechanism.
- datasets: 1 to 4 datasets or benchmarks, chosen from the listed ones.
- baselines: 2 to 5 methods to compare with, chosen from the listed ones; include the closest existing work.
- metrics: 1 to 4 metrics, chosen from the listed ones.
- conditions: 1 to 4 experimental conditions: the settings held fixed or varied (models, sizes, budgets, data splits) and the controls.
- ablations: 1 to 4 ablations, each removing or replacing one part of the mechanism.
- expected_results: 1 to 4 expected outcomes, each stated against a reported result it cites.
- risks: 1 to 4 risks or limitations of the plan, preferably ones the papers report.

For every element:
- name: for a dataset, baseline or metric, its name as listed; otherwise empty.
- text: the element itself, specific, in plain technical language.
- cites: the numbers of the facts it rests on. A dataset, baseline or metric cites the fact that lists it; an expected result cites the reported result it is compared with; a method step cites the work its technique comes from; a risk cites the limitation it repeats. Cite nothing only for a step that is new to this idea.
- basis: one sentence saying what the cited facts themselves state that the element rests on, faithful to them (the same names, numbers and settings) and nothing more: not the element's own choices or predictions, and nothing about the papers that the facts do not say. Empty when nothing is cited.
- decision: one sentence: why this choice, and over which listed alternatives.
- issues: {issues}
Use only the facts listed. Do not cite anything else, and do not invent results or numbers."""

_FIRST_ISSUES = "leave empty."
_REVISION_ISSUES = "the numbers of the critique points the element answers; empty for an element left as it was."

_REVISION_NOTE = """

You are revising your earlier plan after a critique. Answer every critique point: change, add or remove elements to fix it, or decline it with a reason when it is wrong or would make the plan infeasible. Keep the elements that need no change as they were. Facts found for the critique are marked (new)."""


def _idea_lines(idea: Idea) -> list[str]:
    lines = ["Idea:", *idea.block()]
    if idea.kind:
        lines.append(f"Kind: {idea.kind.replace('_', ' ')}")
    if idea.detail.get("builds_on"):
        lines.append("Builds on: " + "; ".join(idea.detail["builds_on"]))
    if idea.detail.get("feasibility"):
        lines.append(f"Feasibility: {idea.detail['feasibility']}")
    assessment = idea.assessment or {}
    if idea.level is not None:
        reasons = " ".join(assessment.get("reasons", [])[:2])
        lines.append(f"Novelty check: level {idea.level} of 5 ({assessment.get('label', '')}). {reasons}".strip())
    return lines


def plan_prompt(idea: Idea, menu: Menu) -> Prompt:
    lines = [*_idea_lines(idea), "", "Facts from the literature:", *menu.render()]
    return Prompt("plan_writeup", 2, _SYSTEM.format(issues=_FIRST_ISSUES), "\n".join(lines))


def revision_prompt(idea: Idea, menu: Menu, previous: Plan, points: list[dict[str, Any]]) -> Prompt:
    lines = [*_idea_lines(idea), "", "Your earlier plan:", *plan_lines(previous), "", "Critique points:"]
    for point in points:
        lines.append(f"[{point['number']}] ({point['element']}) {point['problem']} Suggested fix: {point['fix']}")
    lines += ["", "Facts from the literature:", *menu.render(new_since=previous.round)]
    return Prompt("plan_revision", 2, _SYSTEM.format(issues=_REVISION_ISSUES) + _REVISION_NOTE, "\n".join(lines))


def plan_lines(plan: Plan) -> list[str]:
    """A plan version as the models see it: each element with what it cites."""
    lines: list[str] = []
    for name in FIELDS:
        elements = plan.field_elements(name)
        if not elements:
            continue
        lines.append(f"{FIELD_TITLES[name]}:")
        for element in elements:
            cites = f" [{', '.join(element.cites)}]" if element.cites else " [cites nothing]"
            lines.append(f"- ({element.label}) {element.text}{cites}")
    return lines


def write_plan(
    llm: LLM, judgments: JudgmentStore, prompt: Prompt, output: type[PlanDraft]
) -> tuple[PlanDraft | None, dict[str, Any]]:
    """The model's plan for ``prompt``, kept for reuse; None and the error when no valid reply came."""
    kind = prompt.name
    input_hash = digest(prompt.id, prompt.system, prompt.user)
    kept = judgments.find(kind, input_hash)
    info: dict[str, Any] = {"prompt": prompt.id, "reused": kept is not None}
    if kept is None:
        try:
            answer = llm.answer(Role.REASONING, prompt, output)
        except LLMOutputError as exc:
            return None, {**info, "error": str(exc)}
        kept = {"model": answer.model, "reply": answer.output.model_dump()}
        judgments.add(kind, input_hash, answer.model, kept)
    info["model"] = kept["model"]
    return output.model_validate(kept["reply"]), info


def _norm(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def _evidence(facts: list[Fact]) -> list[dict[str, str]]:
    """The cited facts' evidence, one sentence per fact first, at most MAX_ELEMENT_EVIDENCE."""
    picked: list[dict[str, str]] = []
    seen: set[str] = set()
    for depth in range(max((len(f.evidence) for f in facts), default=0)):
        for fact in facts:
            if depth < len(fact.evidence) and fact.evidence[depth]["sentence_id"] not in seen:
                if len(picked) < MAX_ELEMENT_EVIDENCE:
                    picked.append(fact.evidence[depth])
                    seen.add(fact.evidence[depth]["sentence_id"])
    return picked


def to_elements(
    draft: PlanDraft, menu: Menu, *, round_: int, info: dict[str, Any], previous: Plan | None = None,
    points: list[dict[str, Any]] = (),
) -> list[Element]:
    """The draft's elements with their provenance records."""
    by_number = {point["number"]: point for point in points}
    elements: list[Element] = []
    for name in FIELDS:
        items = [getattr(draft, name)] if name in SINGLE else getattr(draft, name)
        kind = ENTITY_FIELDS.get(name)
        cited_here = {c for item in items for c in item.cites}
        chosen = {f.fact_id for c in cited_here if (f := menu.get(c))}
        alternatives = [f"{f.text} ({f.fact_id})" for f in menu.of_kind(kind) if f.fact_id not in chosen][:ALTERNATIVES] if kind else []
        before = previous.field_elements(name) if previous else []
        for position, item in enumerate(items, start=1):
            if not item.text.strip():
                continue
            facts, unknown = [], []
            for cite in item.cites:
                fact = menu.get(cite)
                if fact is None:
                    unknown.append(cite)
                elif fact not in facts:
                    facts.append(fact)
            checks = []
            if kind and not any(f.kind == kind for f in facts):
                checks.append(f"cites no listed {kind}: not from the literature")
            if name == "expected_results" and not any(f.kind == "result" for f in facts):
                checks.append("cites no reported result")
            if not facts:
                checks.append("cites no fact")
            listed = next((f.text for f in facts if f.kind == kind), None) if kind else None
            if previous is None:
                change = "new"
            elif any(_norm(e.text) == _norm(item.text) for e in before):
                change = "kept"
            else:
                change = "changed" if position <= len(before) else "added"
            elements.append(Element(
                field=name, position=position, text=item.text.strip(),
                name=(item.name.strip() or listed) if kind else None,
                basis=item.basis.strip() or None, cites=[f.fact_id for f in facts], evidence=_evidence(facts),
                derivation={
                    "step": "plan", "round": round_, "prompt": info.get("prompt"), "model": info.get("model"),
                    "reused": info.get("reused"), "change": change,
                    "facts": [{"fact": f.fact_id, "kind": f.kind, "text": f.text[:300], "origin": f.origin,
                               "added_in_round": f.added} for f in facts],
                    "unknown_cites": unknown, "checks": checks,
                },
                decision={
                    "reason": item.decision.strip(), "alternatives": alternatives,
                    "answers": [{"point": n, "problem": by_number[n]["problem"]} for n in item.issues if n in by_number],
                },
            ))
    return elements
