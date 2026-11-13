"""The critique of a plan version: checks by code, then a review by the verifier model.

Code checks what can be checked exactly:
- the closest existing work is compared with: each of the CLOSEST_AS_BASELINES closest
  papers whose purpose the novelty check found the same as or similar to the idea's is
  cited by a baseline (its approach, or a method it proposes);
- datasets, baselines and metrics are listed facts ("not from the literature" otherwise);
- each dataset has a reported result of a chosen baseline, so expected results on it
  rest on something; if not, a targeted search looks for one;
- elements other than method steps cite something.

The verifier model, a different model family from the plan writer, then reviews what
code cannot judge: feasibility for an academic lab, missing strong baselines, whether the
datasets and metrics test the hypothesis, whether each ablation isolates one part of the
mechanism, confounds, and whether the plan answers the idea's gap in full. A point that
needs facts the plan's sources lack carries a search query. Up to MAX_SEARCHES queries a
round go to the targeted search, code's first.
"""

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from research_agent.graph.names import name_key
from research_agent.graph.store import JudgmentStore, digest
from research_agent.ideas.ideas import Idea
from research_agent.llm import LLM, LLMOutputError, Prompt, Role
from research_agent.plans.facts import Fact, Menu
from research_agent.plans.plan import plan_lines
from research_agent.plans.store import Plan

CLOSEST_AS_BASELINES = 3
MAX_POINTS = 6  # from the verifier model
MAX_SEARCHES = 3  # targeted searches per round
FACT_CHARS = (160, 90)  # of each cited fact shown to the verifier; the second when the prompt is too long
UNUSED_NAMES = 8  # per kind, of the listed entities the plan does not use
# Groq's free tier allows 8k tokens a minute and counts the 4k reply budget in, so the prompt
# stays under about 3.5k tokens; longer ones are rejected outright.
PROMPT_CHARS = 12_000


def _keys(fact: Fact) -> set[str]:
    return set(fact.attrs.get("keys") or [name_key(fact.text)])


def _covers(fact: Fact, paper_id: str) -> bool:
    """Whether citing ``fact`` as a baseline compares with that paper's own work."""
    if fact.kind == "approach":
        return paper_id in fact.paper_ids
    return fact.kind == "method" and paper_id in fact.attrs.get("proposed_by", [])


def _point(source: str, kind: str, element: str, problem: str, fix: str, search: str = "") -> dict[str, Any]:
    return {"source": source, "kind": kind, "element": element, "problem": problem, "fix": fix, "search": search}


def code_checks(plan: Plan, menu: Menu) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    baselines = plan.field_elements("baselines")
    baseline_facts = [f for e in baselines for c in e.cites if (f := menu.get(c))]
    for fact in menu.of_kind("approach"):
        a = fact.attrs
        if a["position"] > CLOSEST_AS_BASELINES or a["purpose"] not in ("same", "similar"):
            continue
        paper = fact.paper_ids[0]
        if not any(_covers(f, paper) for f in baseline_facts):
            points.append(_point(
                "code", "missing_baseline", "baselines",
                f"The closest existing work, {menu.label(paper)}, whose purpose the novelty check found {a['purpose']} "
                "to the idea's, is not a baseline.",
                f"Add its approach ({fact.fact_id}) as a baseline, or say why it cannot be compared."))

    for element in plan.elements:
        if element.status == "removed":
            continue
        checks = element.derivation.get("checks", [])
        if any("not from the literature" in c for c in checks):
            subject = element.name or " ".join(element.text.split()[:8])
            points.append(_point(
                "code", "unsupported", element.label,
                f"{subject} is not among the listed facts, so nothing in the literature backs it.",
                "Choose a listed one, or cite where it comes from.", subject))
        elif element.field == "expected_results" and "cites no reported result" in checks:
            points.append(_point("code", "unsupported", element.label, "The expected result cites no reported result.",
                                 "State it against a reported result of a baseline on the same dataset."))
        elif not element.cites and element.field != "method":
            points.append(_point("code", "unsupported", element.label, f"The {element.label} cites nothing.",
                                 "Cite the facts it rests on, or drop it."))

    baseline_keys = set()
    baseline_names = []
    for element in baselines:
        for fact in (menu.get(c) for c in element.cites):
            if fact is not None and fact.kind == "method":
                baseline_keys |= _keys(fact)
        if element.name:
            baseline_keys.add(name_key(element.name))
            baseline_names.append(element.name)
    results = menu.of_kind("result")
    for element in plan.field_elements("datasets"):
        facts = [f for c in element.cites if (f := menu.get(c)) and f.kind == "dataset"]
        keys = set().union(*(_keys(f) for f in facts)) | ({name_key(element.name)} if element.name else set())
        if not keys:
            continue
        on_it = [r for r in results if name_key(r.attrs.get("dataset") or "") in keys]
        if not any(name_key(r.attrs.get("method") or "") in baseline_keys for r in on_it):
            name = element.name or facts[0].text
            query = f"{baseline_names[0]} {name} results" if baseline_names else f"{name} benchmark results"
            points.append(_point(
                "code", "no_reference_result", element.label,
                f"No reported result of a chosen baseline on {name} is listed, so expected results on it rest on nothing.",
                "Use reported results of a baseline on it, or choose a dataset with reported results.", query))
    return points


class CritiquePoint(BaseModel):
    element: str = Field(description='The element it concerns, e.g. "baselines", "method 2" or "plan"')
    kind: Literal["missing_baseline", "missing_dataset", "metric", "infeasible", "ablation", "confound", "gap",
                  "unsupported", "other"]
    problem: str = Field(description="One sentence")
    fix: str = Field(description="One sentence")
    search: str = Field(description="A literature search query when the fix needs facts the plan's sources do not give; otherwise empty")


class PlanCritique(BaseModel):
    """The verifier model's reply."""

    answers_gap: Literal["yes", "partly", "no"] = Field(description="Whether the plan, if it works, answers the research gap in full")
    points: list[CritiquePoint] = Field(description=f"At most {MAX_POINTS} points, most important first; none when the plan is sound")


_SYSTEM = f"""You review experiment plans for research ideas in computer science and machine learning, as a careful reviewer would before the experiments are run. Check:
- feasibility for an academic lab within a few months: compute, data and model access, time;
- baselines: is the closest existing work compared with, and are standard strong baselines for these datasets missing?
- datasets and metrics: do they test the hypothesis, and are they the field's usual ones for it?
- ablations: does each isolate one part of the mechanism?
- conditions: are confounds controlled, such as the same backbone model, budget and data for every method?
- the gap: does the plan, if it works, answer the research gap in full, or only part of it?
Report at most {MAX_POINTS} points, most important first, each with the element it concerns, the problem, the fix, and a short literature search query when the fix needs facts the plan's sources do not give (for example reported results of a baseline on a dataset, or a dataset for a setting); otherwise leave search empty. Report nothing about what is already fine."""


def critique_prompt(idea: Idea, gap: str | None, plan: Plan, menu: Menu) -> Prompt:
    """The plan with what it cites, shortened to PROMPT_CHARS: cited facts first lose length,
    then the entities the plan does not use are left out."""
    cited = list(dict.fromkeys(c for e in plan.elements if e.status != "removed" for c in e.cites))
    unused = []
    for kind, label in (("dataset", "datasets"), ("method", "methods"), ("metric", "metrics")):
        names = [f.text for f in menu.of_kind(kind) if f.fact_id not in cited][:UNUSED_NAMES]
        if names:
            unused.append(f"{label}: {', '.join(names)}")
    for chars, with_unused in ((FACT_CHARS[0], True), (FACT_CHARS[1], True), (FACT_CHARS[1], False)):
        lines = ["Idea:", *idea.block(), f"Research gap it answers: {gap or 'none given'}", "", "Closest existing work:"]
        for fact in menu.of_kind("approach"):
            a = fact.attrs
            lines.append(f"- {menu.label(fact.paper_ids[0])}: {fact.text[:chars]} "
                         f"(purpose {a['purpose']}, mechanism {a['mechanism']} to the idea's)")
        lines += ["", "Plan:", *plan_lines(plan), "", "Cited facts:"]
        lines += [f"[{fact.fact_id}] {menu.line(fact)[:chars]}" for cite in cited if (fact := menu.get(cite))]
        if unused and with_unused:
            lines += ["", "Listed in the sources but not used: " + "; ".join(unused)]
        if len("\n".join(lines)) <= PROMPT_CHARS:
            break
    return Prompt("plan_critique", 1, _SYSTEM, "\n".join(lines))


def review(llm: LLM, judgments: JudgmentStore, prompt: Prompt) -> tuple[PlanCritique | None, dict[str, Any]]:
    input_hash = digest(prompt.id, prompt.system, prompt.user)
    kept = judgments.find("plan_critique", input_hash)
    info: dict[str, Any] = {"prompt": prompt.id, "reused": kept is not None}
    if kept is None:
        try:
            answer = llm.answer(Role.VERIFIER, prompt, PlanCritique)
        except LLMOutputError as exc:
            return None, {**info, "error": str(exc)}
        kept = {"model": answer.model, "reply": answer.output.model_dump()}
        judgments.add("plan_critique", input_hash, answer.model, kept)
    info["model"] = kept["model"]
    return PlanCritique.model_validate(kept["reply"]), info


def _norm(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def critique(
    llm: LLM, judgments: JudgmentStore, idea: Idea, gap: str | None, plan: Plan, menu: Menu, round_: int
) -> dict[str, Any]:
    """The critique record: numbered points (code's first), the searches to run, the verifier's view."""
    points = code_checks(plan, menu)
    reply, info = review(llm, judgments, critique_prompt(idea, gap, plan, menu))
    if reply is not None:
        points += [_point("verifier", p.kind, p.element, p.problem, p.fix, p.search.strip())
                   for p in reply.points[:MAX_POINTS]]
    for number, point in enumerate(points, start=1):
        point["number"] = number
    searches: list[dict[str, Any]] = []
    for point in points:
        if point["search"] and len(searches) < MAX_SEARCHES and _norm(point["search"]) not in {_norm(s["query"]) for s in searches}:
            searches.append({"query": point["search"], "point": point["number"]})
    return {"round": round_, "points": points, "searches": searches,
            "answers_gap": reply.answers_gap if reply is not None else None, "verifier": info,
            "counts": {"code": sum(p["source"] == "code" for p in points), "verifier": sum(p["source"] == "verifier" for p in points)}}
