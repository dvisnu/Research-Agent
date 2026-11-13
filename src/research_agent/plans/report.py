"""The report's Phase 5 part, built from what is stored: the run's latest plan version,
each element with its numbered sources, its decision and its verdict, the elements
verification removed, the critique points left open, and the revisions made."""

import re
from typing import Any

from research_agent.ideas.ideas import IdeaStore
from research_agent.plans.facts import ENTITY_KINDS, Menu, load_plan_papers
from research_agent.plans.store import FIELDS, SINGLE, Element, PlanStore
from research_agent.report import ExperimentPlan, PlanElementReport, PlanSource
from research_agent.storage import Database

REPORT_FIELDS = {"risks": "risks_and_limitations"}  # plan field -> ExperimentPlan attribute


def plan_report(db: Database, run_id: str) -> ExperimentPlan:
    versions = PlanStore(db).versions(run_id)
    if not versions:
        planned = any(i.final_rank == 1 for i in IdeaStore(db).for_run(run_id))
        return ExperimentPlan(note="the plan stages have not run" if planned else "the run has no assessed idea to plan")
    plan = versions[-1]
    idea = IdeaStore(db).get(plan.idea_id)
    menu = Menu.from_json(plan.menu, load_plan_papers(db, run_id))
    where = _sentence_places(db, [e["sentence_id"] for el in plan.elements for e in el.evidence])
    sources: dict[str, PlanSource] = {}

    def readable(text: str | None) -> str | None:
        """The text with each fact number replaced by what it names, as the report lists no fact numbers."""
        def name(match: re.Match[str]) -> str:
            fact = menu.get(match[0])
            if fact is None:
                return match[0]
            return fact.text if fact.kind in ENTITY_KINDS else menu.label(fact.paper_ids[0]) if fact.paper_ids else match[0]
        return re.sub(r"\bF\d+\b", name, text) if text else text

    def report(element: Element) -> PlanElementReport:
        numbers = []
        for e in element.evidence:
            if e["sentence_id"] not in sources:
                paper, section = where.get(e["sentence_id"], ("a paper no longer stored", None))
                sources[e["sentence_id"]] = PlanSource(number=len(sources) + 1, paper=paper, sentence_id=e["sentence_id"],
                                                       quote=e["quote"], section=section)
            numbers.append(sources[e["sentence_id"]].number)
        return PlanElementReport(
            element_id=element.element_id, label=element.label, text=readable(element.text), name=element.name,
            basis=readable(element.basis), sources=numbers, decision=readable(element.decision.get("reason")) or None,
            alternatives=[re.sub(r"\s*\(F\d+\)$", "", a) for a in element.decision.get("alternatives", [])],
            derivation=derivation_line(element),
            verdict=element.verdict, verdict_reason=(element.verification or {}).get("reason"), status=element.status,
        )

    out = ExperimentPlan(idea_id=idea.idea_id, idea_title=idea.title, provenance_coverage=plan.detail.get("coverage", {}))
    for name in FIELDS:
        kept = [report(e) for e in plan.field_elements(name)]
        setattr(out, REPORT_FIELDS.get(name, name), (kept[0] if kept else None) if name in SINGLE else kept)
    out.removed = [report(e) for e in plan.elements if e.status == "removed"]
    out.sources = sorted(sources.values(), key=lambda s: s.number)
    if plan.critique:
        out.open_points = [f"({p['element']}) {p['problem']} {p['fix']}" for p in plan.critique.get("points", [])]
    for version, before in zip(versions[1:], versions):
        answers = version.detail.get("answers", [])
        fixed = sum(1 for a in answers if a["action"] == "fixed")
        points = len((before.critique or {}).get("points", []))
        searched = len((before.critique or {}).get("searches", []))
        changes = [e.derivation.get("change") for e in version.elements]
        out.revisions.append(
            f"Round {version.round}: the critique of round {before.round} made {points} point(s), {searched} needing a "
            f"targeted search; the revision fixed {fixed} and declined {len(answers) - fixed}, changing "
            f"{changes.count('changed')} element(s) and adding {changes.count('added')}.")
    return out


def derivation_line(element: Element) -> str:
    d = element.derivation
    change = {"kept": "kept from the previous round", "changed": "changed in this round", "added": "added in this round"}
    parts = [f"plan round {d.get('round')}" + (f", {change[d['change']]}" if d.get("change") in change else "")]
    answers = element.decision.get("answers", [])
    if answers:
        parts.append("for critique point(s) " + ", ".join(str(a["point"]) for a in answers))
    if d.get("facts"):
        parts.append("cites " + ", ".join(f"{f['fact']} ({f['kind']})" for f in d["facts"]))
    if d.get("model"):
        parts.append(f"written by {d['model']}")
    return "; ".join(parts)


def _sentence_places(db: Database, ids: list[str]) -> dict[str, tuple[str, str | None]]:
    """Each sentence's paper label and section heading."""
    places: dict[str, tuple[str, str | None]] = {}
    unique = list(dict.fromkeys(ids))
    for start in range(0, len(unique), 500):
        part = unique[start : start + 500]
        rows: list[Any] = db.query(
            "SELECT t.sentence_id, s.heading, p.title, p.year FROM sentences t JOIN sections s ON s.section_id = t.section_id"
            f" JOIN papers p ON p.paper_id = s.paper_id WHERE t.sentence_id IN ({','.join('?' * len(part))})", tuple(part))
        for row in rows:
            label = f"{row['title']} ({row['year']})" if row["year"] else row["title"]
            places[row["sentence_id"]] = (label, row["heading"])
    return places
