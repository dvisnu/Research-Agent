"""The plan's provenance as a W3C PROV document, written as PROV-JSON with the ``prov``
package, so every element can be traced: recommendation -> reasoning -> evidence ->
paper -> section.

Entities: the plan versions and their elements, each element's decision and verdict,
the facts the elements cite, the evidence sentences with their sections and papers,
the idea and its gap, each critique and the report. Activities: the pipeline steps that
made them, from the run trace, each associated with the pipeline and with the models
that answered in it. Relations:

    plan        -hadMember->         element
    element     -wasGeneratedBy->    plan step          decision -wasGeneratedBy-> plan step
    element     -wasInfluencedBy->   decision           (reason, alternatives, critique points answered)
    element     -wasDerivedFrom->    fact               (attribute: the element's basis)
    fact        -wasDerivedFrom->    sentence           (attribute: the quote)
    section     -hadMember->         sentence           paper -hadMember-> section
    verdict     -wasDerivedFrom->    element            verdict -wasGeneratedBy-> verify step
    plan        -wasDerivedFrom->    idea               idea -wasDerivedFrom-> gap
    plan v2     -wasRevisionOf->     plan v1            plan step (v2) -used-> critique of v1
    critique    -wasDerivedFrom->    plan version       critique -wasGeneratedBy-> critique step
    report      -wasDerivedFrom->    final plan
    step        -wasAssociatedWith-> pipeline, models
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from prov.model import ProvDocument

from research_agent.ideas.ideas import IdeaStore
from research_agent.plans.facts import Menu, load_plan_papers
from research_agent.plans.store import PlanStore
from research_agent.storage import Database
from research_agent.trace import TraceRecorder

NAMESPACE = "https://research-agent.local/prov#"
TEXT_CHARS = 500
FILE = "provenance.json"


def _latest_steps(events: list[dict[str, Any]], step: str, key: str) -> dict[str, dict[str, Any]]:
    """The latest successful ``step`` event for each plan ID its detail[key] names."""
    found: dict[str, dict[str, Any]] = {}
    for event in events:
        if event["activity_type"] == "step" and event["step"] == step and event["status"] == "ok":
            plan_id = (event["detail"].get(key) or {}).get("plan_id")
            if plan_id:
                found[plan_id] = event
    return found


def provenance_document(db: Database, run_id: str) -> ProvDocument | None:
    versions = PlanStore(db).versions(run_id)
    if not versions:
        return None
    doc = ProvDocument()
    doc.add_namespace("ra", NAMESPACE)
    events = TraceRecorder(db).events(run_id)
    calls: dict[str, set[str]] = defaultdict(set)
    for event in events:
        if event["activity_type"] == "llm_call" and event["status"] == "ok" and event["agent"].get("model"):
            calls[event["parent_event_id"]].add(f"{event['agent']['provider']}:{event['agent']['model']}")
    pipeline = doc.agent("ra:agent/pipeline", {"prov:type": "prov:SoftwareAgent", "ra:name": "research-agent pipeline"})
    activities: dict[str, Any] = {}
    models: dict[str, Any] = {}

    def activity(event: dict[str, Any] | None) -> Any:
        if event is None:
            return None
        if event["event_id"] not in activities:
            act = doc.activity(f"ra:step/{event['event_id']}", event["started_at"], event["ended_at"],
                               {"ra:step": event["step"], "ra:attempt": event["attempt"] or 1})
            doc.wasAssociatedWith(act, pipeline)
            for model in sorted(calls.get(event["event_id"], ())):
                if model not in models:
                    models[model] = doc.agent(f"ra:model/{model}", {"prov:type": "prov:SoftwareAgent", "ra:model": model})
                doc.wasAssociatedWith(act, models[model])
            activities[event["event_id"]] = act
        return activities[event["event_id"]]

    plan_steps = _latest_steps(events, "plan", "plan")
    critique_steps = _latest_steps(events, "critique", "critique")
    verify_steps = _latest_steps(events, "verify", "verify")
    papers_seen: set[str] = set()
    sections_seen: set[str] = set()
    sentences_seen: set[str] = set()
    facts_seen: set[str] = set()

    def sentence(sentence_id: str) -> str:
        qualified = f"ra:sentence/{sentence_id}"
        if sentence_id in sentences_seen:
            return qualified
        sentences_seen.add(sentence_id)
        rows = db.query(
            "SELECT t.text, s.section_id, s.heading, p.paper_id, p.title, p.year, p.arxiv_id, p.doi FROM sentences t"
            " JOIN sections s ON s.section_id = t.section_id JOIN papers p ON p.paper_id = s.paper_id"
            " WHERE t.sentence_id = ?", (sentence_id,))
        if not rows:
            doc.entity(qualified, {"prov:type": "ra:Sentence", "ra:missing": "the sentence is no longer stored"})
            return qualified
        row = rows[0]
        doc.entity(qualified, {"prov:type": "ra:Sentence", "ra:text": row["text"][:TEXT_CHARS]})
        section = f"ra:section/{row['section_id']}"
        if row["section_id"] not in sections_seen:
            sections_seen.add(row["section_id"])
            doc.entity(section, {"prov:type": "ra:Section", "ra:heading": row["heading"] or ""})
            paper = f"ra:paper/{row['paper_id']}"
            if row["paper_id"] not in papers_seen:
                papers_seen.add(row["paper_id"])
                attrs = {"prov:type": "ra:Paper", "ra:title": row["title"]}
                attrs |= {f"ra:{k}": row[k] for k in ("year", "arxiv_id", "doi") if row[k]}
                doc.entity(paper, attrs)
            doc.hadMember(paper, section)
        doc.hadMember(section, qualified)
        return qualified

    idea = IdeaStore(db).get(versions[0].idea_id)
    idea_entity = doc.entity(f"ra:idea/{idea.idea_id}", {
        "prov:type": "ra:Idea", "ra:title": idea.title, "ra:purpose": idea.purpose, "ra:mechanism": idea.mechanism,
        "ra:evaluation": idea.evaluation, "ra:level": idea.level or 0})
    if idea.gap_node_id:
        gap = db.query("SELECT name FROM nodes WHERE node_id = ?", (idea.gap_node_id,))
        if gap:
            doc.wasDerivedFrom(idea_entity, doc.entity(f"ra:gap/{idea.gap_node_id}", {"prov:type": "ra:Gap", "ra:text": gap[0]["name"]}))

    papers = load_plan_papers(db, run_id)
    previous = None
    for plan in versions:
        menu = Menu.from_json(plan.menu, papers)
        step = activity(plan_steps.get(plan.plan_id))
        entity = doc.entity(f"ra:plan/{plan.plan_id}", {"prov:type": "ra:ExperimentPlan", "ra:round": plan.round,
                                                         "ra:status": plan.status})
        doc.wasDerivedFrom(entity, idea_entity)
        if step is not None:
            doc.wasGeneratedBy(entity, step)
        if previous is not None:
            doc.wasRevisionOf(entity, f"ra:plan/{previous.plan_id}")
            if previous.critique is not None and step is not None:
                doc.used(step, f"ra:critique/{previous.plan_id}")
        if plan.critique is not None:
            critique = doc.entity(f"ra:critique/{plan.plan_id}", {
                "prov:type": "ra:Critique", "ra:round": plan.critique.get("round", 0),
                "ra:points": json.dumps(plan.critique.get("points", []), ensure_ascii=False),
                "ra:answers_gap": plan.critique.get("answers_gap") or ""})
            doc.wasDerivedFrom(critique, entity)
            if (made := activity(critique_steps.get(plan.plan_id))) is not None:
                doc.wasGeneratedBy(critique, made)
        verify = activity(verify_steps.get(plan.plan_id))
        for element in plan.elements:
            qualified = f"ra:element/{element.element_id}"
            node = doc.entity(qualified, {
                "prov:type": "ra:PlanElement", "ra:field": element.field, "ra:position": element.position,
                "ra:text": element.text, "ra:name": element.name or "", "ra:status": element.status,
                "ra:verdict": element.verdict or ""})
            doc.hadMember(entity, node)
            decision = doc.entity(f"ra:decision/{element.element_id}", {
                "prov:type": "ra:Decision", "ra:reason": element.decision.get("reason", ""),
                "ra:alternatives": "; ".join(element.decision.get("alternatives", [])),
                "ra:answers": json.dumps(element.decision.get("answers", []), ensure_ascii=False)})
            doc.wasInfluencedBy(node, decision)
            if step is not None:
                doc.wasGeneratedBy(node, step)
                doc.wasGeneratedBy(decision, step)
            for cite in element.cites:
                fact = menu.get(cite)
                if fact is None:
                    continue
                fact_name = f"ra:fact/{run_id}/{fact.fact_id}"
                if fact_name not in facts_seen:
                    facts_seen.add(fact_name)
                    doc.entity(fact_name, {"prov:type": f"ra:Fact/{fact.kind}", "ra:text": fact.text[:TEXT_CHARS],
                                           "ra:origin": json.dumps(fact.origin, ensure_ascii=False, default=str)})
                    for e in fact.evidence:
                        doc.wasDerivedFrom(fact_name, sentence(e["sentence_id"]), other_attributes={"ra:quote": e["quote"]})
                doc.wasDerivedFrom(node, fact_name, other_attributes={"ra:basis": element.basis or ""})
            if element.verdict is not None:
                verdict = doc.entity(f"ra:verdict/{element.element_id}", {
                    "prov:type": "ra:Verification", "ra:label": element.verdict,
                    "ra:reason": (element.verification or {}).get("reason", "")})
                doc.wasDerivedFrom(verdict, node)
                if verify is not None:
                    doc.wasGeneratedBy(verdict, verify)
        previous = plan
    report = doc.entity(f"ra:report/{run_id}", {"prov:type": "ra:Report"})
    doc.wasDerivedFrom(report, f"ra:plan/{versions[-1].plan_id}")
    doc.wasDerivedFrom(report, idea_entity)
    return doc


def write_provenance(db: Database, run_id: str, out_dir: Path) -> Path | None:
    """Write the run's plan provenance as PROV-JSON; None when the run has no plan."""
    doc = provenance_document(db, run_id)
    if doc is None:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / FILE
    path.write_text(json.dumps(json.loads(doc.serialize(format="json")), indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def trace_chains(document: dict[str, Any], plan_id: str) -> list[dict[str, Any]]:
    """Walk a PROV-JSON document from each element of one plan version: is its reasoning
    there (a decision and the step that made it), and does each cited fact lead through an
    evidence sentence to a section and a paper?"""
    derived: dict[str, list[str]] = defaultdict(list)
    for relation in document.get("wasDerivedFrom", {}).values():
        derived[relation["prov:generatedEntity"]].append(relation["prov:usedEntity"])
    container: dict[str, str] = {}
    members: dict[str, list[str]] = defaultdict(list)
    for relation in document.get("hadMember", {}).values():
        container[relation["prov:entity"]] = relation["prov:collection"]
        members[relation["prov:collection"]].append(relation["prov:entity"])
    influencers: dict[str, list[str]] = defaultdict(list)
    for relation in document.get("wasInfluencedBy", {}).values():
        influencers[relation["prov:influencee"]].append(relation["prov:influencer"])
    generated = {r["prov:entity"]: r["prov:activity"] for r in document.get("wasGeneratedBy", {}).values()}
    entities = document.get("entity", {})

    def kind(name: str) -> str:
        return str(entities.get(name, {}).get("prov:type", ""))

    walked = []
    for element in members.get(f"ra:plan/{plan_id}", []):
        chains = []
        for fact in (f for f in derived.get(element, []) if kind(f).startswith("ra:Fact")):
            for sentence in derived.get(fact, []):
                section = container.get(sentence)
                paper = container.get(section) if section else None
                chains.append({"fact": fact, "sentence": sentence, "section": section, "paper": paper,
                               "complete": bool(section and paper and kind(paper) == "ra:Paper")})
        reasoning = any(kind(i) == "ra:Decision" for i in influencers.get(element, [])) and element in generated
        walked.append({"element": element, "field": entities.get(element, {}).get("ra:field"),
                       "status": entities.get(element, {}).get("ra:status"), "reasoning": reasoning,
                       "chains": chains, "traceable": reasoning and any(c["complete"] for c in chains)})
    return walked
