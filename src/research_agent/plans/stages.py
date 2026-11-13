"""The Phase 5 stages: ``plan``, ``critique``, ``targeted_search`` and ``verify``.

    plan -> critique -> (targeted_search -> plan -> critique, while the critique makes points,
    at most MAX_CRITIQUE_ROUNDS critiques in all) -> verify -> report

The plan is for the run's best idea (Phase 4's rank 1). The first ``plan`` round stores
the plan papers (the idea's closest papers, its gap's papers and the related work),
extracts those the run has not extracted, builds the fact menu and has the reasoning
model write the plan. ``critique`` reviews the latest version (code checks, then the
verifier model); ``targeted_search`` looks for what it found missing; a later ``plan``
round revises the plan with the new facts. ``verify`` has the verifier model check every
cited claim of the last version and computes its Provenance Coverage.

The stages keep plans, elements and plan papers in SQLite; the pipeline state holds a
short summary.
"""

import json
from collections import Counter
from typing import TYPE_CHECKING, Any

from research_agent.graph.stages import extract_papers
from research_agent.graph.store import JudgmentStore
from research_agent.ideas.compare import FULL_READS, ensure_abstract
from research_agent.ideas.ideas import Idea, IdeaStore, load_gaps
from research_agent.ideas.stages import latest_detail
from research_agent.literature.stages import is_full_text
from research_agent.llm import Role
from research_agent.plans.critique import critique
from research_agent.plans.facts import Menu, add_plan_paper, build_menu, load_plan_papers
from research_agent.plans.plan import PlanDraft, PlanRevision, plan_prompt, revision_prompt, to_elements, write_plan
from research_agent.plans.search import run_searches
from research_agent.plans.store import Plan, PlanStore
from research_agent.plans.verify import coverage, verify_elements
from research_agent.storage import new_id
from research_agent.trace import Event, ref

if TYPE_CHECKING:
    from research_agent.pipeline.graph import PipelineContext, PipelineState

MAX_CRITIQUE_ROUNDS = 2  # a first critique and one of the revision; the last one's points stay open


def best_idea(ctx: "PipelineContext", run_id: str) -> Idea | None:
    return next((i for i in IdeaStore(ctx.db).for_run(run_id) if i.final_rank == 1), None)


def _gap(ctx: "PipelineContext", run_id: str, idea: Idea) -> dict[str, Any] | None:
    return next((g for g in load_gaps(ctx.db, run_id) if g["node_id"] == idea.gap_node_id), None)


def _comparisons(ctx: "PipelineContext", idea_id: str) -> list[dict[str, Any]]:
    rows = ctx.db.query(
        "SELECT i.paper_id, i.position, p.title, p.year, c.comparison_id, c.verdict FROM idea_candidates i"
        " JOIN papers p USING (paper_id) LEFT JOIN comparisons c ON c.idea_id = i.idea_id AND c.paper_id = i.paper_id"
        " WHERE i.idea_id = ? AND i.position IS NOT NULL ORDER BY i.position", (idea_id,))
    return [{**dict(row), "verdict": json.loads(row["verdict"]) if row["verdict"] else {}} for row in rows]


def _summary(plan: Plan) -> dict[str, Any]:
    return {"plan_id": plan.plan_id, "round": plan.round, "idea_id": plan.idea_id, "elements": len(plan.elements),
            "status": plan.status, "coverage": (plan.detail.get("coverage") or {}).get("coverage")}


# plan ----------------------------------------------------------------------------------

def gather_papers(ctx: "PipelineContext", state: "PipelineState", idea: Idea, gap: dict[str, Any] | None,
                  event: Event) -> dict[str, Any]:
    """Store the plan papers and extract those the run has no extraction of. One of the
    FULL_READS closest papers whose full text is stored is extracted in full even when the
    run read it by abstract: its experiments are in the body."""
    run_id = state["run_id"]
    roles: list[tuple[str, str, int | None]] = []
    for row in ctx.db.query("SELECT paper_id, position FROM idea_candidates WHERE idea_id = ? AND position IS NOT NULL"
                            " ORDER BY position", (idea.idea_id,)):
        roles.append((row["paper_id"], "closest", row["position"]))
    for paper_id in dict.fromkeys(e["paper_id"] for e in (gap or {}).get("evidence", [])):
        roles.append((paper_id, "gap", None))
    related = state.get("related_work") or latest_detail(ctx.db, run_id, "related_work").get("related_work", [])
    for position, paper in enumerate(related, start=1):
        roles.append((paper["paper_id"], "related", position))

    corpus: dict[str, list[tuple[str, str]]] = {}
    for row in ctx.db.query("SELECT e.paper_id, e.extraction_id, e.kind FROM run_extractions r JOIN extractions e"
                            " USING (extraction_id) WHERE r.run_id = ? ORDER BY e.paper_id, e.chunk", (run_id,)):
        corpus.setdefault(row["paper_id"], []).append((row["extraction_id"], row["kind"]))
    top = {paper_id for paper_id, role, position in roles if role == "closest" and (position or 99) <= FULL_READS}
    full, abstract, reused = [], [], {}
    for paper_id in dict.fromkeys(p for p, _, _ in roles):
        row = ctx.db.query("SELECT * FROM papers WHERE paper_id = ?", (paper_id,))[0]
        held = corpus.get(paper_id, [])
        in_full = is_full_text(row["text_source"])
        if held and (paper_id not in top or not in_full or any(kind == "full" for _, kind in held)):
            reused[paper_id] = [x for x, _ in held]
            continue
        ensure_abstract(ctx, paper_id)
        row = ctx.db.query("SELECT * FROM papers WHERE paper_id = ?", (paper_id,))[0]
        (full if in_full else abstract).append(row)
    extracted, counts = extract_papers(ctx, full, abstract, event)
    for paper_id, role, position in roles:
        ids = reused.get(paper_id) or extracted.get(paper_id, [])
        add_plan_paper(ctx.db, run_id, paper_id, role, position,
                       {"extractions": ids, "extracted": "by the run" if paper_id in reused else "for the plan"})
    return {"papers": len({p for p, _, _ in roles}), "roles": dict(Counter(role for _, role, _ in roles)),
            "extracted_by_the_run": len(reused), "extraction": counts}


def plan_stage(state: "PipelineState", ctx: "PipelineContext", event: Event) -> dict[str, Any]:
    run_id = state["run_id"]
    round_ = state.get("plan_round", 0) + 1
    store = PlanStore(ctx.db)
    idea = best_idea(ctx, run_id)
    if idea is None:
        event.detail["plan"] = {"round": round_, "skipped": "the run has no assessed idea to plan"}
        return {"plan_round": round_, "plan": None}
    event.used.append(ref("idea", idea.idea_id))
    gap = _gap(ctx, run_id, idea)
    comparisons = _comparisons(ctx, idea.idea_id)
    judgments = JudgmentStore(ctx.db, ctx.llm.models(Role.REASONING))
    previous = store.latest(run_id) if round_ > 1 else None
    detail: dict[str, Any] = {"round": round_, "idea": idea.idea_id}
    if previous is None:
        store.clear_run(run_id)  # Phase 5 run again on a run starts afresh
        detail["papers"] = gather_papers(ctx, state, idea, gap, event)
        menu = build_menu(ctx.db, run_id, idea, gap, comparisons)
        draft, info = write_plan(ctx.llm, judgments, plan_prompt(idea, menu), PlanDraft)
        points: list[dict[str, Any]] = []
    else:
        points = (previous.critique or {}).get("points", [])
        old = Menu.from_json(previous.menu, load_plan_papers(ctx.db, run_id))
        menu = build_menu(ctx.db, run_id, idea, gap, comparisons, previous=old, round_=round_)
        draft, info = write_plan(ctx.llm, judgments, revision_prompt(idea, menu, previous, points), PlanRevision)
    detail["llm"] = info
    detail["menu"] = dict(Counter(f.kind for f in menu.facts.values()))
    event.used += [ref("paper", paper_id) for paper_id in menu.papers]
    if draft is None:
        event.detail["plan"] = {**detail, "kept": previous.plan_id if previous else None}
        if previous is None:
            return {"plan_round": round_, "plan": None}
        return {"plan_round": round_, "plan": _summary(previous)}  # the revision failed: the earlier version stands

    elements = to_elements(draft, menu, round_=round_, info=info, previous=previous, points=points)
    answers = [a.model_dump() for a in draft.answers] if isinstance(draft, PlanRevision) else []
    plan = store.add(Plan(new_id("plan"), run_id, idea.idea_id, round_, "draft", menu.as_json(), None,
                          {"prompt": info.get("prompt"), "model": info.get("model"), "reused": info.get("reused"),
                           "answers": answers, "menu": detail["menu"]}, elements))
    if previous is not None:
        store.set_status(previous.plan_id, "superseded")
    for element in elements:
        for cite in element.derivation.get("unknown_cites", []):
            event.drop(ref("cite", cite), f"{element.label} cites {cite}, which is not a listed fact")
    event.generated += [ref("plan", plan.plan_id), *(ref("plan_element", e.element_id) for e in elements)]
    event.detail["plan"] = {**detail, "plan_id": plan.plan_id, "elements": len(elements),
                            "changes": dict(Counter(e.derivation["change"] for e in elements)),
                            "answers": dict(Counter(a["action"] for a in answers))}
    return {"plan_round": round_, "plan": _summary(plan)}


# critique ---------------------------------------------------------------------------------

def critique_stage(state: "PipelineState", ctx: "PipelineContext", event: Event) -> dict[str, Any]:
    round_ = state.get("critique_round", 0) + 1
    options = getattr(ctx, "options", None)
    forced = bool(options is not None and options.force_loops and round_ < 2)
    plan = PlanStore(ctx.db).latest(state["run_id"]) if state.get("plan") else None
    if plan is None:
        event.detail["critique"] = {"round": round_, "skipped": "no plan"}
        return {"critique_round": round_, "critique_revise": forced}
    menu = Menu.from_json(plan.menu, load_plan_papers(ctx.db, plan.run_id))
    idea = IdeaStore(ctx.db).get(plan.idea_id)
    gap = _gap(ctx, plan.run_id, idea)
    judgments = JudgmentStore(ctx.db, ctx.llm.models(Role.VERIFIER))
    record = critique(ctx.llm, judgments, idea, gap["gap"] if gap else idea.detail.get("gap"), plan, menu, round_)
    PlanStore(ctx.db).set_critique(plan.plan_id, record)
    revise = (bool(record["points"]) and round_ < MAX_CRITIQUE_ROUNDS) or forced
    event.used.append(ref("plan", plan.plan_id))
    event.generated.append(ref("critique", plan.plan_id))
    event.detail["critique"] = {"round": round_, "plan_id": plan.plan_id, "points": record["points"],
                                "searches": record["searches"], "answers_gap": record["answers_gap"],
                                "verifier": record["verifier"], "revise": revise}
    return {"critique_round": round_, "critique_revise": revise}


# targeted_search ------------------------------------------------------------------------------

def targeted_search_stage(state: "PipelineState", ctx: "PipelineContext", event: Event) -> dict[str, Any]:
    plan = PlanStore(ctx.db).latest(state["run_id"]) if state.get("plan") else None
    searches = (plan.critique or {}).get("searches", []) if plan else []
    log = run_searches(ctx, state["run_id"], searches, state.get("critique_round", 1), event) if searches else []
    event.detail["targeted_search"] = {"round": state.get("critique_round", 1), "searches": log,
                                       "plan_id": plan.plan_id if plan else None}
    return {}


# verify ------------------------------------------------------------------------------------------

def verify_stage(state: "PipelineState", ctx: "PipelineContext", event: Event) -> dict[str, Any]:
    store = PlanStore(ctx.db)
    plan = store.latest(state["run_id"]) if state.get("plan") else None
    if plan is None:
        event.detail["verify"] = {"skipped": "no plan"}
        return {}
    judgments = JudgmentStore(ctx.db, ctx.llm.models(Role.VERIFIER))
    stats = verify_elements(ctx.db, ctx.llm, judgments, plan.elements)
    store.set_verdicts(plan.elements)
    plan.detail["coverage"] = coverage(plan.elements)
    plan.detail["verify"] = stats
    store.set_detail(plan.plan_id, plan.detail)
    store.set_status(plan.plan_id, "final")
    plan.status = "final"
    event.used.append(ref("plan", plan.plan_id))
    event.generated += [ref("verdict", e.element_id) for e in plan.elements]
    for element in plan.elements:
        if element.status == "removed":
            event.drop(ref("plan_element", element.element_id),
                       f"{element.label} contradicted by its sources: {(element.verification or {}).get('reason', '')}")
    event.detail["verify"] = {"plan_id": plan.plan_id, **stats, "coverage": plan.detail["coverage"]}
    return {"plan": _summary(plan)}


# Phase 5 outside the pipeline graph -----------------------------------------------------------------

def run_phase5(ctx: "PipelineContext", state: dict[str, Any]) -> dict[str, Any]:
    """Phase 5 on a run whose ideas are assessed, with the same loop as the pipeline."""
    state = {**state, "plan_round": 0, "critique_round": 0, "critique_revise": False}

    def step(name: str, work: Any) -> None:
        nonlocal state
        with ctx.recorder.step(state["run_id"], name) as event:
            state |= work(state, ctx, event)

    step("plan", plan_stage)
    step("critique", critique_stage)
    while state.get("critique_revise") and state["critique_round"] < MAX_CRITIQUE_ROUNDS:
        step("targeted_search", targeted_search_stage)
        step("plan", plan_stage)
        step("critique", critique_stage)
    step("verify", verify_stage)
    return state
