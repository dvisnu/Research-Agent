"""The Phase 4 stages: ``ideate``, ``novelty_search`` and ``novelty_assess``.

    ideate -> novelty_search -> novelty_assess -> (ideate again for ideas rated 1-2, at most
    MAX_ROUNDS rounds in all) -> plan

Round 1 writes ideas from the top gaps and picks IDEAS_CHECKED to check; when
the user's input is an idea, it is checked too (never revised). A later round
revises each idea rated 1-2 by changing one facet. When no idea needs another
round, each idea's best version is ranked: level first, conclusive before
inconclusive, the user's own idea before generated ones on a tie, then gap rank.

The stages keep ideas, candidates and comparisons in SQLite; the pipeline state
holds a short summary of each idea.
"""

import json
from typing import TYPE_CHECKING, Any

from research_agent.graph.store import JudgmentStore
from research_agent.ideas.compare import FULL_READS, compare, ensure_abstract, read_in_full, store_comparison
from research_agent.ideas.ideas import (
    FACETS,
    Idea,
    REPEAT_IN_LINEAGE,
    IdeaStore,
    duplicates,
    idea_facets,
    load_gaps,
    revise_ideas,
    revised,
    select,
    write_ideas,
)
from research_agent.ideas.levels import GapInfo, Verdict, assess
from research_agent.ideas.search import FILTER, coverage, run_search
from research_agent.literature.stages import is_full_text
from research_agent.llm import Role
from research_agent.storage import Database, new_id
from research_agent.trace import Event, TraceRecorder, ref

if TYPE_CHECKING:
    from research_agent.pipeline.graph import PipelineContext, PipelineState

MAX_ROUNDS = 3  # a first version and up to two retries
REVISION_PAPERS = 4  # closest papers shown when revising an idea
UNCHECKED_LINEAGE = 100  # ideas written but not checked get lineages from here


def _question(state: "PipelineState") -> str:
    brief = state.get("brief")
    return brief["rank_query"] if brief else " ".join(state["question"].split())


def latest_detail(db: Database, run_id: str, step: str) -> dict[str, Any]:
    """The detail of a step's latest successful attempt in the trace, or {}."""
    for event in reversed(TraceRecorder(db).events(run_id)):
        if event["activity_type"] == "step" and event["step"] == step and event["status"] == "ok":
            return event["detail"]
    return {}


def _summaries(db: Database, run_id: str) -> list[dict[str, Any]]:
    return [idea.summary() for idea in IdeaStore(db).for_run(run_id) if idea.status != "unchecked"]


# ideate -------------------------------------------------------------------------------

def ideate_stage(state: "PipelineState", ctx: "PipelineContext", event: Event) -> dict[str, Any]:
    run_id = state["run_id"]
    round_ = state.get("novelty_round", 0) + 1
    if round_ == 1:
        IdeaStore(ctx.db).clear_run(run_id)  # Phase 4 run again on a run starts afresh
        detail = _first_ideas(state, ctx, event)
    else:
        detail = _revise(state, ctx, event, round_)
    event.detail["ideate"] = {"round": round_, **detail}
    return {"novelty_round": round_, "ideas": _summaries(ctx.db, run_id)}


def _user_idea(state: "PipelineState", ctx: "PipelineContext") -> Idea | None:
    brief = state.get("brief") or latest_detail(ctx.db, state["run_id"], "brief").get("brief")
    if not brief or brief.get("input_kind") != "idea" or not brief.get("facets"):
        return None
    title = " ".join(state["question"].split())
    return Idea(new_id("i"), state["run_id"], 0, 1, "user", title=title,
                detail={"source": "the user's input, with the facets the research brief gave it"},
                **{facet: brief["facets"][facet] for facet in FACETS})


def _first_ideas(state: "PipelineState", ctx: "PipelineContext", event: Event) -> dict[str, Any]:
    run_id, store = state["run_id"], IdeaStore(ctx.db)
    detail: dict[str, Any] = {}
    user = _user_idea(state, ctx)
    if user is not None:
        store.add(user)
        event.generated.append(ref("idea", user.idea_id))
        detail["user_idea"] = user.idea_id
    gaps = load_gaps(ctx.db, run_id)
    detail["gaps"] = len(gaps)
    if not gaps:
        return detail
    event.used += [ref("node", gap["node_id"]) for gap in gaps]
    related = state.get("related_work") or latest_detail(ctx.db, run_id, "related_work").get("related_work", [])
    judgments = JudgmentStore(ctx.db, ctx.llm.models(Role.REASONING))
    drafts, detail["writeup"] = write_ideas(ctx.llm, judgments, _question(state), gaps, related, rejected=[])
    ideas = [
        Idea(new_id("i"), run_id, 0, 1, "generated", title=d.title, purpose=d.purpose, mechanism=d.mechanism,
             evaluation=d.evaluation, kind=d.kind,
             gap_node_id=gaps[d.gap - 1]["node_id"] if 1 <= d.gap <= len(gaps) else None,
             detail={"gap": gaps[d.gap - 1]["gap"] if 1 <= d.gap <= len(gaps) else None, "gap_rank": d.gap,
                     "builds_on": d.builds_on, "feasibility": d.feasibility})
        for d in drafts
    ]
    duplicate_of, detail["embedding"] = duplicates(ctx.db, ctx.text_embedder, [idea.text() for idea in ideas], [])
    chosen, dropped = select(drafts, len(gaps), duplicate_of)
    for lineage, i in enumerate(chosen, start=1):
        ideas[i].lineage = lineage
        store.add(ideas[i])
        event.generated.append(ref("idea", ideas[i].idea_id))
    reasons = {d["draft"]: d["reason"] for d in dropped}
    for i, idea in enumerate(ideas):
        if i not in chosen:
            idea.lineage, idea.status = UNCHECKED_LINEAGE + i, "unchecked"
            idea.detail["not_checked"] = reasons.get(i, "not among the ideas picked for checking")
            store.add(idea)
            event.drop(ref("idea", idea.idea_id), f"{idea.detail['not_checked']}: {idea.title}")
    detail.update(written=len(drafts), checked=len(chosen))
    return detail


def _closest_for_revision(ctx: "PipelineContext", idea: Idea) -> list[dict[str, Any]]:
    """The papers that set the idea's level first, then the other closest, with their facets."""
    deciding = (idea.assessment or {}).get("deciding", [])
    rows = ctx.db.query(
        "SELECT c.paper_id, c.verdict, p.title, p.year, i.position FROM comparisons c JOIN papers p USING (paper_id)"
        " JOIN idea_candidates i ON i.idea_id = c.idea_id AND i.paper_id = c.paper_id WHERE c.idea_id = ?",
        (idea.idea_id,),
    )
    rows = sorted(rows, key=lambda r: (r["paper_id"] not in deciding, r["position"] or 99))
    papers = []
    for row in rows:
        verdict = json.loads(row["verdict"])
        if verdict.get("status") != "ok":
            continue
        papers.append({"title": row["title"], "year": row["year"],
                       **{f: {"paper": verdict[f]["paper"], "match": verdict[f]["match"]} for f in FACETS}})
    return papers[:REVISION_PAPERS]


def _repeats_rejected(ctx: "PipelineContext", idea: Idea, rejected: list[Idea]) -> bool:
    """Whether a revised idea is one already rejected. Versions of one idea share two facets,
    so within its lineage only the changed facet is compared, and only a near-verbatim repeat
    counts (REPEAT_IN_LINEAGE); other ideas are compared whole."""
    facet = idea.detail["changed"]
    own = [getattr(r, facet) for r in rejected if r.lineage == idea.lineage]
    others = [r.text() for r in rejected if r.lineage != idea.lineage]
    by_facet, _ = duplicates(ctx.db, ctx.text_embedder, [getattr(idea, facet)], own, REPEAT_IN_LINEAGE)
    whole, _ = duplicates(ctx.db, ctx.text_embedder, [idea.text()], others)
    return by_facet[0] is not None or whole[0] is not None


def _revise(state: "PipelineState", ctx: "PipelineContext", event: Event, round_: int) -> dict[str, Any]:
    run_id, store = state["run_id"], IdeaStore(ctx.db)
    low = [i for i in store.for_run(run_id, "assessed")
           if i.round == round_ - 1 and i.origin in ("generated", "revised") and (i.level or 0) <= 2]
    if not low:
        return {"revised": 0}
    items = [{"idea": idea, "gap": idea.detail.get("gap"), "papers": _closest_for_revision(ctx, idea)} for idea in low]
    earlier = store.for_run(run_id, "rejected")
    rejected = [f"{i.title}: {i.mechanism}" for i in earlier]
    judgments = JudgmentStore(ctx.db, ctx.llm.models(Role.REASONING))
    revisions, detail = revise_ideas(ctx.llm, judgments, items, rejected)
    new: list[Idea] = []
    for number, idea in enumerate(low, start=1):
        store.set_status(idea.idea_id, "rejected")  # remembered, so it isn't proposed again
        event.used.append(ref("idea", idea.idea_id))
        if number in revisions:
            new.append(revised(idea, revisions[number], round_))
        else:
            event.drop(ref("idea", idea.idea_id), "the revision left this idea out")
    rejected_all = [*earlier, *low]
    for idea in new:
        if _repeats_rejected(ctx, idea, rejected_all):
            idea.status = "rejected"
            idea.detail["rejected_because"] = "repeats an earlier version or a rejected idea"
            event.drop(ref("idea", idea.idea_id), f"repeats a rejected idea: {idea.title}")
        store.add(idea)
        event.generated.append(ref("idea", idea.idea_id))
    return {**detail, "low": len(low), "new_versions": sum(1 for i in new if i.status == "pending")}


# novelty_search ------------------------------------------------------------------------------

def novelty_search_stage(state: "PipelineState", ctx: "PipelineContext", event: Event) -> dict[str, Any]:
    """A fresh search for each idea waiting for one."""
    run_id, store = state["run_id"], IdeaStore(ctx.db)
    method = state.get("novelty_filter") or FILTER
    log = []
    for idea in store.for_run(run_id, "pending"):
        event.used.append(ref("idea", idea.idea_id))
        search = run_search(ctx, idea, run_id, method, event)
        idea.detail["coverage"] = coverage(search, ctx.sources.s2 is not None)
        store.set_detail(idea)
        store.set_status(idea.idea_id, "searched")
        log.append({"idea": idea.idea_id, "title": idea.title, "channels": search.log, "filter": search.filter_info,
                    "rerank": search.rerank_info, "coverage": idea.detail["coverage"],
                    "closest": [{"paper": c.paper_id, "title": c.title, "similarity": round(c.similarity or 0, 4),
                                 "match": c.facet_match} for c in search.closest]})
    event.detail["novelty_search"] = log
    return {"ideas": _summaries(ctx.db, run_id)}


# novelty_assess --------------------------------------------------------------------------------

def _gap_info(gap: dict[str, Any] | None) -> GapInfo:
    if gap is None:
        return GapInfo(None)
    years = [e["year"] for e in gap["evidence"] if e["year"]]
    return GapInfo(gap["kind"], gap["papers"], max(years) if years else None)


def assess_idea(ctx: "PipelineContext", idea: Idea, gaps: dict[str, dict[str, Any]], event: Event) -> dict[str, Any]:
    judgments = JudgmentStore(ctx.db, ctx.llm.models(Role.EXTRACTION))
    closest = [dict(row) for row in ctx.db.query(
        "SELECT i.paper_id, i.position, p.title, p.year FROM idea_candidates i JOIN papers p USING (paper_id)"
        " WHERE i.idea_id = ? AND i.position IS NOT NULL ORDER BY i.position", (idea.idea_id,))]
    gap = gaps.get(idea.gap_node_id or "")
    gap_text = gap["gap"] if gap else idea.detail.get("gap")
    gap_info = _gap_info(gap)
    weak = (idea.detail.get("coverage") or {}).get("weak", [])

    verdicts: dict[str, dict[str, Any]] = {}
    for paper in closest:
        ensure_abstract(ctx, paper["paper_id"])
        verdicts[paper["paper_id"]] = compare(ctx, judgments, idea, gap_text, paper["paper_id"], "abstract")
        store_comparison(ctx, idea.idea_id, paper["paper_id"], verdicts[paper["paper_id"]])

    def level() -> Any:
        return assess([Verdict.from_comparison(p, verdicts[p["paper_id"]]) for p in closest], gap_info, weak)

    result = level()
    full_reads = []
    if result.level >= 3:  # read the closest papers in full before claiming novelty
        for paper in closest[:FULL_READS]:
            read = {"paper": paper["paper_id"], "title": paper["title"], **read_in_full(ctx, paper["paper_id"])}
            source = ctx.db.query("SELECT text_source FROM papers WHERE paper_id = ?", (paper["paper_id"],))[0]["text_source"]
            if is_full_text(source):
                verdict = compare(ctx, judgments, idea, gap_text, paper["paper_id"], "full")
                read["compared"] = verdict["status"]
                if verdict["status"] == "ok":
                    verdicts[paper["paper_id"]] = verdict
                    store_comparison(ctx, idea.idea_id, paper["paper_id"], verdict)
                else:
                    read["error"] = verdict.get("error")
            full_reads.append(read)
        result = level()
    event.used += [ref("paper", p["paper_id"]) for p in closest]
    quotes = {"quotes": 0, "passed": 0}
    for verdict in verdicts.values():
        for key in quotes:
            quotes[key] += verdict.get("quote_check", {}).get(key, 0)
    return {
        **result.as_dict(),
        "coverage": idea.detail.get("coverage"),
        "compared": sum(1 for v in verdicts.values() if v.get("status") == "ok"),
        "read_in_full": sum(1 for v in verdicts.values() if v.get("status") == "ok" and v["depth"] == "full"),
        "full_reads": full_reads,
        "quote_check": quotes,
        "gap": {"node_id": idea.gap_node_id, "text": gap_text, "kind": gap_info.kind, "papers": gap_info.papers,
                "newest_year": gap_info.newest},
    }


def novelty_assess_stage(state: "PipelineState", ctx: "PipelineContext", event: Event) -> dict[str, Any]:
    """Compare each searched idea with its closest papers and assign its level; decide
    whether another round is needed, and rank the ideas when not."""
    run_id, store = state["run_id"], IdeaStore(ctx.db)
    round_ = state.get("novelty_round", 1)
    gaps = {gap["node_id"]: gap for gap in load_gaps(ctx.db, run_id)}
    log = []
    for idea in store.for_run(run_id, "searched"):
        assessment = assess_idea(ctx, idea, gaps, event)
        store.set_assessment(idea.idea_id, assessment["level"], assessment)
        event.generated.append(ref("assessment", idea.idea_id))
        log.append({"idea": idea.idea_id, "title": idea.title, "level": assessment["level"],
                    "inconclusive": assessment["inconclusive"], "reasons": assessment["reasons"],
                    "compared": assessment["compared"], "read_in_full": assessment["read_in_full"]})
    low = [i for i in store.for_run(run_id, "assessed")
           if i.round == round_ and i.origin in ("generated", "revised") and (i.level or 0) <= 2]
    retry = bool(low) and round_ < MAX_ROUNDS
    options = getattr(ctx, "options", None)
    if options is not None and options.force_loops and round_ < 2:
        retry = True
    if not retry:
        ranked = finalize(store, run_id)
        event.detail["final_ranking"] = ranked
    event.detail["novelty_assess"] = {"round": round_, "ideas": log, "retry": retry}
    return {"novelty_retry": retry, "ideas": _summaries(ctx.db, run_id)}


def finalize(store: IdeaStore, run_id: str) -> list[str]:
    """Rank each lineage's best version (highest level, then latest)."""
    best: dict[int, Idea] = {}
    for idea in store.for_run(run_id):
        if idea.level is None:
            continue
        held = best.get(idea.lineage)
        if held is None or (idea.level, idea.round) > (held.level or 0, held.round):
            best[idea.lineage] = idea
    ranked = sorted(best.values(), key=lambda i: (
        -(i.level or 0), bool((i.assessment or {}).get("inconclusive")), i.origin != "user",
        i.detail.get("gap_rank", 99), i.lineage))
    store.set_final_ranks(run_id, [i.idea_id for i in ranked])
    return [i.idea_id for i in ranked]


# Phase 4 outside the pipeline graph -------------------------------------------------------------

STEPS = (("ideate", ideate_stage), ("novelty_search", novelty_search_stage), ("novelty_assess", novelty_assess_stage))


def run_phase4(ctx: "PipelineContext", state: dict[str, Any]) -> dict[str, Any]:
    """Phase 4 on a run whose gaps exist, with the same loop as the pipeline."""
    state = {**state, "novelty_round": 0}
    while True:
        for step, work in STEPS:
            with ctx.recorder.step(state["run_id"], step) as event:
                state |= work(state, ctx, event)
        if not (state.get("novelty_retry") and state["novelty_round"] < MAX_ROUNDS):
            return state


def check_idea(ctx: "PipelineContext", run_id: str, origin: str, *, facets: dict[str, str] | None = None,
               text: str | None = None, detail: dict[str, Any] | None = None, method: str | None = None) -> Idea:
    """Check one given idea in a run of its own: search, compare, level. No revisions. The
    idea comes as facets (title, purpose, mechanism, evaluation), or as free text that the
    extraction model splits into them."""
    store = IdeaStore(ctx.db)
    with ctx.recorder.step(run_id, "ideate") as event:
        if facets is None:
            facets = idea_facets(ctx.llm, JudgmentStore(ctx.db, ctx.llm.models(Role.EXTRACTION)), text or "").model_dump()
        idea = store.add(Idea(new_id("i"), run_id, 0, 1, origin, detail=detail or {}, **facets))
        event.generated.append(ref("idea", idea.idea_id))
        event.detail["ideate"] = {"round": 1, origin: idea.idea_id}
    state: dict[str, Any] = {"run_id": run_id, "question": idea.title, "novelty_round": 1}
    if method:
        state["novelty_filter"] = method
    for step, work in STEPS[1:]:
        with ctx.recorder.step(run_id, step) as event:
            state |= work(state, ctx, event)
    return store.get(idea.idea_id)
