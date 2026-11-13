"""Phase 5 check: experiment plans and their provenance for dev topics.

    uv run python scripts/plan_check.py --runs run_ab12 run_cd34      # measure finished runs, no requests
    uv run python scripts/plan_check.py --runs run_ab12 --rerun       # run Phase 5 again on them first
    uv run python scripts/plan_check.py rag-knowledge-conflicts       # new full runs on these dev topics

--rerun runs plan, critique, targeted_search and verify again, with the critique loop,
on runs whose ideas are assessed (for instance runs made before Phase 5 existed), then
writes their reports and provenance again. Per run it needs the reasoning model (1-2
calls), the extraction model (about 10-20 calls for papers the run had not extracted,
fewer when kept from earlier), the verifier model (about 5-10 calls) and Semantic
Scholar (a few requests per targeted search). It stops, saying so, when a Google or
Groq quota runs out.

For each run: the plan, element by element, with its verdict; Provenance Coverage;
whether every element has its three provenance records; whether every evidence quote
resolves to a real sentence containing it; whether the PROV export traces every cited
element to a section and a paper; the critique points and targeted searches; and
Phase 5's LLM calls, API requests and time. Results are printed and saved to var/checks/.
"""

import argparse
import json
import sys
import tomllib
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from research_agent.config import Settings
from research_agent.graph.quotes import QuoteChecker
from research_agent.llm import LLMUnavailable
from research_agent.pipeline import RunError, default_services, start_run
from research_agent.pipeline.runner import open_pipeline
from research_agent.pipeline.stored_report import stored_report
from research_agent.plans.prov import provenance_document, trace_chains, write_provenance
from research_agent.plans.stages import run_phase5
from research_agent.plans.store import PlanStore
from research_agent.report import write_report
from research_agent.storage import Database
from research_agent.trace import TraceRecorder

TOPICS = Path(__file__).parents[1] / "data" / "dev_topics.toml"
PHASE5 = ("plan", "critique", "targeted_search", "verify")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("topics", nargs="*", help="dev topic IDs to run")
    parser.add_argument("--runs", nargs="*", default=[], help="finished runs to measure instead")
    parser.add_argument("--rerun", action="store_true", help="run Phase 5 again on --runs first")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    settings = Settings()

    run_ids = list(args.runs)
    if args.topics:
        topics = {t["id"]: t for t in tomllib.loads(TOPICS.read_text(encoding="utf-8"))["topic"]}
        unknown = [t for t in args.topics if t not in topics]
        if unknown:
            parser.error(f"unknown topics: {', '.join(unknown)} (see {TOPICS.name})")
        services = default_services(settings)
        try:
            for topic_id in args.topics:
                try:
                    run_ids.append(start_run(settings, topics[topic_id]["text"], services=services).run_id)
                except RunError as exc:
                    print(f"{topic_id}: FAILED ({exc.run_id}): {exc}", flush=True)
        finally:
            services.close()
    if not run_ids:
        parser.error("give dev topic IDs or --runs")
    if args.rerun:
        with open_pipeline(settings) as (_, ctx):
            for run_id in args.runs:
                question = ctx.db.get_run(run_id)["question"]
                started = datetime.now()
                try:
                    run_phase5(ctx, {"run_id": run_id, "question": question})
                except LLMUnavailable as exc:
                    print(f"STOP: {run_id}: a model is unavailable ({exc}). If a quota ran out, switch its key "
                          "and rerun this run and the ones after it.", flush=True)
                    break
                out_dir = settings.runs_dir / run_id
                write_report(stored_report(ctx.db, run_id), out_dir)
                write_provenance(ctx.db, run_id, out_dir)
                print(f"reran Phase 5 on {run_id} in {(datetime.now() - started).seconds} s", flush=True)
                exhausted = quota_used_up(TraceRecorder(ctx.db).events(run_id))
                if exhausted:
                    print(f"STOP: a quota ran out ({exhausted[0]}). Switch the key and rerun the runs after this one.",
                          flush=True)
                    break

    measured = []
    with Database(settings.db_path) as db:
        for run_id in run_ids:
            measured.append(measure(db, run_id))
            _print(measured[-1])
    out_dir = settings.data_dir / "checks"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"plans-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(measured, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {path}")


def _latest_pass(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The step events of the latest Phase 5 pass: from its first plan round on."""
    starts = [i for i, e in enumerate(events) if e["activity_type"] == "step" and e["step"] == "plan"
              and e["status"] == "ok" and e["detail"].get("plan", {}).get("round") == 1]
    if not starts:
        return []
    return [e for e in events[starts[-1]:] if e["activity_type"] == "step" and e["step"] in PHASE5]


def quota_used_up(events: list[dict[str, Any]]) -> list[str]:
    """In the latest Phase 5 pass: Google or Groq models a call had to skip because a quota ran
    out, and embedding requests left without a vector."""
    found: list[str] = []
    ids = {e["event_id"] for e in _latest_pass(events)}
    for event in (e for e in events if e["event_id"] in ids or e["parent_event_id"] in ids):
        for reason in event["detail"].get("unavailable", []) if event["activity_type"] == "llm_call" else []:
            if reason.startswith(("google:", "groq:")) and ("quota" in reason.lower() or "earlier in this session" in reason
                                                            or "per day" in reason.lower()):
                found.append(reason[:200])
    return found


def phase5_cost(events: list[dict[str, Any]]) -> dict[str, Any]:
    steps = _latest_pass(events)
    if not steps:
        return {}
    ids = {e["event_id"] for e in steps}
    calls = [e for e in events if e["parent_event_id"] in ids]
    seconds = sum((datetime.fromisoformat(e["ended_at"]) - datetime.fromisoformat(e["started_at"])).total_seconds() for e in steps)
    llm = Counter(f"{e['agent'].get('role')}:{e['agent'].get('model')}" for e in calls if e["activity_type"] == "llm_call")
    tokens = Counter()
    for e in calls:
        if e["activity_type"] == "llm_call":
            for key, value in (e["detail"].get("usage") or {}).items():
                tokens[f"{e['agent'].get('role')}:{key}"] += value
    api = Counter(e["agent"].get("name") for e in calls if e["activity_type"] == "api_call")
    per_step = Counter()
    for e in steps:
        per_step[e["step"]] += round((datetime.fromisoformat(e["ended_at"]) - datetime.fromisoformat(e["started_at"])).total_seconds())
    return {"seconds": round(seconds), "per_step_seconds": dict(per_step), "llm_calls": dict(llm), "tokens": dict(tokens),
            "api_calls": dict(api), "failed_steps": sum(1 for e in steps if e["status"] != "ok")}


def measure(db: Database, run_id: str) -> dict[str, Any]:
    run = db.get_run(run_id)
    versions = PlanStore(db).versions(run_id)
    events = TraceRecorder(db).events(run_id)
    if not versions:
        skipped = [e["detail"].get("plan", {}) for e in events if e["activity_type"] == "step" and e["step"] == "plan"]
        return {"run_id": run_id, "question": run["question"] if run else None, "plan": None,
                "why": skipped[-1] if skipped else "no plan stage ran"}
    plan = versions[-1]
    rows = db.query(
        "SELECT e.sentence_id, e.quote, s.text FROM evidence e LEFT JOIN sentences s USING (sentence_id)"
        " WHERE e.owner_kind = 'plan_element' AND e.owner_id IN (SELECT element_id FROM plan_elements WHERE plan_id = ?)",
        (plan.plan_id,))
    resolved = sum(1 for r in rows if r["text"] and QuoteChecker({r["sentence_id"]: r["text"]}).locate(
        r["sentence_id"], r["quote"])[0] == "exact")
    document = json.loads(provenance_document(db, run_id).serialize(format="json"))
    walked = trace_chains(document, plan.plan_id)
    final = [w for w in walked if w["status"] != "removed"]
    records = {
        "source": sum(1 for e in plan.elements if e.evidence),
        "derivation": sum(1 for e in plan.elements if e.derivation.get("step") and e.derivation.get("model")),
        "decision": sum(1 for e in plan.elements if e.decision.get("reason")),
        "elements": len(plan.elements),
    }
    critiques = [{"round": v.round, "answers_gap": (v.critique or {}).get("answers_gap"),
                  "points": [f"[{p['source']}:{p['kind']}] ({p['element']}) {p['problem']}" + (f" -> search {p['search']!r}" if p["search"] else "")
                             for p in (v.critique or {}).get("points", [])],
                  "verifier": (v.critique or {}).get("verifier", {}).get("model") or (v.critique or {}).get("verifier", {}).get("error")}
                 for v in versions if v.critique is not None]
    searches = [e["detail"]["targeted_search"] for e in _latest_pass(events) if e["step"] == "targeted_search"]
    search_papers = db.query("SELECT p.title, r.detail FROM plan_papers r JOIN papers p USING (paper_id)"
                             " WHERE r.run_id = ? AND r.role = 'search'", (run_id,))
    papers = db.query("SELECT role, COUNT(*) AS n FROM plan_papers WHERE run_id = ? GROUP BY role", (run_id,))
    return {
        "run_id": run_id, "question": run["question"] if run else None, "plan_id": plan.plan_id, "rounds": len(versions),
        "idea": db.query("SELECT title, level FROM ideas WHERE idea_id = ?", (plan.idea_id,))[0]["title"],
        "coverage": plan.detail.get("coverage"), "verify": plan.detail.get("verify"),
        "records": records, "evidence_resolved": f"{resolved}/{len(rows)}",
        "traceable": f"{sum(1 for w in final if w['traceable'])}/{len(final)}",
        "untraceable": [w["field"] for w in final if not w["traceable"]],
        "menu": plan.detail.get("menu"), "plan_papers": {r["role"]: r["n"] for r in papers},
        "critiques": critiques,
        "searches": [{"query": s["query"], "found": s.get("found"), "kept": [k["title"][:70] for k in s.get("kept", [])]}
                     for entry in searches for s in entry.get("searches", []) if "query" in s],
        "search_papers": [{"title": r["title"][:80], "chunk": json.loads(r["detail"]).get("chunk"),
                           "extractions": len(json.loads(r["detail"]).get("extractions", []))} for r in search_papers],
        "elements": [{"label": e.label, "status": e.status, "verdict": e.verdict, "text": e.text, "name": e.name,
                      "basis": e.basis, "cites": e.cites, "change": e.derivation.get("change"),
                      "checks": e.derivation.get("checks"), "decision": e.decision.get("reason"),
                      "reason": (e.verification or {}).get("reason")} for e in plan.elements],
        "cost": phase5_cost(events),
    }


def _print(m: dict[str, Any]) -> None:
    print(f"\n{m['run_id']}  {m['question']}")
    if m.get("plan") is None and "plan_id" not in m:
        print(f"  no plan: {m['why']}")
        return
    c = m["coverage"] or {}
    print(f"  idea: {m['idea']}")
    print(f"  rounds {m['rounds']}; coverage {c.get('supported')}/{c.get('elements')} = {c.get('coverage')}; "
          f"labels {c.get('labels')}; removed {c.get('removed')}")
    print(f"  records {m['records']}; evidence resolved {m['evidence_resolved']}; traceable {m['traceable']} "
          f"(not: {m['untraceable']})")
    print(f"  menu {m['menu']}; plan papers {m['plan_papers']}")
    for critique in m["critiques"]:
        print(f"  critique round {critique['round']} ({critique['verifier']}), answers gap: {critique['answers_gap']}")
        for point in critique["points"]:
            print(f"     - {point}")
    for search in m["searches"]:
        print(f"  search {search['query']!r}: {search['found']} found, kept {search['kept']}")
    for element in m["elements"]:
        flag = "" if element["verdict"] == "supported" else f"  [{element['verdict']}: {element['reason']}]"
        print(f"  {element['label']:<20} {element['text'][:110]} {element['cites']}{flag}")
    print(f"  cost: {m['cost']}")


if __name__ == "__main__":
    main()
