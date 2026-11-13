"""Phase 4 check: ideas and their novelty levels for dev topics.

    uv run python scripts/novelty_check.py --runs run_ab12 run_cd34      # measure finished runs, no requests
    uv run python scripts/novelty_check.py --runs run_ab12 --rerun       # run Phase 4 again on them first
    uv run python scripts/novelty_check.py rag-knowledge-conflicts       # new full runs on these dev topics

--rerun runs ideate, novelty_search and novelty_assess again, with the retry loop,
on runs whose gaps exist (for instance runs made before Phase 4 existed), then
writes their reports again. Per run it needs the reasoning model (1-3 calls),
the extraction model (about 15 calls per idea checked, fewer when the same input
was seen before), Semantic Scholar (about 12 requests per idea) and arxiv.org
for the closest papers read in full (15 s apart).

For each run: each checked idea with its level, reasons, coverage and closest
papers; whether every comparison quote resolves to a real sentence containing it;
and Phase 4's LLM calls, API requests and time. Results are printed and saved to
var/checks/.
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
from research_agent.ideas.ideas import IdeaStore
from research_agent.ideas.stages import run_phase4
from research_agent.pipeline import RunError, default_services, start_run
from research_agent.pipeline.runner import open_pipeline
from research_agent.pipeline.stored_report import stored_report
from research_agent.report import write_report
from research_agent.storage import Database
from research_agent.trace import TraceRecorder

TOPICS = Path(__file__).parents[1] / "data" / "dev_topics.toml"
PHASE4 = ("ideate", "novelty_search", "novelty_assess")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("topics", nargs="*", help="dev topic IDs to run")
    parser.add_argument("--runs", nargs="*", default=[], help="finished runs to measure instead")
    parser.add_argument("--rerun", action="store_true", help="run Phase 4 again on --runs first")
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
                run_phase4(ctx, {"run_id": run_id, "question": question})
                write_report(stored_report(ctx.db, run_id), settings.runs_dir / run_id)
                print(f"reran Phase 4 on {run_id} in {(datetime.now() - started).seconds} s", flush=True)
                exhausted = google_quota_used_up(TraceRecorder(ctx.db).events(run_id))
                if exhausted:
                    print(f"STOP: a Google quota ran out ({exhausted[0]}). Switch RA_GOOGLE_API_KEY and rerun "
                          "the runs after this one.", flush=True)
                    break

    measured = []
    with Database(settings.db_path) as db:
        for run_id in run_ids:
            measured.append(measure(db, run_id))
            _print(measured[-1])
    out_dir = settings.data_dir / "checks"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"novelty-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(measured, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {path}")


def google_quota_used_up(events: list[dict[str, Any]]) -> list[str]:
    """In the latest Phase 4 pass: Google models a call had to skip because a daily quota ran
    out (now or earlier in the session), and embedding requests left without a vector."""
    found: list[str] = []

    def embedding_reports(value: Any) -> None:
        if isinstance(value, dict):
            if str(value.get("model", "")).startswith("google:") and value.get("missing"):
                found.append(f"embeddings: {value.get('unavailable', '')}"[:200])
            for item in value.values():
                embedding_reports(item)
        elif isinstance(value, list):
            for item in value:
                embedding_reports(item)

    steps = _latest_pass(events)
    ids = {e["event_id"] for e in steps}
    for event in (e for e in events if e["event_id"] in ids or e["parent_event_id"] in ids):
        for reason in event["detail"].get("unavailable", []) if event["activity_type"] == "llm_call" else []:
            if reason.startswith("google:") and ("quota" in reason.lower() or "earlier in this session" in reason):
                found.append(reason[:200])
        embedding_reports(event["detail"])
    return found


def _latest_pass(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The step events of the latest Phase 4 pass: from its first ideate round on."""
    starts = [i for i, e in enumerate(events) if e["activity_type"] == "step" and e["step"] == "ideate"
              and e["status"] == "ok" and e["detail"].get("ideate", {}).get("round") == 1]
    if not starts:
        return []
    return [e for e in events[starts[-1]:] if e["activity_type"] == "step" and e["step"] in PHASE4]


def phase4_cost(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Time, LLM calls and API requests of the latest Phase 4 pass."""
    steps = _latest_pass(events)
    if not steps:
        return {}
    ids = {e["event_id"] for e in steps}
    calls = [e for e in events if e["parent_event_id"] in ids]
    seconds = sum((datetime.fromisoformat(e["ended_at"]) - datetime.fromisoformat(e["started_at"])).total_seconds() for e in steps)
    llm = Counter(f"{e['agent'].get('role')}:{e['agent'].get('model')}" for e in calls if e["activity_type"] == "llm_call")
    api = Counter(e["agent"].get("name") for e in calls if e["activity_type"] == "api_call")
    return {"seconds": round(seconds), "rounds": sum(1 for e in steps if e["step"] == "ideate"),
            "llm_calls": dict(llm), "api_calls": dict(api), "failed_steps": sum(1 for e in steps if e["status"] != "ok")}


def measure(db: Database, run_id: str) -> dict[str, Any]:
    ideas = IdeaStore(db).for_run(run_id)
    rows = db.query(
        "SELECT e.sentence_id, e.quote, s.text FROM evidence e LEFT JOIN sentences s USING (sentence_id)"
        " WHERE e.owner_kind = 'comparison' AND e.owner_id IN"
        " (SELECT c.comparison_id FROM comparisons c JOIN ideas i USING (idea_id) WHERE i.run_id = ?)", (run_id,))
    # The quote check's own test (normalising LaTeX and escape debris), on the cited sentence only.
    resolved = sum(1 for r in rows if r["text"] and QuoteChecker({r["sentence_id"]: r["text"]}).locate(
        r["sentence_id"], r["quote"])[0] == "exact")
    checked = []
    for idea in sorted((i for i in ideas if i.level is not None or i.status == "rejected"), key=lambda i: (i.lineage, i.round)):
        a = idea.assessment or {}
        closest = db.query(
            "SELECT p.title, p.year, c.facet_match, c.similarity, k.depth, k.verdict FROM idea_candidates c"
            " JOIN papers p USING (paper_id) LEFT JOIN comparisons k ON k.idea_id = c.idea_id AND k.paper_id = c.paper_id"
            " WHERE c.idea_id = ? AND c.position IS NOT NULL ORDER BY c.position", (idea.idea_id,))
        checked.append({
            "idea_id": idea.idea_id, "lineage": idea.lineage, "round": idea.round, "origin": idea.origin,
            "status": idea.status, "final_rank": idea.final_rank, "title": idea.title, "purpose": idea.purpose,
            "mechanism": idea.mechanism, "evaluation": idea.evaluation, "gap": idea.detail.get("gap"),
            "level": idea.level, "inconclusive": a.get("inconclusive"), "reasons": a.get("reasons"),
            "coverage": a.get("coverage"), "compared": a.get("compared"), "read_in_full": a.get("read_in_full"),
            "full_reads": [f"{r['title'][:50]}: {r['result']}" for r in a.get("full_reads", [])],
            "quote_check": a.get("quote_check"), "changed": idea.detail.get("changed"),
            "rejected_because": idea.detail.get("rejected_because"),
            "closest": [_closest_line(r) for r in closest],
        })
    events = TraceRecorder(db).events(run_id)
    run = db.get_run(run_id)
    return {
        "run_id": run_id, "question": run["question"] if run else None, "cost": phase4_cost(events),
        "written": sum(1 for i in ideas if i.round == 1 and i.origin == "generated"),
        "levels": dict(Counter(str(i.level) for i in ideas if i.final_rank)),
        "evidence_resolved": f"{resolved}/{len(rows)}", "ideas": checked,
    }


def _closest_line(row: Any) -> str:
    verdict = json.loads(row["verdict"]) if row["verdict"] else {}
    if verdict.get("status") == "ok":
        facets = "/".join(verdict[f]["effective"] for f in ("purpose", "mechanism", "evaluation"))
        how = f"{row['depth']}: {facets}, solves gap {verdict['solves_gap']}"
    else:
        how = f"not compared ({verdict.get('error', 'no comparison')})"[:80]
    year = f" ({row['year']})" if row["year"] else ""
    return f"{row['title'][:70]}{year} [{row['facet_match']}, sim {row['similarity']:.3f}] {how}"


def _print(m: dict[str, Any]) -> None:
    print(f"\n{m['run_id']}  {m['question']}")
    print(f"  cost: {m['cost']}")
    print(f"  ideas written: {m['written']}  final levels: {m['levels']}  evidence resolved: {m['evidence_resolved']}")
    for idea in m["ideas"]:
        rank = f"#{idea['final_rank']} " if idea["final_rank"] else ""
        flag = ", inconclusive" if idea["inconclusive"] else ""
        print(f"  {rank}[lineage {idea['lineage']}, round {idea['round']}, {idea['origin']}, {idea['status']}] "
              f"level {idea['level']}{flag}: {idea['title']}")
        print(f"     purpose: {idea['purpose']}\n     mechanism: {idea['mechanism']}\n     evaluation: {idea['evaluation']}")
        if idea["changed"]:
            print(f"     (changed its {idea['changed']})")
        for reason in idea["reasons"] or []:
            print(f"     - {reason}")
        coverage = idea["coverage"] or {}
        print(f"     coverage: {coverage.get('candidates')} candidates, channels {coverage.get('channels_answered')}, "
              f"weak {coverage.get('weak')}; compared {idea['compared']}, in full {idea['read_in_full']}; quotes {idea['quote_check']}")
        for line in idea["closest"][:5]:
            print(f"       {line}")


if __name__ == "__main__":
    main()
