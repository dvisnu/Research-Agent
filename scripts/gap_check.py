"""Phase 3 check: related work, the landscape grid and ranked gaps for dev topics.

    uv run python scripts/gap_check.py rag-knowledge-conflicts          # new runs on these dev topics
    uv run python scripts/gap_check.py --runs run_ab12 run_cd34         # measure finished runs, no requests
    uv run python scripts/gap_check.py --runs run_ab12 --rerun          # run Phase 3 again on them first

--rerun runs the related_work, landscape and gaps stages again on runs whose graph
is built, for instance runs made before Phase 3 existed or after a change to it. It
needs the reasoning model (about 2 calls a run) and about 40-60 embedding requests a
run for gap relevance, fewer when the same input was seen before. The trace records it as
another attempt of each stage; the run's report files are not rewritten.

For each run: the gaps with their kind, warrant and evidence, whether every piece of
evidence resolves to a real sentence that contains its quote, the candidates found,
the closest papers, the grid, and the stages' LLM calls and time. Results are printed
and saved to var/checks/.
"""

import argparse
import json
import sys
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

from research_agent.config import Settings
from research_agent.gaps.stages import gaps_stage, landscape_stage, related_work_stage
from research_agent.graph.quotes import normalize
from research_agent.pipeline import RunError, default_services, start_run
from research_agent.pipeline.runner import open_pipeline
from research_agent.storage import Database
from research_agent.trace import TraceRecorder

TOPICS = Path(__file__).parents[1] / "data" / "dev_topics.toml"
STAGES = (("related_work", related_work_stage), ("landscape", landscape_stage), ("gaps", gaps_stage))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("topics", nargs="*", help="dev topic IDs to run")
    parser.add_argument("--runs", nargs="*", default=[], help="finished runs to measure instead")
    parser.add_argument("--rerun", action="store_true", help="run Phase 3 again on --runs first")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # paper text has symbols cp1252 lacks
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
                state: dict[str, Any] = {"run_id": run_id, "question": question}
                for step, work in STAGES:
                    with ctx.recorder.step(run_id, step) as event:
                        state |= work(state, ctx, event)
                print(f"reran Phase 3 on {run_id}", flush=True)

    measured = []
    with Database(settings.db_path) as db:
        for run_id in run_ids:
            measured.append(_measure(db, run_id))
            _print(measured[-1])
    out_dir = settings.data_dir / "checks"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"gaps-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(measured, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {path}")


def _measure(db: Database, run_id: str) -> dict[str, Any]:
    events = TraceRecorder(db).events(run_id)
    steps = {e["step"]: e for e in events if e["activity_type"] == "step" and e["status"] == "ok"}  # latest attempts
    cost = {}
    for step, _ in STAGES:
        if step in steps:
            event = steps[step]
            seconds = (datetime.fromisoformat(event["ended_at"]) - datetime.fromisoformat(event["started_at"])).total_seconds()
            calls = [e for e in events if e["parent_event_id"] == event["event_id"] and e["activity_type"] == "llm_call"]
            cost[step] = {"seconds": round(seconds), "llm_calls": len(calls),
                          "models": sorted({e["agent"].get("model", "") for e in calls})}
    gap_rows = db.query("SELECT node_id, name, attrs FROM nodes WHERE run_id = ? AND node_type = 'gap'", (run_id,))
    gaps, checked, resolved = [], 0, 0
    for row in sorted(gap_rows, key=lambda r: json.loads(r["attrs"])["rank"]):
        attrs = json.loads(row["attrs"])
        evidence = db.query(
            "SELECT e.sentence_id, e.quote, s.text, p.title FROM evidence e LEFT JOIN sentences s USING (sentence_id)"
            " LEFT JOIN sections c ON c.section_id = s.section_id LEFT JOIN papers p ON p.paper_id = c.paper_id"
            " WHERE e.owner_kind = 'node' AND e.owner_id = ?", (row["node_id"],))
        for e in evidence:
            checked += 1
            resolved += bool(e["text"]) and normalize(e["quote"]).strip(" .\"'") in normalize(e["text"])
        gaps.append({"rank": attrs["rank"], "kind": attrs["kind"], "gap": row["name"], "warrant": attrs.get("warrant"),
                     "relevance": attrs["relevance"], "papers": len(attrs["papers"]),
                     "evidence": [{"sentence_id": e["sentence_id"], "quote": e["quote"], "paper": e["title"]} for e in evidence]})
    detail = steps.get("gaps", {}).get("detail", {}).get("gaps", {})
    landscape = steps.get("landscape", {}).get("detail", {}).get("landscape", {})
    related = steps.get("related_work", {}).get("detail", {}).get("related_work", [])
    run = db.get_run(run_id)
    return {
        "run_id": run_id,
        "question": run["question"] if run else None,
        "cost": cost,
        "candidates": detail.get("candidates"),
        "structural_cells": detail.get("structural_cells"),
        "writeup": detail.get("writeup"),
        "evidence_resolved": f"{resolved}/{checked}",
        "landscape": landscape,
        "related_work": [f"{p['title'][:70]} ({p['similarity']:.3f})" for p in related[:5]],
        "gaps": gaps,
    }


def _print(m: dict[str, Any]) -> None:
    print(f"\n{m['run_id']}  {m['question']}")
    print(f"  cost: {m['cost']}")
    print(f"  candidates: {m['candidates']}  structural: {m['structural_cells']}  write-up: {m['writeup']}")
    print(f"  evidence resolved: {m['evidence_resolved']}  landscape: {m['landscape']}")
    print("  related work: " + " | ".join(m["related_work"]))
    for gap in m["gaps"]:
        print(f"  {gap['rank']}. [{gap['kind']}, relevance {gap['relevance']:.3f}, {gap['papers']} paper(s)] {gap['gap']}")
        print(f"     warrant: {gap['warrant']}")
        for e in gap["evidence"][:3]:
            print(f"     - \"{e['quote'][:140]}\" ({(e['paper'] or '?')[:50]})")


if __name__ == "__main__":
    main()
