"""Phase 6 regression: the whole pipeline, end to end, on dev topics, as `agent run` runs it.

    uv run python scripts/regression.py                          # the 7 dev topics checked since Phase 2
    uv run python scripts/regression.py moe-load-balancing       # chosen topics from data/dev_topics.toml
    uv run python scripts/regression.py --resume run_ab12        # finish a run that stopped, then check it
    uv run python scripts/regression.py moe-load-balancing --resume run_ab12  # topics before --resume
    uv run python scripts/regression.py --runs run_ab12 run_cd34 # check finished runs, no requests

Each topic gets a new run. Work kept from earlier runs is reused (paper texts,
extractions, LLM replies, and search responses younger than RA_RESPONSE_CACHE_DAYS),
so a topic run before costs much less the second time; a topic no run has seen costs
a full run (see COMMANDS.md for the free tiers' limits).

For each run it checks what Phase 6's "done when" asks:
- it completed without manual steps: `agent run`, and `agent resume` after a stop;
- its report has every part (the Novel Evidence Report, the experiment plan with a
  Provenance Coverage figure, the gaps, the knowledge graph), and provenance.json exists;
- every evidence item the run holds resolves to a stored sentence containing its quote
  (graph nodes and edges, novelty comparisons, plan elements);
- every cited plan element traces through the PROV export to a section and a paper;
and records the run's cost: time per stage, model calls and tokens, API requests and
reuse. It stops at the first quota that runs out, saying which key to switch; the run
is left to finish with --resume.
"""

import argparse
import json
import sys
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

from research_agent.config import Settings
from research_agent.costs import run_cost
from research_agent.graph.quotes import QuoteChecker
from research_agent.pipeline import RunError, default_services, resume_run, start_run
from research_agent.plans.prov import provenance_document, trace_chains
from research_agent.plans.store import PlanStore
from research_agent.storage import Database

TOPICS = Path(__file__).parents[1] / "data" / "dev_topics.toml"
DEV_TOPICS = ("rag-knowledge-conflicts", "parameter-efficient-finetuning", "scientific-rag-retrieval",
              "long-context-efficiency", "offline-rl-overestimation", "vision-distribution-shift", "llm-as-judge-bias")
_EVIDENCE = (
    "SELECT e.owner_kind, e.sentence_id, e.quote, s.text FROM evidence e LEFT JOIN sentences s USING (sentence_id)"
    " WHERE e.owner_id IN (SELECT node_id FROM nodes WHERE run_id = ?1 UNION SELECT edge_id FROM edges WHERE run_id = ?1"
    " UNION SELECT comparison_id FROM comparisons JOIN ideas USING (idea_id) WHERE run_id = ?1"
    " UNION SELECT element_id FROM plan_elements JOIN plans USING (plan_id) WHERE run_id = ?1)"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("topics", nargs="*", help=f"dev topic IDs (default: {', '.join(DEV_TOPICS)})")
    parser.add_argument("--resume", nargs="*", default=[], metavar="RUN", help="runs to finish first")
    parser.add_argument("--runs", nargs="*", default=[], metavar="RUN", help="finished runs to check, no requests")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    settings = Settings()
    topics = {t["id"]: t for t in tomllib.loads(TOPICS.read_text(encoding="utf-8"))["topic"]}
    wanted = args.topics or ([] if args.resume or args.runs else list(DEV_TOPICS))
    unknown = [t for t in wanted if t not in topics]
    if unknown:
        parser.error(f"unknown topics: {', '.join(unknown)} (see {TOPICS.name})")

    checked: list[dict[str, Any]] = []
    run_ids = list(args.runs)
    jobs = [("resume", run_id) for run_id in args.resume] + [("run", topic) for topic in wanted]
    if jobs:
        services = default_services(settings)
        try:
            for kind, target in jobs:
                started = datetime.now()
                try:
                    if kind == "resume":
                        run_id = resume_run(settings, target, services).run_id
                    else:
                        run_id = start_run(settings, topics[target]["text"], services=services).run_id
                except RunError as exc:
                    print(f"\nSTOP: {target}: run {exc.run_id} stopped: {exc}", flush=True)
                    # Topics go before --resume, which takes every word after it as a run ID.
                    later = "".join(f"{t} " for _, t in jobs[jobs.index((kind, target)) + 1:] if _ == "run")
                    print(f"If a quota ran out, switch that provider's key in .env, then:\n"
                          f"  uv run python scripts/regression.py {later}--resume {exc.run_id}", flush=True)
                    checked.append({"target": target, "run_id": exc.run_id, "status": "stopped", "error": str(exc)})
                    break
                print(f"\n{target}: {run_id} completed in {(datetime.now() - started).seconds} s", flush=True)
                run_ids.append(run_id)
        finally:
            services.close()

    with Database(settings.db_path) as db:
        for run_id in run_ids:
            checked.append(check(db, settings, run_id))
            _print(checked[-1])
    out_dir = settings.data_dir / "checks"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"regression-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(checked, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {path}")


def check(db: Database, settings: Settings, run_id: str) -> dict[str, Any]:
    run = db.get_run(run_id)
    out_dir = settings.runs_dir / run_id
    files = {name: (out_dir / name).exists() for name in ("report.md", "report.json", "provenance.json")}
    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8")) if files["report.json"] else {}
    ner, plan_part = report.get("novel_evidence_report") or {}, report.get("experiment_plan") or {}
    parts = {
        "novel_evidence_report": bool(ner.get("proposed_idea") and ner.get("novelty_assessment")),
        "experiment_plan": (plan_part.get("provenance_coverage") or {}).get("coverage") is not None,
        "research_gaps": bool(report.get("research_gaps")),
        "knowledge_graph": bool((report.get("knowledge_graph") or {}).get("nodes")),
        "run_cost": bool((report.get("cost") or {}).get("stages")),
    }
    evidence: dict[str, list[int]] = {}
    for row in db.query(_EVIDENCE, (run_id,)):
        found = row["text"] is not None and QuoteChecker({row["sentence_id"]: row["text"]}).locate(
            row["sentence_id"], row["quote"])[0] == "exact"
        counts = evidence.setdefault(row["owner_kind"], [0, 0])
        counts[0] += found
        counts[1] += 1
    plan = PlanStore(db).latest(run_id)
    traced = {}
    if plan is not None:
        document = provenance_document(db, run_id)
        walked = trace_chains(json.loads(document.serialize(format="json")), plan.plan_id) if document else []
        cited = [w for w in walked if w["status"] != "removed" and w["chains"]]
        traced = {"cited": len(cited), "traceable": sum(1 for w in cited if w["traceable"]),
                  "uncited": sum(1 for w in walked if w["status"] != "removed" and not w["chains"])}
    cost = run_cost(db, run_id)
    replies = cost.reuse_of(exclude=("embedding",))
    return {
        "run_id": run_id, "question": run["question"], "status": run["status"], "files": files, "parts": parts,
        "evidence": {kind: f"{ok}/{total}" for kind, (ok, total) in evidence.items()},
        "plan": {"coverage": (plan.detail.get("coverage") or {}) if plan else None, **traced},
        "ideas": [{"title": i["title"], "level": i["level"]} for i in report.get("ideas", [])],
        "cost": {
            "minutes_in_stages": round(cost.seconds / 60, 1), "minutes_elapsed": round(cost.elapsed / 60, 1),
            "failed_attempts": cost.failed_attempts,
            "models": {m.model: {"calls": m.calls, "failed": m.failed, "tokens_in": m.input_tokens,
                                 "tokens_out": m.output_tokens} for m in cost.models},
            "apis": {a.service: {"requests": a.requests, "cached": a.cached} for a in cost.apis},
            "llm_replies_reused": f"{replies.reused}/{replies.reused + replies.asked}",
            "embedding_texts": {"reused": cost.reuse_of(("embedding",)).reused, "sent": cost.reuse_of(("embedding",)).asked},
            "minutes_per_stage": {s.stage: round(s.seconds / 60, 1) for s in cost.stages},
        },
    }


def _print(c: dict[str, Any]) -> None:
    print(f"\n{c['run_id']}  {c.get('question') or c.get('target')}")
    if c.get("status") == "stopped":
        print(f"  stopped: {c['error']}")
        return
    missing = [k for k, ok in {**c["files"], **c["parts"]}.items() if not ok]
    print(f"  status {c['status']}; " + (f"MISSING: {', '.join(missing)}" if missing else "every report part present"))
    print(f"  evidence resolves: {c['evidence']}")
    p = c["plan"] or {}
    cov = p.get("coverage") or {}
    print(f"  plan: coverage {cov.get('supported')}/{cov.get('elements')} ({cov.get('coverage')}); "
          f"traceable {p.get('traceable')}/{p.get('cited')} cited elements, {p.get('uncited')} uncited")
    print(f"  ideas: " + "; ".join(f"{i['title'][:60]} (level {i['level']})" for i in c["ideas"]))
    cost = c["cost"]
    print(f"  cost: {cost['minutes_in_stages']} min in stages ({cost['minutes_elapsed']} elapsed), "
          f"{cost['failed_attempts']} failed attempts; LLM replies reused {cost['llm_replies_reused']}; "
          f"embedding texts {cost['embedding_texts']}")
    for model, m in cost["models"].items():
        print(f"    {model}: {m['calls']} calls ({m['failed']} failed), {m['tokens_in']:,} in / {m['tokens_out']:,} out")
    print("    APIs: " + ", ".join(f"{s} {a['requests']} (+{a['cached']} kept)" for s, a in cost["apis"].items()))


if __name__ == "__main__":
    main()
