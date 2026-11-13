"""Phase 2 check: build knowledge graphs for dev topics and measure them.

    uv run python scripts/graph_check.py rag-knowledge-conflicts     # new runs on these dev topics
    uv run python scripts/graph_check.py --runs run_ab12 run_cd34    # measure finished runs, no requests
    uv run python scripts/graph_check.py --runs run_ab12 --sample 20 # and write a sample to check by hand
    uv run python scripts/graph_check.py --runs run_ab12 --rebuild   # build the graphs again first

For each run: the quote check's pass rate, how often each field of the form was
filled, names merged, cross-paper links, LLM calls and time, and three example
queries (methods on the most-used dataset, limitations for the most-addressed task,
contradictions). A new run on a topic costs about 70-100 extraction-model requests
and up to about 400 embedding requests (1,000 a day), less where extractions and
vectors are kept from earlier runs.

--rebuild runs the build_graph stage of the --runs again, for instance after a change
to the quote check. It reuses the kept extractions, name groupings, pair labels and
vectors, so it only asks the models about what is new. The trace records it as another
attempt of the stage; the run's report files are not rewritten.

--sample N writes N extracted items, picked at random, with the sentences they cite,
as a Markdown checklist in var/checks/, for checking by hand that each item says what
its sentences say. Results are printed and saved to var/checks/.
"""

import argparse
import json
import random
import sys
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

from research_agent.config import Settings
from research_agent.graph import queries
from research_agent.graph.build import STATEMENT_KINDS, checked_facts, load_graph
from research_agent.graph.stages import build_graph_stage
from research_agent.pipeline import RunError, default_services, start_run
from research_agent.pipeline.runner import open_pipeline
from research_agent.storage import Database
from research_agent.trace import Event, TraceRecorder

TOPICS = Path(__file__).parents[1] / "data" / "dev_topics.toml"
FIELDS = ("tasks", "methods", "datasets", "metrics", *STATEMENT_KINDS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("topics", nargs="*", help="dev topic IDs to run")
    parser.add_argument("--runs", nargs="*", default=[], help="finished runs to measure instead")
    parser.add_argument("--sample", type=int, default=0, help="write this many items to check by hand")
    parser.add_argument("--rebuild", action="store_true", help="build the graphs of --runs again first")
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
    if args.rebuild:
        with open_pipeline(settings) as (_, ctx):
            for run_id in args.runs:
                with ctx.recorder.step(run_id, "build_graph") as event:
                    build_graph_stage({"run_id": run_id}, ctx, event)
                print(f"rebuilt {run_id}", flush=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = settings.data_dir / "checks"
    out_dir.mkdir(parents=True, exist_ok=True)
    measured = []
    with Database(settings.db_path) as db:
        for run_id in run_ids:
            measured.append(_measure(db, run_id))
            _print(measured[-1])
        if args.sample:
            path = out_dir / f"graph_sample-{stamp}.md"
            path.write_text(_sample(db, run_ids, args.sample), encoding="utf-8")
            print(f"\nHand-check sample: {path}")
    path = out_dir / f"graph-{stamp}.json"
    path.write_text(json.dumps(measured, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {path}")


def _facts(db: Database, run_id: str) -> dict:
    scratch = Event(run_id, "check", "step", {}, [], None, None, None)  # not recorded
    return checked_facts(db, run_id, scratch)[0]


def _measure(db: Database, run_id: str) -> dict[str, Any]:
    run = db.get_run(run_id)
    events = TraceRecorder(db).events(run_id)
    steps = {e["step"]: e for e in events if e["activity_type"] == "step" and e["status"] == "ok"}
    graph_summary = steps.get("build_graph", {}).get("detail", {}).get("graph", {})
    levels = {row["paper_id"]: row["read_level"] for row in db.query(
        "SELECT paper_id, read_level FROM run_papers WHERE run_id = ?", (run_id,))}
    facts = _facts(db, run_id)

    filled: dict[str, dict[str, float]] = {}
    for level in ("full", "abstract"):
        papers = [f for pid, f in facts.items() if levels.get(pid) == level]
        if papers:
            filled[level] = {
                name: round(sum(bool(p.lists.get(name)) for p in papers) / len(papers), 2) for name in FIELDS
            }
            filled[level]["facets"] = round(
                sum(sum(bool(v) for v in p.facets.values()) for p in papers) / (3 * len(papers)), 2
            )

    timing, calls = {}, {}
    for step in ("extract", "build_graph"):
        if step in steps:
            event = steps[step]
            timing[step] = round((datetime.fromisoformat(event["ended_at"])
                                  - datetime.fromisoformat(event["started_at"])).total_seconds())
            calls[step] = sum(1 for e in events if e["parent_event_id"] == event["event_id"] and e["activity_type"] == "llm_call")

    graph = load_graph(db, run_id)
    examples: dict[str, Any] = {}
    top = _most_linked(graph, "dataset")
    if top:
        examples[f"methods on {top}"] = sorted({a.subject for a in queries.methods_on_dataset(graph, db, top)})[:10]
    top = _most_linked(graph, "task")
    if top:
        examples[f"limitations for {top}"] = [a.detail for a in queries.limitations_for_task(graph, db, top)][:5]
    examples["contradictions"] = [f"{a.subject} | {a.detail}" for a in queries.conflicts(graph, db)][:5]

    return {
        "run_id": run_id,
        "question": run["question"] if run else None,
        "seconds": timing,
        "llm_calls": calls,
        "extraction": steps.get("extract", {}).get("detail", {}).get("extraction", {}),
        "quote_check": graph_summary.get("quote_check", {}),
        "fields_filled": filled,
        "nodes": graph_summary.get("nodes", {}),
        "edges": graph_summary.get("edges", {}),
        "names": graph_summary.get("names", {}),
        "links": graph_summary.get("links", {}),
        "examples": examples,
    }


def _most_linked(graph: Any, node_type: str) -> str | None:
    papers = {n for n, d in graph.nodes(data=True) if d["node_type"] == "paper"}
    ranked = sorted(
        ((len({src for src, _ in graph.in_edges(n) if src in papers}), d["name"])
         for n, d in graph.nodes(data=True) if d["node_type"] == node_type),
        reverse=True,
    )
    return ranked[0][1] if ranked else None


def _print(m: dict[str, Any]) -> None:
    q = m["quote_check"]
    print(f"\n{m['run_id']}  {m['question']}")
    print(f"  time: {m['seconds']}  LLM calls: {m['llm_calls']}  extraction: {m['extraction']}")
    if q:
        print(f"  quote check: {q['pass_rate']:.1%} of {q['quotes']} quotes "
              f"({q['exact']} exact, {q['readdressed']} readdressed, {q['failed']} failed); "
              f"items kept {q['items_kept']}/{q['items']}")
    for level, rates in m["fields_filled"].items():
        print(f"  filled ({level}): " + ", ".join(f"{k} {v:.0%}" for k, v in rates.items()))
    print(f"  nodes: {m['nodes']}")
    print(f"  edges: {m['edges']}")
    links = {k: v for k, v in m["links"].items() if k != "embedding"}
    print(f"  links: {links}")
    print(f"  embedding: {m['links'].get('embedding')}")
    for question, answer in m["examples"].items():
        print(f"  {question}: {answer}")


def _sample(db: Database, run_ids: list[str], size: int) -> str:
    """Random items across the runs, each with the sentences it cites."""
    items = []
    for run_id in run_ids:
        for paper in _facts(db, run_id).values():
            for name, values in paper.lists.items():
                items += [(run_id, paper.title, name, item) for item in values]
            items += [(run_id, paper.title, f"facet: {k}", v) for k, v in paper.facets.items() if v]
    picked = random.Random(0).sample(items, min(size, len(items)))
    lines = ["# Knowledge graph: hand-check sample", "",
             "Tick an item when its sentences really say it (and its name, role or numbers are right).", ""]
    for number, (run_id, title, field_name, item) in enumerate(picked, start=1):
        label = item.get("name") or item.get("text")
        extra = {k: item[k] for k in ("role", "method", "dataset", "metric", "conditions") if item.get(k)}
        lines += [f"## {number}. {field_name}", "", f"- [ ] **{label}**" + (f" `{json.dumps(extra)}`" if extra else ""),
                  f"- Paper: {title} (`{run_id}`)"]
        for evidence in item["evidence"]:
            row = db.query("SELECT text FROM sentences WHERE sentence_id = ?", (evidence["sentence"],))
            text = row[0]["text"] if row else "(sentence not found)"
            lines += [f"- `{evidence['sentence']}` ({evidence.get('check', '')}): {text}", f"  - quote: \"{evidence['quote']}\""]
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
