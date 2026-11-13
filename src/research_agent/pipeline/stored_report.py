"""A run's report rebuilt from what is stored, for runs whose graph or gaps were
rebuilt after the run finished (``scripts/graph_check.py --rebuild``,
``scripts/gap_check.py --rerun``), which update the database and trace but not
the report files, and for runs that check one idea (``agent novelty``).

Each part comes from its latest successful stage in the trace (the extraction
counts, the graph summary, the related work), from the graph (the landscape grid,
which is computed from it), the stored gap nodes, the stored ideas, or the stored plan.
"""

import json
from typing import Any

from research_agent.costs import run_cost
from research_agent.gaps.landscape import build_landscape
from research_agent.graph.build import load_graph
from research_agent.ideas.report import idea_reports
from research_agent.plans.report import plan_report
from research_agent.report import KnowledgeGraphSummary, RunReport
from research_agent.storage import Database, utc_now
from research_agent.trace import TraceRecorder


def stored_report(db: Database, run_id: str) -> RunReport:
    run = db.get_run(run_id)
    if run is None:
        raise KeyError(run_id)
    latest: dict[str, dict[str, Any]] = {}
    for event in TraceRecorder(db).events(run_id):
        if event["activity_type"] == "step" and event["status"] == "ok":
            latest[event["step"]] = event["detail"]

    report = RunReport(run_id=run_id, question=run["question"], generated_at=utc_now())
    if "build_graph" in latest:
        report.knowledge_graph = KnowledgeGraphSummary.from_stages(
            latest.get("extract", {}).get("extraction", {}), latest["build_graph"]["graph"]
        )
    graph = load_graph(db, run_id)
    landscape = build_landscape(graph) if "landscape" in latest else None
    report.add_phase3(latest.get("related_work", {}).get("related_work", []), landscape, _gaps(db, run_id))
    report.add_phase4(idea_reports(db, run_id))
    if any(e["step"] == "plan" for e in TraceRecorder(db).events(run_id) if e["activity_type"] == "step"):
        report.add_phase5(plan_report(db, run_id))
    report.cost = run_cost(db, run_id)
    return report


def _gaps(db: Database, run_id: str) -> list[dict[str, Any]]:
    gaps = []
    for row in db.query("SELECT node_id, name, attrs FROM nodes WHERE run_id = ? AND node_type = 'gap'", (run_id,)):
        attrs = json.loads(row["attrs"])
        evidence = db.query(
            "SELECT e.sentence_id, e.quote, p.title FROM evidence e JOIN sentences s USING (sentence_id)"
            " JOIN sections c ON c.section_id = s.section_id JOIN papers p ON p.paper_id = c.paper_id"
            " WHERE e.owner_kind = 'node' AND e.owner_id = ? ORDER BY e.sentence_id",
            (row["node_id"],),
        )
        gaps.append({
            "rank": attrs["rank"], "kind": attrs["kind"], "gap": row["name"], "warrant": attrs.get("warrant"),
            "relevance": attrs["relevance"], "papers": len(attrs["papers"]),
            "evidence": [{"sentence_id": e["sentence_id"], "quote": e["quote"], "paper": e["title"]} for e in evidence],
        })
    return sorted(gaps, key=lambda gap: gap["rank"])
