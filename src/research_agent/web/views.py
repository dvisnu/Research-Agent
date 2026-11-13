"""What the web UI shows, read from the store: runs, live progress, provenance chains,
sentences in context, the knowledge graph and the corpus."""

import json
import re
from datetime import UTC, datetime
from typing import Any

from research_agent.literature.corpus import Corpus
from research_agent.pipeline import STAGES
from research_agent.plans.facts import Menu, load_plan_papers
from research_agent.plans.store import PlanStore
from research_agent.storage import Database

# A run marked running with no trace event for this long has probably been interrupted
# (the longest quiet stretch seen is SPECTER2 embedding a few hundred papers, minutes).
STALE_MINUTES = 20
# The .env key that a provider's quota belongs to.
KEYS = {
    "google": "RA_GOOGLE_API_KEY",
    "openrouter": "RA_OPENROUTER_API_KEY",
    "groq": "RA_GROQ_API_KEY",
    "semantic_scholar": "RA_S2_API_KEY",
    "openalex": "RA_OPENALEX_API_KEY",
}
_QUOTA = re.compile(r"\b429\b|quota|rate.?limit|RESOURCE_EXHAUSTED|per day|\bTPD\b|\bRPD\b", re.I)
RECENT_CALLS = 12


def diagnose(error: str | None) -> dict[str, Any] | None:
    """What stopped a run: whether a quota ran out, and whose key to switch."""
    if not error:
        return None
    providers = [p for p in ("google", "openrouter", "groq") if re.search(rf"\b{p}:", error)]
    if re.search(r"semantic.?scholar", error, re.I):
        providers.append("semantic_scholar")
    if re.search(r"openalex", error, re.I):
        providers.append("openalex")
    quota = bool(_QUOTA.search(error))
    role = re.search(r"No model for the (\w+) role", error)
    return {
        "quota": quota,
        "role": role.group(1) if role else None,
        "providers": providers,
        "keys": [KEYS[p] for p in providers] if quota else [],
    }


def _age_minutes(timestamp: str | None) -> float | None:
    if not timestamp:
        return None
    return (datetime.now(UTC) - datetime.fromisoformat(timestamp)).total_seconds() / 60


def run_rows(db: Database, run_ids: list[str] | None = None, limit: int = 100) -> list[dict[str, Any]]:
    """Runs, newest first, with their last activity, best idea and plan coverage."""
    if run_ids is None:
        rows = db.list_runs(limit)
    else:
        rows = [row for run_id in run_ids if (row := db.get_run(run_id)) is not None]
    ids = [row["run_id"] for row in rows]
    if not ids:
        return []
    marks = ",".join("?" * len(ids))
    last = dict(db.query(
        f"SELECT run_id, MAX(ended_at) FROM trace_events WHERE run_id IN ({marks}) GROUP BY run_id", tuple(ids)))
    best = {row["run_id"]: row for row in db.query(
        f"SELECT run_id, title, level FROM ideas WHERE final_rank = 1 AND run_id IN ({marks})", tuple(ids))}
    coverage = {}
    for row in db.query(
        f"SELECT run_id, detail FROM plans WHERE run_id IN ({marks}) ORDER BY round", tuple(ids)
    ):
        found = json.loads(row["detail"] or "{}").get("coverage")
        if found:
            coverage[row["run_id"]] = found
    out = []
    for row in rows:
        idea = best.get(row["run_id"])
        out.append({
            "run_id": row["run_id"],
            "question": row["question"],
            "status": row["status"],
            "error": row["error"],
            "diagnosis": diagnose(row["error"]) if row["status"] == "failed" else None,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "last_activity": last.get(row["run_id"]),
            "best_idea": {"title": idea["title"], "level": idea["level"]} if idea else None,
            "coverage": coverage.get(row["run_id"]),
        })
    return out


def state(run: dict[str, Any], job_alive: bool | None) -> str:
    """running, completed, failed, interrupted (a run started here whose process ended
    without saying so), or stale (running elsewhere, quiet for STALE_MINUTES)."""
    if run["status"] != "running":
        return run["status"]
    if job_alive is not None:
        return "running" if job_alive else "interrupted"
    quiet = _age_minutes(run["last_activity"] or run["updated_at"])
    return "stale" if quiet is not None and quiet > STALE_MINUTES else "running"


def progress(db: Database, run_id: str) -> dict[str, Any]:
    """Each stage's status, attempts, time and calls, the stage now running, and the latest calls."""
    events = [dict(row) for row in db.query(
        "SELECT event_id, parent_event_id, step, attempt, activity_type, agent, status, error, started_at, ended_at,"
        " json_extract(detail, '$.cached') AS cached FROM trace_events WHERE run_id = ? ORDER BY seq",
        (run_id,),
    )]
    steps = [e for e in events if e["parent_event_id"] is None]
    calls = [e for e in events if e["parent_event_id"] is not None]
    # A step's row is written when it ends, so calls whose step has no row yet (followed up
    # through calls made inside calls) belong to the step running now, or to one a killed
    # process never finished; only those after the last finished step count.
    by_id = {e["event_id"]: e for e in events}

    def unfinished(event: dict[str, Any]) -> bool:
        while event["parent_event_id"] in by_id:
            event = by_id[event["parent_event_id"]]
        return event["parent_event_id"] is not None

    since = steps[-1]["ended_at"] if steps else None
    open_calls = [e for e in calls if unfinished(e) and (since is None or e["started_at"] >= since)]
    current = open_calls[-1]["step"] if open_calls else None

    stages = []
    for stage in STAGES:
        mine = [e for e in steps if e["step"] == stage.name]
        done = [e for e in mine if e["status"] == "ok"]
        failed = [e for e in mine if e["status"] != "ok"]
        mine_calls = [e for e in calls if e["step"] == stage.name]
        if stage.name == current:
            status = "running"
        elif failed and (not done or failed[-1]["ended_at"] > done[-1]["ended_at"]):
            status = "failed"
        elif done:
            status = "done"
        else:
            status = "pending"
        stages.append({
            "name": stage.name,
            "phase": stage.phase,
            "summary": stage.summary,
            "status": status,
            "runs": len(done),
            "failed_attempts": len(failed),
            "seconds": sum(_seconds(e) for e in mine),
            "llm_calls": sum(1 for e in mine_calls if e["activity_type"] == "llm_call"),
            "api_calls": sum(1 for e in mine_calls if e["activity_type"] == "api_call"),
            "error": failed[-1]["error"] if status == "failed" else None,
        })
    return {
        "current": {
            "stage": current,
            "since": since,
            "calls": len(open_calls),
        } if current else None,
        "last_finished": steps[-1]["step"] if steps else None,
        "stages": stages,
        "recent": [_call(e) for e in calls[-RECENT_CALLS:]][::-1],
    }


def _seconds(event: dict[str, Any]) -> float:
    return (datetime.fromisoformat(event["ended_at"]) - datetime.fromisoformat(event["started_at"])).total_seconds()


def _call(event: dict[str, Any]) -> dict[str, Any]:
    agent = json.loads(event["agent"] or "{}")
    if event["activity_type"] == "llm_call":
        who = f"{agent.get('provider')}:{agent.get('model')}" + (f" ({agent['role']})" if agent.get("role") else "")
    else:
        who = agent.get("name") or agent.get("service") or ""
    return {
        "kind": event["activity_type"],
        "step": event["step"],
        "who": who,
        "status": event["status"],
        "error": event["error"],
        "cached": bool(event["cached"]),
        "seconds": round(_seconds(event), 1),
        "ended_at": event["ended_at"],
    }


def paper_links(row: Any) -> dict[str, str]:
    links = {}
    if row["arxiv_id"]:
        links["arXiv"] = f"https://arxiv.org/abs/{row['arxiv_id']}"
    if row["doi"]:
        links["DOI"] = f"https://doi.org/{row['doi']}"
    if row["s2_id"]:
        links["Semantic Scholar"] = f"https://www.semanticscholar.org/paper/{row['s2_id']}"
    return links


def papers(db: Database, run_id: str) -> list[dict[str, Any]]:
    """The run's corpus, best-ranked first: read level, relevance label and why."""
    out = []
    for row in Corpus(db, run_id).papers():
        found = json.loads(row["found_by"] or "[]")
        out.append({
            "paper_id": row["paper_id"],
            "title": row["title"],
            "year": row["year"],
            "venue": row["venue"],
            "authors": json.loads(row["authors"] or "[]")[:6],
            "citations": row["citation_count"],
            "rank": row["rank"],
            "read_level": row["read_level"],
            "relevance": row["relevance"],
            "reason": row["reason"],
            "text_source": row["text_source"],
            "channels": sorted({f["channel"] for f in found}),
            "links": paper_links(row),
        })
    return out


def sentence(db: Database, sentence_id: str, context: int = 2) -> dict[str, Any] | None:
    """A stored sentence with its neighbours, its section and its paper."""
    rows = db.query(
        "SELECT t.sentence_id, t.idx, t.text, t.section_id, s.heading, s.retired_by, p.*"
        " FROM sentences t JOIN sections s ON s.section_id = t.section_id JOIN papers p ON p.paper_id = s.paper_id"
        " WHERE t.sentence_id = ?",
        (sentence_id,),
    )
    if not rows:
        return None
    row = rows[0]
    around = db.query(
        "SELECT sentence_id, text FROM sentences WHERE section_id = ? AND idx BETWEEN ? AND ? ORDER BY idx",
        (row["section_id"], row["idx"] - context, row["idx"] + context),
    )
    return {
        "sentence_id": row["sentence_id"],
        "text": row["text"],
        "section": row["heading"],
        "retired": row["retired_by"] is not None,
        "context": [{"sentence_id": r["sentence_id"], "text": r["text"]} for r in around],
        "paper": {"paper_id": row["paper_id"], "title": row["title"], "year": row["year"], "links": paper_links(row)},
    }


def element_chain(db: Database, run_id: str, element_id: str) -> dict[str, Any] | None:
    """A plan element's chain, as `agent plan --element` prints it: the verdict, the decision,
    the derivation, and each cited fact with the sentences it rests on."""
    plan = PlanStore(db).latest(run_id)
    if plan is None:
        return None
    element = next((e for e in plan.elements if e.element_id == element_id), None)
    if element is None:
        return None
    menu = Menu.from_json(plan.menu, load_plan_papers(db, run_id))
    kept = {e["sentence_id"] for e in element.evidence}
    facts = []
    for cite in element.cites:
        fact = menu.get(cite)
        if fact is None:
            continue
        origin = fact.origin
        facts.append({
            "fact_id": fact.fact_id,
            "kind": fact.kind,
            "text": fact.text,
            "origin": {"from": origin.get("from"), "query": origin.get("query")},
            "papers": [{"paper": menu.label(paper_id), "why": info.get("why", [])}
                       for paper_id, info in (origin.get("papers") or {}).items()],
            "evidence": [{"sentence_id": e["sentence_id"], "quote": e["quote"], "kept": e["sentence_id"] in kept,
                          "sentence": sentence(db, e["sentence_id"], context=0)} for e in fact.evidence],
        })
    return {
        "element_id": element.element_id,
        "label": element.label,
        "text": element.text,
        "name": element.name,
        "status": element.status,
        "verdict": element.verdict,
        "verification": element.verification,
        "decision": element.decision,
        "derivation": element.derivation,
        "basis": element.basis,
        "facts": facts,
    }


def graph(db: Database, run_id: str) -> dict[str, Any]:
    """The run's knowledge graph: nodes with how many sentences back them, and edges."""
    backed = dict(db.query(
        "SELECT e.owner_id, COUNT(*) FROM evidence e JOIN nodes n ON n.node_id = e.owner_id"
        " WHERE e.owner_kind = 'node' AND n.run_id = ? GROUP BY e.owner_id", (run_id,)))
    nodes = [{"id": row["node_id"], "type": row["node_type"], "name": row["name"],
              "evidence": backed.get(row["node_id"], 0)}
             for row in db.query("SELECT node_id, node_type, name FROM nodes WHERE run_id = ?", (run_id,))]
    edges = [{"id": row["edge_id"], "type": row["edge_type"], "source": row["src_id"], "target": row["dst_id"]}
             for row in db.query("SELECT edge_id, edge_type, src_id, dst_id FROM edges WHERE run_id = ?", (run_id,))]
    return {"nodes": nodes, "edges": edges}


def graph_item(db: Database, kind: str, item_id: str) -> dict[str, Any] | None:
    """One node or edge: its attributes, its neighbours (for a node) and the sentences it rests on."""
    table, key = ("nodes", "node_id") if kind == "node" else ("edges", "edge_id")
    rows = db.query(f"SELECT * FROM {table} WHERE {key} = ?", (item_id,))
    if not rows:
        return None
    row = dict(rows[0])
    row["attrs"] = json.loads(row["attrs"] or "{}")
    evidence = [{"sentence_id": e["sentence_id"], "quote": e["quote"], "sentence": sentence(db, e["sentence_id"], 0)}
                for e in db.query("SELECT sentence_id, quote FROM evidence WHERE owner_kind = ? AND owner_id = ?",
                                  (kind, item_id))]
    out: dict[str, Any] = {"kind": kind, **row, "evidence": evidence}
    if kind == "node":
        out["aliases"] = [r[0] for r in db.query("SELECT alias FROM node_aliases WHERE node_id = ?", (item_id,))]
        out["links"] = [
            {"edge_id": r["edge_id"], "type": r["edge_type"], "direction": r["direction"], "node_id": r["other"],
             "node_type": r["node_type"], "name": r["name"]}
            for r in db.query(
                "SELECT e.edge_id, e.edge_type, 'out' AS direction, e.dst_id AS other, n.node_type, n.name"
                " FROM edges e JOIN nodes n ON n.node_id = e.dst_id WHERE e.src_id = ?1"
                " UNION ALL SELECT e.edge_id, e.edge_type, 'in', e.src_id, n.node_type, n.name"
                " FROM edges e JOIN nodes n ON n.node_id = e.src_id WHERE e.dst_id = ?1",
                (item_id,),
            )
        ]
    else:
        names = {r["node_id"]: dict(r) for r in db.query(
            "SELECT node_id, node_type, name FROM nodes WHERE node_id IN (?, ?)", (row["src_id"], row["dst_id"]))}
        out["source"], out["target"] = names.get(row["src_id"]), names.get(row["dst_id"])
    return out
