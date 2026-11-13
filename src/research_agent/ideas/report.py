"""The report's Phase 4 part, built from what is stored: each ranked idea with its
assessment, its gap's evidence, its closest papers and its earlier versions."""

import json
from typing import Any

from research_agent.ideas.ideas import IdeaStore, load_gaps
from research_agent.report import ClosestWork, IdeaReport, Quote
from research_agent.storage import Database


def idea_reports(db: Database, run_id: str) -> list[IdeaReport]:
    store = IdeaStore(db)
    ideas = store.for_run(run_id)
    gaps = {gap["node_id"]: gap for gap in load_gaps(db, run_id)}
    ranked = sorted((i for i in ideas if i.final_rank is not None), key=lambda i: i.final_rank or 0)
    reports = []
    for idea in ranked:
        assessment = idea.assessment or {}
        gap = gaps.get(idea.gap_node_id or "")
        closest = _closest(db, idea.idea_id)
        titles = {w["paper_id"]: w["label"] for w in closest}
        history = [
            f"round {v.round}, level {v.level or '?'}: {v.title}"
            + (f" (then changed its {child.detail['changed']}: {child.detail.get('why', '')})" if child else "")
            for v in ideas if v.lineage == idea.lineage and v.round < idea.round and v.level is not None
            for child in [next((c for c in ideas if c.parent_id == v.idea_id), None)]
        ]
        reports.append(IdeaReport(
            rank=idea.final_rank, idea_id=idea.idea_id, origin=idea.origin, title=idea.title, purpose=idea.purpose,
            mechanism=idea.mechanism, evaluation=idea.evaluation, kind=idea.kind,
            builds_on=idea.detail.get("builds_on", []), feasibility=idea.detail.get("feasibility"),
            level=idea.level, label=assessment.get("label"), rating=assessment.get("rating"),
            inconclusive=bool(assessment.get("inconclusive")), reasons=assessment.get("reasons", []),
            coverage=assessment.get("coverage") or {},
            gap=gap["gap"] if gap else idea.detail.get("gap"), gap_kind=gap["kind"] if gap else None,
            gap_evidence=[Quote(sentence_id=e["sentence_id"], quote=e["quote"], paper=e["paper"])
                          for e in (gap["evidence"] if gap else [])],
            addressed_by=[titles.get(p, p) for p in assessment.get("addressed_by", [])],
            closest=[w["work"] for w in closest],
            history=history,
        ))
    return reports


def _closest(db: Database, idea_id: str) -> list[dict[str, Any]]:
    rows = db.query(
        "SELECT i.paper_id, i.position, p.title, p.year, c.verdict FROM idea_candidates i JOIN papers p USING (paper_id)"
        " LEFT JOIN comparisons c ON c.idea_id = i.idea_id AND c.paper_id = i.paper_id"
        " WHERE i.idea_id = ? AND i.position IS NOT NULL ORDER BY i.position",
        (idea_id,),
    )
    works = []
    for row in rows:
        verdict = json.loads(row["verdict"]) if row["verdict"] else {}
        label = f"{row['title']} ({row['year']})" if row["year"] else row["title"]
        if verdict.get("status") != "ok":
            work = ClosestWork(position=row["position"], title=row["title"], year=row["year"], read="not compared")
        else:
            quotes = [Quote(sentence_id=e["sentence_id"], quote=e["quote"])
                      for facet in ("mechanism", "purpose", "evaluation") for e in verdict[facet]["evidence"]]
            work = ClosestWork(
                position=row["position"], title=row["title"], year=row["year"],
                read="full text" if verdict["depth"] == "full" else "abstract",
                purpose=verdict["purpose"]["effective"], mechanism=verdict["mechanism"]["effective"],
                evaluation=verdict["evaluation"]["effective"], paper_mechanism=verdict["mechanism"]["paper"] or None,
                solves_gap=verdict["solves_gap"], difference=verdict["difference"], evidence=quotes,
            )
        works.append({"paper_id": row["paper_id"], "label": label, "work": work})
    return works
