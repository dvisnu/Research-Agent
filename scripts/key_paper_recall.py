"""Phase 1 check: run the pipeline on dev topics and report how many key papers it finds.

    uv run python scripts/key_paper_recall.py                  # every dev topic
    uv run python scripts/key_paper_recall.py gnn-oversmoothing moe-load-balancing
    uv run python scripts/key_paper_recall.py --full-text      # also download and parse full texts

A key paper counts as read in full or kept by its place among the papers the run
kept, after the relevance check dropped the off-topic ones. Without --full-text
nothing is read in full, which skips the slow downloads; the ranking and the
relevance check are the same, so "first N kept" still counts the papers that would
be read in full.
A topic whose run fails is reported and skipped; it can be resumed with `agent resume`.
Results are printed and saved to var/checks/.
"""

import argparse
import json
import tomllib
from datetime import datetime
from pathlib import Path

from research_agent.config import Settings
from research_agent.literature.papers import title_key
from research_agent.pipeline import RunError, default_services, start_run
from research_agent.storage import Database

TOPICS = Path(__file__).parents[1] / "data" / "dev_topics.toml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("topics", nargs="*", help="dev topic IDs (default: all)")
    parser.add_argument("--full-text", action="store_true", help="read the top papers in full as usual")
    args = parser.parse_args()

    topics = tomllib.loads(TOPICS.read_text(encoding="utf-8"))["topic"]
    if args.topics:
        topics = [t for t in topics if t["id"] in args.topics]
    base = Settings()
    full, kept = base.full_read, base.full_read + base.abstract_only
    settings = base if args.full_text else base.model_copy(update={"full_read": 0, "abstract_only": kept})

    services = default_services(settings)
    results, failed = [], []
    try:
        for topic in topics:
            try:
                run = start_run(settings, topic["text"], services=services)
            except RunError as exc:
                failed.append({"topic": topic["id"], "run_id": exc.run_id, "error": str(exc)})
                print(f"{topic['id']:<36} FAILED ({exc.run_id}): {exc}", flush=True)
                continue
            results.append(_score(settings, topic, run.run_id, full, kept))
            _print_row(results[-1], full, kept)
    finally:
        services.close()

    total = sum(r["key_papers"] for r in results)
    for label, key in (("found", "found"), (f"first {full} kept", "in_full"), (f"kept ({kept})", "in_kept"),
                       ("off-topic", "judged_off_topic")):
        hits = sum(r[key] for r in results)
        print(f"{label:>14}: {hits}/{total} key papers ({hits / total:.0%})" if total else "")

    out = settings.data_dir / "checks" / f"key_papers-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"full": full, "kept": kept, "results": results, "failed": failed}, indent=2), encoding="utf-8")
    print(f"saved {out}")


def _score(settings: Settings, topic: dict, run_id: str, full: int, kept: int) -> dict:
    with Database(settings.db_path) as db:
        rows = db.query(
            "SELECT p.arxiv_id, p.title_key, r.rank, r.read_level, r.relevance FROM run_papers r"
            " JOIN papers p USING (paper_id) WHERE r.run_id = ? ORDER BY r.rank",
            (run_id,),
        )
    info, position = [], 0  # position: place among the kept papers, best-ranked first
    for row in rows:
        position += row["read_level"] != "excluded"
        kept_at = position if row["read_level"] != "excluded" else None
        info.append({"rank": row["rank"], "relevance": row["relevance"], "kept_at": kept_at})
    by_arxiv = {row["arxiv_id"]: info[i] for i, row in enumerate(rows) if row["arxiv_id"]}
    by_title = {row["title_key"]: info[i] for i, row in enumerate(rows)}
    papers = []
    for paper in topic["key_papers"]:
        match = by_arxiv.get(paper["arxiv"]) or by_title.get(title_key(paper["title"]))
        papers.append({**paper, **(match or {"rank": None, "relevance": None, "kept_at": None})})
    found = [p for p in papers if p["rank"] is not None]
    return {
        "topic": topic["id"],
        "run_id": run_id,
        "candidates": len(rows),
        "key_papers": len(papers),
        "found": len(found),
        "in_full": sum(1 for p in found if p["kept_at"] and p["kept_at"] <= full),
        "in_kept": sum(1 for p in found if p["kept_at"]),
        "judged_off_topic": sum(1 for p in found if p["relevance"] == "off_topic"),
        "papers": papers,
    }


def _place(paper: dict) -> str:
    """'-' if not found; else its rank, then its place among the kept papers or why it wasn't kept."""
    if paper["rank"] is None:
        return "-"
    if paper["kept_at"]:
        return f"{paper['rank']}>{paper['kept_at']}"
    return f"{paper['rank']}>off-topic" if paper["relevance"] == "off_topic" else f"{paper['rank']}>cut"


def _print_row(result: dict, full: int, kept: int) -> None:
    places = ", ".join(_place(p) for p in result["papers"])
    print(
        f"{result['topic']:<36} candidates {result['candidates']:>4}  found {result['found']}/{result['key_papers']}"
        f"  full {result['in_full']}  kept {result['in_kept']}  rank>kept [{places}]",
        flush=True,
    )


if __name__ == "__main__":
    main()
