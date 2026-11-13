"""Planted-duplicate test: does the novelty check recognise ideas that published papers already present?

    uv run python scripts/duplicate_test.py plant --runs run_ab12 run_cd34   # pick papers, write their ideas
    uv run python scripts/duplicate_test.py check --filter specter2          # check every planted idea
    uv run python scripts/duplicate_test.py brief                            # the same papers as brief ideas
    uv run python scripts/duplicate_test.py check --filter specter2 --set brief
    uv run python scripts/duplicate_test.py filters var/checks/duplicates-specter2-....json   # both filters, same pools

plant: for each dev run, searches Semantic Scholar for papers on its question published
from PLANT_YEAR on, keeps those outside the run's corpus that have an arXiv ID and an
abstract, takes the closest to the question (SPECTER2), and has the extraction model
restate each as an idea in its own words, without the title's distinctive words or the
method's name. Surveys and position papers are skipped. PER_RUN ideas a run are added
to data/planted_ideas.json; the file fixes the test set, so it is made once.

check: checks each planted idea in a run of its own (as `agent novelty` does, with the
idea's facets) using one embedding filter, and records whether the source paper was
found, kept by the filter, among the closest 10 and judged a match, and the level. An
idea is caught at level 1 or 2. Only "not novel" ideas are tested: a "novel" label
depends on what a search happened to find (Shahid et al. 2025).

filters: for the ideas of a check's results file, scores every paper in each idea's
candidate pool with both filters and reports the source paper's rank under each.
Gemini needs an embedding request for every paper text not embedded before.
"""

import argparse
import json
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from research_agent.config import Settings
from research_agent.embeddings import EmbeddingCache
from research_agent.ideas.ideas import IdeaStore
from research_agent.ideas.search import FILTER_KEEP, Candidate, paper_text
from research_agent.ideas.stages import latest_detail
from research_agent.literature.papers import PaperStore, from_s2
from research_agent.llm import LLMOutputError, Prompt, Role
from research_agent.pipeline import RunError, check_idea_run, default_services
from research_agent.pipeline.runner import open_pipeline
from research_agent.storage import Database
from research_agent.trace import TraceRecorder

from novelty_check import google_quota_used_up  # scripts/ is on the path when run as a script

PLANTED = Path(__file__).parents[1] / "data" / "planted_ideas.json"
# The same papers restated as short, high-level proposals: closer to how generated ideas read,
# and harder to match, since the paper's specific design choices are left out.
PLANTED_BRIEF = Path(__file__).parents[1] / "data" / "planted_ideas_brief.json"
SETS = {"exact": PLANTED, "brief": PLANTED_BRIEF}
PLANT_YEAR = "2025-"
PER_RUN = 3
TRIES = 8  # candidates restated per run before giving up
MIN_ABSTRACT = 400


class PlantedIdea(BaseModel):
    """The extraction model's reply."""

    kind: Literal["method", "analysis", "resource", "survey", "position"] = Field(
        description="method: proposes a method, model, system or technique; analysis: an empirical study; "
                    "resource: a dataset or benchmark; survey; position: an opinion or perspective paper")
    title: str = Field(description="At most 15 words, in your own words")
    purpose: str = Field(description="One sentence: the problem it addresses")
    mechanism: str = Field(description="One or two sentences: how it works")
    evaluation: str = Field(description="One sentence: how it is tested, as the abstract says")


_PLANT_SYSTEM = """You restate a published computer science or machine learning paper as the research idea it presents, written as a proposal someone might have made before the paper existed. It will test whether a novelty checker can find the paper.

Write the idea's title, purpose (the problem it addresses), mechanism (how it works) and evaluation (how it is tested), each in your own words. Do not reuse the paper title's distinctive words, and never use the name or acronym the paper gives its method, model, dataset or benchmark; describe what it does instead. Keep every technical point the abstract makes about the mechanism. Also classify the paper's kind."""

_BRIEF_SYSTEM = """You restate a published computer science or machine learning paper as the short research idea a researcher might have jotted down before doing the work. It will test whether a novelty checker can find the paper from a rough idea.

Write the idea's title (at most 12 words), purpose, mechanism and evaluation, each one short sentence in your own words. Name the kind of technique and the problem, one level more general than the paper: leave out its specific design choices, numbers, model sizes and secondary components, and name a dataset or metric only if the idea is about it. Do not reuse the paper title's distinctive words, and never use the name or acronym the paper gives its method, model, dataset or benchmark. Also classify the paper's kind."""


def brief(settings: Settings) -> None:
    """Restate the planted papers as brief ideas (the same papers and IDs as the exact set)."""
    planted = json.loads(PLANTED.read_text(encoding="utf-8"))
    out = []
    with open_pipeline(settings) as (_, ctx):
        for item in planted:
            row = ctx.db.query("SELECT title, abstract FROM papers WHERE paper_id = ?", (item["source"]["paper_id"],))[0]
            paper = f"Title: {row['title']}\nAbstract: {row['abstract']}"
            idea = ctx.llm.structured(Role.EXTRACTION, Prompt("planted_idea_brief", 1, _BRIEF_SYSTEM, paper), PlantedIdea)
            leaks = _leaks(row["title"], idea)
            if leaks:
                note = f"\n\nDo not use these words from the paper's title or names: {', '.join(leaks)}."
                idea = ctx.llm.structured(Role.EXTRACTION, Prompt("planted_idea_brief", 1, _BRIEF_SYSTEM, paper + note), PlantedIdea)
            out.append({**item, "idea": idea.model_dump(exclude={"kind"}), "leaks": _leaks(row["title"], idea)})
            print(f"{item['id']}: {idea.title} | {idea.mechanism}", flush=True)
    PLANTED_BRIEF.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


_COMMON = set("""a an and the of for in on with to via by from as at into using based towards toward
through over under is are be learning model models large language method methods approach framework
efficient effective new deep neural network networks data""".split())


def _content(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _COMMON and len(w) > 2}


def _leaks(source_title: str, idea: PlantedIdea) -> list[str]:
    """What in the restated idea gives its paper away: the name before the title's colon (or
    any coined word of the title, such as an acronym) anywhere in the idea, or a title sharing
    more than half its content words with the paper's."""
    head = source_title.split(":")[0] if ":" in source_title else ""
    names = {w for w in re.findall(r"[A-Za-z][A-Za-z0-9-]*", source_title)
             if (w == head.strip() and len(head.split()) == 1) or (sum(c.isupper() for c in w) >= 2 and len(w) > 2)}
    text = " ".join((idea.title, idea.purpose, idea.mechanism, idea.evaluation)).lower()
    leaks = sorted(name for name in names if re.search(rf"\b{re.escape(name.lower())}\b", text))
    shared = _content(idea.title) & _content(source_title)
    if len(shared) > len(_content(idea.title)) / 2:
        leaks += sorted(shared)
    return leaks


def plant(settings: Settings, run_ids: list[str]) -> None:
    planted = json.loads(PLANTED.read_text(encoding="utf-8")) if PLANTED.exists() else []
    done = {item["source"]["paper_id"] for item in planted}
    with open_pipeline(settings) as (_, ctx):
        s2 = ctx.sources.s2
        if s2 is None:
            sys.exit("plant needs RA_S2_API_KEY")
        for run_id in run_ids:
            if any(item["run_id"] == run_id for item in planted):
                print(f"{run_id}: already planted", flush=True)
                continue
            question = ctx.db.get_run(run_id)["question"]
            angles = (latest_detail(ctx.db, run_id, "brief").get("brief") or {}).get("angles", [])
            corpus = {row["paper_id"] for row in ctx.db.query("SELECT paper_id FROM run_papers WHERE run_id = ?", (run_id,))}
            store, found = PaperStore(ctx.db), {}
            for angle in angles[:4]:  # short queries: long ones with a year filter often find nothing
                for paper in s2.search(" ".join(angle["keywords"][:3]), limit=30, year=PLANT_YEAR):
                    record = from_s2(paper)
                    if record and record.arxiv_id and len(record.abstract or "") >= MIN_ABSTRACT:
                        paper_id = store.upsert(record)
                        if paper_id not in corpus and paper_id not in done:
                            found[paper_id] = record
            ids = list(found)
            if not ids:
                print(f"{run_id}: no candidate papers", flush=True)
                continue
            vectors = ctx.embedder.papers([(found[i].title, found[i].abstract) for i in ids])
            query = ctx.embedder.query(question)
            order = sorted(range(len(ids)), key=lambda n: -float(vectors[n] @ query))
            kept = 0
            for n in order[:TRIES]:
                record = found[ids[n]]
                paper = f"Title: {record.title}\nAbstract: {record.abstract}"
                try:
                    idea = ctx.llm.structured(Role.EXTRACTION, Prompt("planted_idea", 2, _PLANT_SYSTEM, paper), PlantedIdea)
                    leaks = _leaks(record.title, idea)
                    if leaks:  # one more try, saying what gave the paper away
                        note = f"\n\nDo not use these words from the paper's title or names: {', '.join(leaks)}."
                        idea = ctx.llm.structured(Role.EXTRACTION, Prompt("planted_idea", 2, _PLANT_SYSTEM, paper + note), PlantedIdea)
                        leaks = _leaks(record.title, idea)
                except LLMOutputError as exc:
                    print(f"  skipped {record.title[:60]}: {exc}", flush=True)
                    continue
                if leaks:
                    print(f"  skipped {record.title[:60]}: still uses {', '.join(leaks)}", flush=True)
                    continue
                if idea.kind in ("survey", "position"):
                    print(f"  skipped {record.title[:60]}: {idea.kind}", flush=True)
                    continue
                kept += 1
                planted.append({
                    "id": f"{run_id}:{kept}", "run_id": run_id, "question": question, "kind": idea.kind,
                    "source": {"paper_id": ids[n], "title": record.title, "year": record.year,
                               "arxiv_id": record.arxiv_id, "s2_id": record.s2_id},
                    "idea": idea.model_dump(exclude={"kind"}),
                })
                print(f"{run_id}: planted {record.title[:70]} ({record.year}) as: {idea.title}", flush=True)
                if kept == PER_RUN:
                    break
            PLANTED.write_text(json.dumps(planted, indent=2, ensure_ascii=False), encoding="utf-8")


def check(settings: Settings, method: str, only: list[str], which: str = "exact") -> Path:
    planted = json.loads(SETS[which].read_text(encoding="utf-8"))
    items = [p for p in planted if not only or p["id"] in only]
    results = []
    services = default_services(settings)
    try:
        for item in items:
            text = ". ".join(item["idea"][k] for k in ("title", "purpose", "mechanism", "evaluation"))
            try:
                run = check_idea_run(settings, text, services, origin="planted", facets=item["idea"],
                                     detail={"planted": item["id"], "source": item["source"]["paper_id"]}, method=method)
            except RunError as exc:
                print(f"{item['id']}: FAILED ({exc.run_id}): {exc}", flush=True)
                results.append({"id": item["id"], "run_id": exc.run_id, "error": str(exc)})
                continue
            with Database(settings.db_path) as db:
                results.append({"id": item["id"], "run_id": run.run_id, **_outcome(db, run.run_id, item)})
                exhausted = google_quota_used_up(TraceRecorder(db).events(run.run_id))
            r = results[-1]
            print(f"{item['id']}: level {r['level']}, source {r['source']}, deciding {r['deciding_is_source']}", flush=True)
            if exhausted:
                print(f"STOP: a Google quota ran out ({exhausted[0]}). Switch RA_GOOGLE_API_KEY and check the "
                      "remaining ideas with --only.", flush=True)
                break
    finally:
        services.close()
    summary = _summary(results)
    print(json.dumps(summary, indent=2))
    out = settings.data_dir / "checks" / f"duplicates-{which}-{method}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"set": which, "filter": method, "summary": summary, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out}")
    return out


def _outcome(db: Database, run_id: str, item: dict[str, Any]) -> dict[str, Any]:
    idea = IdeaStore(db).for_run(run_id)[0]
    source = item["source"]["paper_id"]
    pool = db.query("SELECT paper_id, similarity, position FROM idea_candidates WHERE idea_id = ?", (idea.idea_id,))
    ranked = sorted(pool, key=lambda r: -(r["similarity"] or 0))
    rank = next((n for n, r in enumerate(ranked, start=1) if r["paper_id"] == source), None)
    position = next((r["position"] for r in pool if r["paper_id"] == source), None)
    verdict = db.query("SELECT verdict FROM comparisons WHERE idea_id = ? AND paper_id = ?", (idea.idea_id, source))
    match = None
    if verdict and (v := json.loads(verdict[0]["verdict"])).get("status") == "ok":
        match = {f: v[f]["effective"] for f in ("purpose", "mechanism", "evaluation")} | {"depth": v["depth"]}
    if rank is None:
        where = "not found"
    elif position is None:
        where = f"found, similarity rank {rank}, not among the closest"
    else:
        where = f"closest #{position} (similarity rank {rank})"
    a = idea.assessment or {}
    return {"idea_id": idea.idea_id, "level": idea.level, "inconclusive": a.get("inconclusive"),
            "caught": (idea.level or 9) <= 2, "source": where, "source_rank": rank, "source_position": position,
            "pool": len(pool), "source_match": match, "deciding_is_source": source in a.get("deciding", []),
            "reasons": a.get("reasons")}


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    done = [r for r in results if "error" not in r]
    n = len(done) or 1

    def share(test: Any) -> str:
        hits = sum(1 for r in done if test(r))
        return f"{hits}/{len(done)} ({hits / n:.0%})"

    return {
        "checked": len(done), "failed": len(results) - len(done),
        "source_found": share(lambda r: r["source_rank"] is not None),
        f"source_kept_by_filter_top{FILTER_KEEP}": share(lambda r: r["source_rank"] is not None and r["source_rank"] <= FILTER_KEEP),
        "source_among_closest": share(lambda r: r["source_position"] is not None),
        "caught_level_1_or_2": share(lambda r: r["caught"]),
        "caught_by_the_source_itself": share(lambda r: r["caught"] and r["deciding_is_source"]),
    }


@contextmanager
def _ctx(settings: Settings) -> Iterator[Any]:
    with open_pipeline(settings) as (_, ctx):
        yield ctx


HYBRID_TOP = 60  # the hybrid filter: Gemini re-sorts SPECTER2's top this many


def filters(settings: Settings, results_file: Path, only: list[str]) -> None:
    """Rank each idea's source paper in its pool under SPECTER2, Gemini, and Gemini re-sorting
    SPECTER2's top HYBRID_TOP (about a third of the embedding requests a full Gemini filter needs)."""
    data = json.loads(results_file.read_text(encoding="utf-8"))
    planted = {p["id"]: p for p in json.loads(SETS[data.get("set", "exact")].read_text(encoding="utf-8"))}
    rows = []
    with _ctx(settings) as ctx:
        for result in data["results"]:
            if "error" in result or (only and result["id"] not in only):
                continue
            idea = IdeaStore(ctx.db).get(result["idea_id"])
            source = planted[result["id"]]["source"]["paper_id"]
            pool = [Candidate(r["paper_id"], r["title"], r["abstract"], r["year"]) for r in ctx.db.query(
                "SELECT c.paper_id, p.title, p.abstract, p.year FROM idea_candidates c JOIN papers p USING (paper_id)"
                " WHERE c.idea_id = ?", (idea.idea_id,))]
            if source not in {c.paper_id for c in pool}:
                rows.append({"id": result["id"], "pool": len(pool), "specter2": None, "gemini": None, "hybrid": None})
                continue
            vectors = ctx.embedder.papers([(c.title, c.abstract) for c in pool])
            query = ctx.embedder.papers([(idea.title, " ".join((idea.purpose, idea.mechanism, idea.evaluation)))])[0]
            specter = {c.paper_id: float(v @ query) for c, v in zip(pool, vectors)}
            texts = [idea.text(), *(paper_text(c) for c in pool)]
            found, report = EmbeddingCache(ctx.db, ctx.text_embedder).vectors(texts)
            if report.get("missing"):
                print(f"STOP: Gemini embeddings missing ({report.get('unavailable')}). Switch the Google key and rerun.", flush=True)
                break
            gemini = {c.paper_id: float(found[t] @ found[texts[0]]) for c, t in zip(pool, texts[1:])}
            top = sorted(specter, key=lambda p: -specter[p])[:HYBRID_TOP]
            hybrid = _rank({p: gemini[p] for p in top}, source) if source in top else _rank(specter, source)
            rows.append({"id": result["id"], "pool": len(pool), "specter2": _rank(specter, source),
                         "gemini": _rank(gemini, source), "hybrid": hybrid, "embedding": report})
            print(f"{result['id']}: pool {len(pool)}, source rank specter2 {rows[-1]['specter2']}, "
                  f"gemini {rows[-1]['gemini']}, hybrid {hybrid}", flush=True)
    found_rows = [r for r in rows if r["specter2"] is not None]
    summary = {"ideas": len(rows), "source_in_pool": len(found_rows)}
    for method in ("specter2", "gemini", "hybrid"):
        ranks = [r[method] for r in found_rows]
        summary[method] = {f"top{k}": sum(1 for x in ranks if x <= k) for k in (10, FILTER_KEEP, 60)} | {
            "median_rank": sorted(ranks)[len(ranks) // 2] if ranks else None}
    print(json.dumps(summary, indent=2))
    out = results_file.with_name(results_file.stem + "-filters.json")
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(f"Saved: {out}")


def _rank(scores: dict[str, float], paper_id: str) -> int:
    ordered = sorted(scores, key=lambda p: -scores[p])
    return ordered.index(paper_id) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plant")
    p.add_argument("--runs", nargs="+", required=True)
    c = sub.add_parser("check")
    c.add_argument("--filter", choices=("specter2", "gemini"), required=True)
    c.add_argument("--only", nargs="*", default=[], help="planted idea IDs")
    c.add_argument("--set", choices=tuple(SETS), default="exact", help="exact restatements, or brief ideas")
    sub.add_parser("brief", help="restate the planted papers as brief ideas")
    f = sub.add_parser("filters")
    f.add_argument("results", type=Path)
    f.add_argument("--only", nargs="*", default=[], help="planted idea IDs")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    settings = Settings()
    if args.command == "plant":
        plant(settings, args.runs)
    elif args.command == "check":
        check(settings, args.filter, args.only, args.set)
    elif args.command == "brief":
        brief(settings)
    else:
        filters(settings, args.results, args.only)


if __name__ == "__main__":
    main()
