"""Command line: `agent run`, `agent resume`, `agent novelty`, `agent inspect`, `agent graph`, `agent ideas`,
`agent plan`, `agent report`, `agent cost`, `agent ui`."""

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.table import Table
from rich.tree import Tree

from research_agent.config import Settings
from research_agent.costs import run_cost
from research_agent.graph import queries
from research_agent.graph.build import load_graph
from research_agent.pipeline.stored_report import stored_report
from research_agent.report import write_report
from research_agent.pipeline import (
    STAGE_NAMES,
    CannotResume,
    RunError,
    RunOptions,
    RunResult,
    check_idea_run,
    resume_run,
    start_run,
)
from research_agent.ideas.ideas import IdeaStore
from research_agent.plans.facts import Menu, load_plan_papers
from research_agent.plans.prov import write_provenance
from research_agent.plans.store import Element, PlanStore
from research_agent.storage import Database
from research_agent.trace import TraceRecorder

app = typer.Typer(
    help="Evidence-grounded research ideas and provenance-aware experiment plans for CS/ML.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _settings() -> Settings:
    try:
        return Settings()
    except ValidationError as exc:
        # Without the input values: a misplaced API key must not end up on screen.
        problems = [
            f"  {'RA_' + str(error['loc'][0]).upper() if error['loc'] else 'settings'}: {error['msg']}"
            for error in exc.errors(include_input=False, include_url=False)
        ]
        console.print("[red]Invalid configuration (.env or RA_* variables):[/red]\n" + escape("\n".join(problems)))
        raise typer.Exit(2) from exc


@app.command()
def run(
    question: Annotated[str, typer.Argument(help="A research question or idea.")],
    upload: Annotated[
        list[Path] | None,
        typer.Option(
            "--upload", "-u", help="A PDF of a relevant paper, always read in full. Repeat for several.",
            exists=True, dir_okay=False, readable=True,
        ),
    ] = None,
    fail_at: Annotated[
        str | None, typer.Option(hidden=True, help="Debug: fail inside this stage.")
    ] = None,
    force_loops: Annotated[
        bool, typer.Option(hidden=True, help="Debug: take each loop once.")
    ] = False,
    refresh: Annotated[
        bool, typer.Option("--refresh", help="Search afresh instead of reusing kept search responses.")
    ] = False,
    run_id: Annotated[
        str | None, typer.Option("--run-id", hidden=True, help="The new run's ID (the web UI picks it).")
    ] = None,
) -> None:
    """Run the pipeline on a research question or idea."""
    settings = _settings()
    if refresh:
        settings = settings.model_copy(update={"response_cache_days": 0})
    if fail_at is not None and fail_at not in STAGE_NAMES:
        console.print(f"[red]Unknown stage {fail_at!r}.[/red] Stages: {', '.join(STAGE_NAMES)}")
        raise typer.Exit(2)
    try:
        result = start_run(
            settings, question, RunOptions(fail_at=fail_at, force_loops=force_loops), uploads=upload or [],
            run_id=run_id,
        )
    except RunError as exc:
        _print_failure(exc)
        raise typer.Exit(1) from exc
    _print_result(result)


@app.command()
def resume(run_id: Annotated[str, typer.Argument(help="The run to continue.")]) -> None:
    """Continue a failed or interrupted run from its last completed stage."""
    try:
        result = resume_run(_settings(), run_id)
    except CannotResume as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc
    except RunError as exc:
        _print_failure(exc)
        raise typer.Exit(1) from exc
    _print_result(result)


@app.command()
def novelty(idea: Annotated[str, typer.Argument(help="The idea to check, in a sentence or a short paragraph.")]) -> None:
    """Check how novel one idea is: search for close work, compare, and rate it 1-5."""
    try:
        result = check_idea_run(_settings(), idea)
    except RunError as exc:
        console.print(f"[red]Run {exc.run_id} failed:[/red] {escape(str(exc))}")
        console.print(f"Trace:  agent inspect {exc.run_id}\nCheck the idea again to retry.")
        raise typer.Exit(1) from exc
    _print_result(result)


@app.command()
def ideas(run_id: Annotated[str, typer.Argument(help="The run whose ideas to list.")]) -> None:
    """List every idea a run wrote or checked, with its novelty level and what became of it."""
    with Database(_settings().db_path) as db:
        if db.get_run(run_id) is None:
            console.print(f"[red]No run with ID {run_id}.[/red]")
            raise typer.Exit(1)
        found = IdeaStore(db).for_run(run_id)
    if not found:
        console.print(f"Run {run_id} has no ideas: its ideate stage has not run, or it had no gaps.")
        return
    table = Table(title=f"Ideas of {run_id}")
    for column in ("rank", "idea", "round", "status", "level", "title", "why"):
        table.add_column(column)
    for idea in sorted(found, key=lambda i: (i.final_rank or 99, i.lineage, i.round)):
        why = idea.detail.get("not_checked") or idea.detail.get("rejected_because") or (
            " ".join((idea.assessment or {}).get("reasons", [])[:2]))
        inconclusive = ", inconclusive" if (idea.assessment or {}).get("inconclusive") else ""
        table.add_row(str(idea.final_rank or ""), idea.idea_id, str(idea.round), idea.status,
                      f"{idea.level}{inconclusive}" if idea.level else "", escape(idea.title), escape(why))
    console.print(table)


@app.command()
def plan(
    run_id: Annotated[str, typer.Argument(help="The run whose experiment plan to show.")],
    element: Annotated[
        str | None,
        typer.Option("--element", "-e", help="Trace one element: its ID, or its field and position, e.g. 'baselines 2'."),
    ] = None,
) -> None:
    """Show a run's experiment plan with each element's verdict, or trace one element back to its sources."""
    with Database(_settings().db_path) as db:
        if db.get_run(run_id) is None:
            console.print(f"[red]No run with ID {run_id}.[/red]")
            raise typer.Exit(1)
        found = PlanStore(db).latest(run_id)
        if found is None:
            console.print(f"Run {run_id} has no plan: its plan stage has not run, or it had no assessed idea.")
            return
        if element is None:
            _print_plan(found)
            return
        wanted = " ".join(element.split()).casefold()
        chosen = next((e for e in found.elements if e.element_id == element or e.label.casefold() == wanted), None)
        if chosen is None:
            console.print(f"[red]No element {element!r} in the plan.[/red] Elements: "
                          + ", ".join(e.label for e in found.elements))
            raise typer.Exit(1)
        _print_trace(db, found, chosen)


def _print_plan(found: Any) -> None:
    c = found.detail.get("coverage") or {}
    table = Table(title=f"Experiment plan of {found.run_id} (round {found.round}, {found.status})")
    for column in ("element", "verdict", "text", "cites"):
        table.add_column(column)
    for e in found.elements:
        verdict = e.verdict or "unchecked"
        style = {"supported": "green", "contradicted": "red"}.get(verdict, "yellow")
        table.add_row(e.label, f"[{style}]{verdict}[/{style}]" + (" (removed)" if e.status == "removed" else ""),
                      escape(e.text), ", ".join(e.cites))
    console.print(table)
    if c.get("elements"):
        console.print(f"Provenance coverage: {c['supported']} of {c['elements']} elements supported ({c['coverage']:.0%}); "
                      f"{c.get('flagged', 0)} flagged, {c.get('removed', 0)} removed.")
    console.print(f"Trace one element: agent plan {found.run_id} --element \"{found.elements[0].label}\"" if found.elements else "")


def _print_trace(db: Database, found: Any, chosen: Element) -> None:
    """The element's chain: recommendation -> reasoning -> evidence -> paper -> section."""
    menu = Menu.from_json(found.menu, load_plan_papers(db, found.run_id))
    d, decision = chosen.derivation, chosen.decision
    tree = Tree(f"[bold]{escape(chosen.label)}[/bold]: {escape(chosen.text)}")
    verdict = tree.add(f"verdict: {chosen.verdict or 'unchecked'} ({chosen.status})"
                       + (f": {escape((chosen.verification or {}).get('reason', ''))}" if chosen.verification else ""))
    if chosen.verification and chosen.verification.get("model"):
        verdict.add(f"by {chosen.verification['model']}")
    reasoning = tree.add(f"decision: {escape(decision.get('reason') or 'none given')}")
    if decision.get("alternatives"):
        reasoning.add("chosen over: " + escape(", ".join(decision["alternatives"])))
    for answer in decision.get("answers", []):
        reasoning.add(f"answers critique point {answer['point']}: {escape(answer['problem'])}")
    tree.add(f"derivation: plan round {d.get('round')} ({d.get('change')}), {d.get('model')}, prompt {d.get('prompt')}"
             + (f"; checks: {escape('; '.join(d['checks']))}" if d.get("checks") else ""))
    if chosen.basis:
        tree.add(f"basis: {escape(chosen.basis)}")
    used = {e["sentence_id"] for e in chosen.evidence}
    for cite in chosen.cites:
        fact = menu.get(cite)
        if fact is None:
            continue
        branch = tree.add(f"[cyan]{fact.fact_id}[/cyan] {fact.kind}: {escape(fact.text)}")
        origin = fact.origin
        branch.add(f"from: {origin.get('from')}: {escape(origin.get('query', ''))}")
        for paper_id, info in (origin.get("papers") or {}).items():
            branch.add(f"paper {escape(menu.label(paper_id))}: {escape('; '.join(info.get('why', [])))}")
        for e in fact.evidence:
            rows = db.query(
                "SELECT t.text, s.heading, p.title, p.year, p.arxiv_id, p.doi FROM sentences t JOIN sections s"
                " ON s.section_id = t.section_id JOIN papers p ON p.paper_id = s.paper_id WHERE t.sentence_id = ?",
                (e["sentence_id"],))
            mark = "*" if e["sentence_id"] in used else " "
            if not rows:
                branch.add(f"{mark} {e['sentence_id']}: no longer stored")
                continue
            r = rows[0]
            ids = ", ".join(x for x in (f"arXiv:{r['arxiv_id']}" if r["arxiv_id"] else "", r["doi"] or "") if x)
            leaf = branch.add(f"{mark} \"{escape(e['quote'])}\" ({e['sentence_id']})")
            leaf.add(f"sentence: {escape(r['text'])}")
            leaf.add(f"section: {escape(r['heading'] or 'untitled')}")
            leaf.add(f"paper: {escape(r['title'])}" + (f" ({r['year']})" if r["year"] else "") + (f", {ids}" if ids else ""))
    console.print(tree)
    console.print("* marks the sentences the element's own evidence keeps.")


@app.command()
def inspect(
    run_id: Annotated[
        str | None, typer.Argument(help="The run to show. Omit to list recent runs.")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the raw trace events.")] = False,
) -> None:
    """Show a run's trace, or list recent runs."""
    with Database(_settings().db_path) as db:
        if run_id is None:
            _print_runs(db)
            return
        run_row = db.get_run(run_id)
        if run_row is None:
            console.print(f"[red]No run with ID {run_id}.[/red]")
            raise typer.Exit(1)
        events = TraceRecorder(db).events(run_id)
    if as_json:
        typer.echo(json.dumps(events, indent=2))
        return
    console.print(_trace_tree(dict(run_row), events))


@app.command()
def graph(
    run_id: Annotated[str, typer.Argument(help="The run whose graph to show.")],
    methods_on: Annotated[
        str | None, typer.Option("--methods-on", metavar="DATASET", help="Methods evaluated on a dataset.")
    ] = None,
    limitations_for: Annotated[
        str | None, typer.Option("--limitations-for", metavar="TASK", help="Limitations reported for a task.")
    ] = None,
    about: Annotated[
        str | None,
        typer.Option("--about", metavar="METHOD", help="Results about a method, with what supports or contradicts them."),
    ] = None,
    sentence: Annotated[
        str | None, typer.Option("--sentence", metavar="SENTENCE_ID", help="Everything that rests on a sentence.")
    ] = None,
    conflicts: Annotated[bool, typer.Option("--conflicts", help="Findings that contradict each other.")] = False,
    show_gaps: Annotated[bool, typer.Option("--gaps", help="The run's ranked research gaps.")] = False,
) -> None:
    """Show a run's knowledge graph, or ask it one question."""
    asked = [name for name, value in (("--methods-on", methods_on), ("--limitations-for", limitations_for),
                                      ("--about", about), ("--sentence", sentence), ("--conflicts", conflicts),
                                      ("--gaps", show_gaps)) if value]
    if len(asked) > 1:
        console.print(f"[red]Ask one question at a time[/red] (got {', '.join(asked)}).")
        raise typer.Exit(2)
    with Database(_settings().db_path) as db:
        if db.get_run(run_id) is None:
            console.print(f"[red]No run with ID {run_id}.[/red]")
            raise typer.Exit(1)
        knowledge = load_graph(db, run_id)
        if knowledge.number_of_nodes() == 0:
            built = any(e["step"] == "build_graph" and e["status"] == "ok" for e in TraceRecorder(db).events(run_id))
            if built:
                console.print(f"Run {run_id}'s knowledge graph is empty: its corpus had no papers.")
                return
            console.print(f"Run {run_id} has no knowledge graph yet: its build_graph stage has not run.")
            raise typer.Exit(1)
        if methods_on:
            title, answers = f"Methods evaluated on {methods_on}", queries.methods_on_dataset(knowledge, db, methods_on)
        elif limitations_for:
            title, answers = (f"Limitations reported for {limitations_for}",
                              queries.limitations_for_task(knowledge, db, limitations_for))
        elif about:
            title, answers = f"Results about {about}", queries.findings_about_method(knowledge, db, about)
        elif sentence:
            title, answers = f"Resting on {sentence}", queries.resting_on(knowledge, db, sentence)
        elif conflicts:
            title, answers = "Contradicting findings", queries.conflicts(knowledge, db)
        elif show_gaps:
            title, answers = "Research gaps", queries.gaps(knowledge, db)
        else:
            console.print(_graph_summary(run_id, knowledge))
            return
    if not answers:
        console.print(f"{escape(title)}: nothing found.")
        return
    table = Table(title=escape(title), show_lines=True)
    for column in ("", "detail", "evidence"):
        table.add_column(column)
    for answer in answers:
        evidence = "\n".join(f"{e['sentence_id']}: \"{e['quote']}\"" for e in answer.evidence[:3])
        table.add_row(escape(answer.subject), escape(answer.detail), escape(evidence))
    console.print(table)


def _graph_summary(run_id: str, knowledge: Any) -> Table:
    papers = {n for n, d in knowledge.nodes(data=True) if d["node_type"] == "paper"}
    table = Table(title=f"Knowledge graph of {run_id}")
    table.add_column("kind")
    table.add_column("count", justify="right")
    table.add_column("most connected (papers)")
    for kind in sorted({d["node_type"] for _, d in knowledge.nodes(data=True)}):
        nodes = [n for n, d in knowledge.nodes(data=True) if d["node_type"] == kind]
        top = ""
        if kind in ("task", "method", "dataset", "metric"):
            reach = sorted(
                ((len({src for src, _ in knowledge.in_edges(n) if src in papers}), knowledge.nodes[n]["name"]) for n in nodes),
                reverse=True,
            )
            top = ", ".join(f"{name} ({count})" for count, name in reach[:6])
        table.add_row(kind, str(len(nodes)), escape(top))
    edge_counts: dict[str, int] = {}
    for _, _, data in knowledge.edges(data=True):
        edge_counts[data["edge_type"]] = edge_counts.get(data["edge_type"], 0) + 1
    for kind, count in sorted(edge_counts.items()):
        table.add_row(f"edge: {kind}", str(count), "")
    return table


@app.command()
def report(run_id: Annotated[str, typer.Argument(help="The run whose report to write again.")]) -> None:
    """Write a run's report again from its stored results, e.g. after its graph or gaps were rebuilt."""
    settings = _settings()
    with Database(settings.db_path) as db:
        if db.get_run(run_id) is None:
            console.print(f"[red]No run with ID {run_id}.[/red]")
            raise typer.Exit(1)
        rebuilt = stored_report(db, run_id)
        md_path, json_path = write_report(rebuilt, settings.runs_dir / run_id)
        provenance = write_provenance(db, run_id, settings.runs_dir / run_id)
    console.print(f"Report written: {md_path} (and {json_path.name}" + (f", {provenance.name})" if provenance else ")"))


@app.command()
def cost(run_id: Annotated[str, typer.Argument(help="The run whose cost to show.")]) -> None:
    """Show what a run cost: time per stage, model calls and tokens, API requests, and what it reused."""
    with Database(_settings().db_path) as db:
        if db.get_run(run_id) is None:
            console.print(f"[red]No run with ID {run_id}.[/red]")
            raise typer.Exit(1)
        found = run_cost(db, run_id)
    console.print(Markdown("\n".join([f"# Cost of {run_id}", "", *found.markdown()])))


@app.command()
def ui(
    port: Annotated[int, typer.Option(help="The port to listen on (127.0.0.1 only).")] = 8000,
    open_browser: Annotated[bool, typer.Option("--open/--no-open", help="Open the UI in the browser.")] = True,
) -> None:
    """Open the web UI: start and resume runs, watch them live, and browse what they found."""
    import threading
    import webbrowser

    import uvicorn

    from research_agent.web.app import create_app

    settings = _settings()
    url = f"http://127.0.0.1:{port}"
    console.print(f"Research Agent UI: {url}  (Ctrl+C stops the UI; runs it started keep going)")
    if open_browser:
        threading.Timer(1.5, webbrowser.open, (url,)).start()
    uvicorn.run(create_app(settings), host="127.0.0.1", port=port, log_level="warning")


def _print_result(result: RunResult) -> None:
    report_md = result.report_dir / "report.md"
    console.print(Markdown(report_md.read_text(encoding="utf-8")))
    console.print()
    console.print(
        f"[green]Run {result.run_id} completed[/green]: {len(result.stages_run)} stages.\n"
        f"Report: {report_md} (and report.json)\n"
        f"Trace:  agent inspect {result.run_id}"
    )


def _print_failure(exc: RunError) -> None:
    console.print(f"[red]Run {exc.run_id} failed:[/red] {escape(str(exc))}")
    console.print(f"Trace:  agent inspect {exc.run_id}\nResume: agent resume {exc.run_id}")


def _print_runs(db: Database) -> None:
    runs = db.list_runs()
    if not runs:
        console.print("No runs yet.")
        return
    table = Table()
    table.add_column("run", no_wrap=True)
    for column in ("status", "created", "question"):
        table.add_column(column)
    for row in runs:
        table.add_row(row["run_id"], row["status"], row["created_at"], escape(row["question"]))
    console.print(table)


def _trace_tree(run_row: dict[str, Any], events: list[dict[str, Any]]) -> Tree:
    tree = Tree(
        f"[bold]{run_row['run_id']}[/bold]  {run_row['status']}  {escape(run_row['question'])}"
    )
    children: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        if event["parent_event_id"]:
            children.setdefault(event["parent_event_id"], []).append(event)

    def add(branch: Tree, event: dict[str, Any]) -> None:
        node = branch.add(_event_label(event))
        for child in children.get(event["event_id"], []):
            add(node, child)

    for event in events:
        if not event["parent_event_id"]:
            add(tree, event)
    return tree


def _event_label(event: dict[str, Any]) -> str:
    if event["activity_type"] == "step":
        name = f"[bold]{event['step']}[/bold]"
        if event["attempt"] > 1:
            name += f" (attempt {event['attempt']})"
    else:
        agent = event["agent"]
        who = agent.get("name") or "/".join(filter(None, (agent.get("role"), agent.get("model"))))
        name = f"{event['activity_type']} {who}"
    ms = (
        datetime.fromisoformat(event["ended_at"]) - datetime.fromisoformat(event["started_at"])
    ).total_seconds() * 1000
    status = "[green]ok[/green]" if event["status"] == "ok" else f"[red]{escape(event['error'])}[/red]"
    parts = [f"{event['seq']:>3}  {name}", status, f"{ms:.0f} ms"]
    if event["prompt_version"]:
        parts.append(f"prompt {event['prompt_version']}")
    parts.append(
        f"used {len(event['used'])}, generated {len(event['generated'])},"
        f" dropped {len(event['dropped'])}"
    )
    return "  ".join(parts)
