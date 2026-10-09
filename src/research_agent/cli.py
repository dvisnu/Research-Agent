"""Command line: `agent run`, `agent resume`, `agent inspect`."""

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
from research_agent.pipeline import (
    STAGE_NAMES,
    CannotResume,
    RunError,
    RunOptions,
    RunResult,
    resume_run,
    start_run,
)
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
) -> None:
    """Run the pipeline on a research question or idea."""
    settings = _settings()
    if fail_at is not None and fail_at not in STAGE_NAMES:
        console.print(f"[red]Unknown stage {fail_at!r}.[/red] Stages: {', '.join(STAGE_NAMES)}")
        raise typer.Exit(2)
    try:
        result = start_run(
            settings, question, RunOptions(fail_at=fail_at, force_loops=force_loops), uploads=upload or []
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
