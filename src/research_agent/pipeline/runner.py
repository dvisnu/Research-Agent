"""Start and resume runs. Each run is a LangGraph thread whose ID is the run ID,
checkpointed after every stage so a failed or interrupted run can continue."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph

from research_agent.config import Settings
from research_agent.embeddings import text_embedder
from research_agent.ideas.stages import check_idea
from research_agent.literature.embed import Specter2
from research_agent.literature.sources import Sources
from research_agent.llm import LLM
from research_agent.pipeline.graph import PipelineContext, RunOptions, Services, build_graph
from research_agent.pipeline.stored_report import stored_report
from research_agent.report import write_report
from research_agent.storage import Database, new_id
from research_agent.trace import TraceRecorder, ref

# Super-steps per invocation; the stages plus every loop round fit well inside this.
RECURSION_LIMIT = 100


@dataclass
class RunResult:
    run_id: str
    stages_run: list[str]
    report_dir: Path


class RunError(RuntimeError):
    """The pipeline raised; the run can be resumed."""

    def __init__(self, run_id: str, message: str) -> None:
        super().__init__(message)
        self.run_id = run_id


class CannotResume(RuntimeError):
    pass


def default_services(settings: Settings) -> Services:
    return Services(
        sources=Sources.from_settings(settings), embedder=Specter2(), llm=LLM(settings),
        text_embedder=text_embedder(settings),
    )


@contextmanager
def open_pipeline(
    settings: Settings, options: RunOptions | None = None, services: Services | None = None
) -> Iterator[tuple[CompiledStateGraph, PipelineContext]]:
    owned = services is None
    services = services or default_services(settings)
    db = Database(settings.db_path)
    checkpoints = sqlite3.connect(settings.checkpoint_path, check_same_thread=False)
    try:
        ctx = PipelineContext(settings, db, TraceRecorder(db), services, options or RunOptions())
        yield build_graph(ctx, SqliteSaver(checkpoints)), ctx
    finally:
        checkpoints.close()
        db.close()
        if owned:
            services.close()


def start_run(
    settings: Settings,
    question: str,
    options: RunOptions | None = None,
    services: Services | None = None,
    uploads: list[Path] | tuple[Path, ...] = (),
    run_id: str | None = None,
) -> RunResult:
    """Run the pipeline on a question, with optional user-provided PDFs. ``run_id`` lets a
    caller that watches the run (the web UI) know its ID before it starts."""
    run_id = run_id or new_id("run")
    with open_pipeline(settings, options, services) as (graph, ctx):
        ctx.db.create_run(run_id, question)
        state = {"run_id": run_id, "question": question, "uploads": [str(p.resolve()) for p in uploads]}
        return _invoke(graph, ctx, run_id, state)


def resume_run(settings: Settings, run_id: str, services: Services | None = None) -> RunResult:
    with open_pipeline(settings, services=services) as (graph, ctx):
        if ctx.db.get_run(run_id) is None:
            raise CannotResume(f"No run with ID {run_id}.")
        if not graph.get_state(_config(run_id)).next:
            raise CannotResume(f"Run {run_id} has already completed.")
        ctx.db.set_run_status(run_id, "running")
        return _invoke(graph, ctx, run_id, None)


def check_idea_run(
    settings: Settings, idea: str, services: Services | None = None, *, origin: str = "user",
    facets: dict[str, str] | None = None, detail: dict[str, Any] | None = None, method: str | None = None,
) -> RunResult:
    """Check one idea's novelty in a run of its own (no literature review, gaps or plan):
    the novelty search, the comparisons and the level, then the report. Such a run is not
    checkpointed; if it fails, check the idea again."""
    run_id = new_id("run")
    with open_pipeline(settings, services=services) as (_, ctx):
        ctx.db.create_run(run_id, idea)
        try:
            check_idea(ctx, run_id, origin, facets=facets, text=idea, detail=detail, method=method)
            with ctx.recorder.step(run_id, "report") as event:
                paths = write_report(stored_report(ctx.db, run_id), settings.runs_dir / run_id)
                event.generated += [ref("file", str(path)) for path in paths]
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            ctx.db.set_run_status(run_id, "failed", error)
            raise RunError(run_id, error) from exc
        ctx.db.set_run_status(run_id, "completed")
    return RunResult(run_id, ["ideate", "novelty_search", "novelty_assess", "report"], settings.runs_dir / run_id)


def _config(run_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": run_id}, "recursion_limit": RECURSION_LIMIT}


def _invoke(
    graph: CompiledStateGraph, ctx: PipelineContext, run_id: str, graph_input: dict[str, Any] | None
) -> RunResult:
    try:
        state = graph.invoke(graph_input, _config(run_id))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        ctx.db.set_run_status(run_id, "failed", error)
        raise RunError(run_id, error) from exc
    ctx.db.set_run_status(run_id, "completed")
    return RunResult(run_id, state["stages_run"], ctx.settings.runs_dir / run_id)
