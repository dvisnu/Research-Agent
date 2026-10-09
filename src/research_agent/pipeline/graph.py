"""The LangGraph pipeline: a fixed sequence of stages with two loops.

    brief -> search -> expand -> rank -> fulltext -> extract -> build_graph
      -> related_work -> landscape -> gaps -> ideate -> novelty_search -> novelty_assess
      novelty_assess -> ideate               (idea scored too low; capped)
      novelty_assess -> plan -> critique
      critique -> targeted_search -> plan    (information missing; capped)
      critique -> verify -> report

The Phase 1 stages (brief to fulltext) and ``report`` do real work; the rest
are stubs that record a trace event and pass the state on. The state holds IDs
and small values only; papers, sentences and graph data live in SQLite, so
checkpoints stay small.
"""

import operator
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from research_agent.config import Settings
from research_agent.literature import stages as literature
from research_agent.literature.embed import Embedder
from research_agent.literature.sources import Sources
from research_agent.llm import LLM
from research_agent.pipeline.stages import STAGES
from research_agent.report import RunReport, write_report
from research_agent.storage import Database, utc_now
from research_agent.trace import Event, TraceRecorder, ref

MAX_NOVELTY_ROUNDS = 3
MAX_CRITIQUE_ROUNDS = 2


class PipelineState(TypedDict, total=False):
    run_id: str
    question: str
    stages_run: Annotated[list[str], operator.add]
    brief: dict[str, Any]  # a ResearchBrief
    uploads: list[str]  # paths of user-provided PDFs
    novelty_round: int
    novelty_retry: bool
    critique_round: int
    critique_needs_search: bool


@dataclass(frozen=True)
class RunOptions:
    """Debug switches for one invocation. They are not checkpointed, so a resume runs without them."""

    fail_at: str | None = None  # raise inside this stage, to test resuming
    force_loops: bool = False  # take each loop once, to test loops and checkpoints


@dataclass
class Services:
    """What the stages call out to; tests swap in fakes."""

    sources: Sources
    embedder: Embedder
    llm: LLM

    def close(self) -> None:
        self.sources.close()
        self.llm.close()


@dataclass
class PipelineContext:
    settings: Settings
    db: Database
    recorder: TraceRecorder
    services: Services
    options: RunOptions = field(default_factory=RunOptions)

    @property
    def sources(self) -> Sources:
        return self.services.sources

    @property
    def embedder(self) -> Embedder:
        return self.services.embedder

    @property
    def llm(self) -> LLM:
        return self.services.llm


class ForcedFailure(RuntimeError):
    pass


StageWork = Callable[[PipelineState, PipelineContext, Event], dict[str, Any]]


def _stub(state: PipelineState, ctx: PipelineContext, event: Event) -> dict[str, Any]:
    event.detail["stub"] = True
    return {}


def _ideate(state: PipelineState, ctx: PipelineContext, event: Event) -> dict[str, Any]:
    event.detail["stub"] = True
    return {"novelty_round": state.get("novelty_round", 0) + 1}


def _novelty_assess(state: PipelineState, ctx: PipelineContext, event: Event) -> dict[str, Any]:
    event.detail["stub"] = True
    return {"novelty_retry": ctx.options.force_loops and state.get("novelty_round", 0) < 2}


def _critique(state: PipelineState, ctx: PipelineContext, event: Event) -> dict[str, Any]:
    event.detail["stub"] = True
    round_ = state.get("critique_round", 0) + 1
    return {"critique_round": round_, "critique_needs_search": ctx.options.force_loops and round_ < 2}


def _report(state: PipelineState, ctx: PipelineContext, event: Event) -> dict[str, Any]:
    report = RunReport(run_id=state["run_id"], question=state["question"], generated_at=utc_now())
    paths = write_report(report, ctx.settings.runs_dir / state["run_id"])
    event.generated += [ref("file", str(path)) for path in paths]
    return {}


_WORK: dict[str, StageWork] = {
    "brief": literature.brief_stage,
    "search": literature.search_stage,
    "expand": literature.expand_stage,
    "rank": literature.rank_stage,
    "fulltext": literature.fulltext_stage,
    "ideate": _ideate,
    "novelty_assess": _novelty_assess,
    "critique": _critique,
    "report": _report,
}


def _node(name: str, ctx: PipelineContext) -> Callable[[PipelineState], dict[str, Any]]:
    work = _WORK.get(name, _stub)

    def run_stage(state: PipelineState) -> dict[str, Any]:
        used = [ref("question", state["run_id"])] if name == "brief" else []
        with ctx.recorder.step(state["run_id"], name, used=used) as event:
            if ctx.options.fail_at == name:
                raise ForcedFailure(f"forced failure in stage {name!r}")
            update = work(state, ctx, event)
        return {"stages_run": [name], **update}

    return run_stage


def _after_novelty(state: PipelineState) -> str:
    if state.get("novelty_retry") and state.get("novelty_round", 0) < MAX_NOVELTY_ROUNDS:
        return "ideate"
    return "plan"


def _after_critique(state: PipelineState) -> str:
    if state.get("critique_needs_search") and state.get("critique_round", 0) < MAX_CRITIQUE_ROUNDS:
        return "targeted_search"
    return "verify"


_SEQUENCE = (
    "brief",
    "search",
    "expand",
    "rank",
    "fulltext",
    "extract",
    "build_graph",
    "related_work",
    "landscape",
    "gaps",
    "ideate",
    "novelty_search",
    "novelty_assess",
)


def build_graph(ctx: PipelineContext, checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    builder = StateGraph(PipelineState)
    for stage in STAGES:
        builder.add_node(stage.name, _node(stage.name, ctx))

    builder.add_edge(START, _SEQUENCE[0])
    for src, dst in zip(_SEQUENCE, _SEQUENCE[1:]):
        builder.add_edge(src, dst)
    builder.add_conditional_edges("novelty_assess", _after_novelty, ["ideate", "plan"])
    builder.add_edge("plan", "critique")
    builder.add_conditional_edges("critique", _after_critique, ["targeted_search", "verify"])
    builder.add_edge("targeted_search", "plan")
    builder.add_edge("verify", "report")
    builder.add_edge("report", END)
    return builder.compile(checkpointer=checkpointer)
