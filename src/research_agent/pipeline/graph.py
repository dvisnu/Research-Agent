"""The LangGraph pipeline: a fixed sequence of stages with two loops.

    brief -> search -> expand -> rank -> fulltext -> extract -> build_graph
      -> related_work -> landscape -> gaps -> ideate -> novelty_search -> novelty_assess
      novelty_assess -> ideate               (idea scored too low; capped)
      novelty_assess -> plan -> critique
      critique -> targeted_search -> plan    (the critique made points; capped)
      critique -> verify -> report

Every stage does real work. The state holds IDs and small values only; papers,
sentences, graph data, ideas and plans live in SQLite, so checkpoints stay small.
"""

import operator
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from research_agent.config import Settings
from research_agent.costs import run_cost
from research_agent.embeddings import TextEmbedder
from research_agent.gaps import stages as gap_stages
from research_agent.graph import stages as graph_stages
from research_agent.ideas import stages as idea_stages
from research_agent.ideas.report import idea_reports
from research_agent.literature import stages as literature
from research_agent.literature.embed import Embedder
from research_agent.literature.sources import Sources
from research_agent.llm import LLM
from research_agent.pipeline.stages import STAGES
from research_agent.plans import stages as plan_stages
from research_agent.plans.prov import write_provenance
from research_agent.plans.report import plan_report
from research_agent.report import KnowledgeGraphSummary, RunReport, write_report
from research_agent.storage import Database, utc_now
from research_agent.trace import Event, TraceRecorder, ref

MAX_NOVELTY_ROUNDS = idea_stages.MAX_ROUNDS
MAX_CRITIQUE_ROUNDS = plan_stages.MAX_CRITIQUE_ROUNDS


class PipelineState(TypedDict, total=False):
    run_id: str
    question: str
    stages_run: Annotated[list[str], operator.add]
    brief: dict[str, Any]  # a ResearchBrief
    uploads: list[str]  # paths of user-provided PDFs
    extraction: dict[str, Any]  # what the extract stage did: calls, reuse, failures
    graph: dict[str, Any]  # the knowledge graph's summary
    related_work: list[dict[str, Any]]  # the closest papers
    landscape: dict[str, Any]  # the methods x datasets grid
    gaps: list[dict[str, Any]]  # the top gaps, with their evidence
    ideas: list[dict[str, Any]]  # a summary of each idea checked; the ideas live in SQLite
    novelty_round: int
    novelty_retry: bool
    novelty_filter: str  # override the novelty search's embedding filter (the duplicate test)
    plan: dict[str, Any] | None  # a summary of the latest plan version; the plan lives in SQLite
    plan_round: int
    critique_round: int
    critique_revise: bool  # the critique made points: search for what is missing, then revise


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
    text_embedder: TextEmbedder | None = None  # for claims; None when RA_EMBEDDING is unset

    def close(self) -> None:
        self.sources.close()
        self.llm.close()
        close = getattr(self.text_embedder, "close", None)
        if close is not None:
            close()


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

    @property
    def text_embedder(self) -> TextEmbedder | None:
        return self.services.text_embedder


class ForcedFailure(RuntimeError):
    pass


StageWork = Callable[[PipelineState, PipelineContext, Event], dict[str, Any]]


def _report(state: PipelineState, ctx: PipelineContext, event: Event) -> dict[str, Any]:
    report = RunReport(run_id=state["run_id"], question=state["question"], generated_at=utc_now())
    if state.get("graph"):
        report.knowledge_graph = KnowledgeGraphSummary.from_stages(state.get("extraction") or {}, state["graph"])
    report.add_phase3(state.get("related_work") or [], state.get("landscape"), state.get("gaps") or [])
    report.add_phase4(idea_reports(ctx.db, state["run_id"]))
    report.add_phase5(plan_report(ctx.db, state["run_id"]))
    report.cost = run_cost(ctx.db, state["run_id"])
    out_dir = ctx.settings.runs_dir / state["run_id"]
    paths = [*write_report(report, out_dir), write_provenance(ctx.db, state["run_id"], out_dir)]
    event.generated += [ref("file", str(path)) for path in paths if path is not None]
    return {}


_WORK: dict[str, StageWork] = {
    "brief": literature.brief_stage,
    "search": literature.search_stage,
    "expand": literature.expand_stage,
    "rank": literature.rank_stage,
    "fulltext": literature.fulltext_stage,
    "extract": graph_stages.extract_stage,
    "build_graph": graph_stages.build_graph_stage,
    "related_work": gap_stages.related_work_stage,
    "landscape": gap_stages.landscape_stage,
    "gaps": gap_stages.gaps_stage,
    "ideate": idea_stages.ideate_stage,
    "novelty_search": idea_stages.novelty_search_stage,
    "novelty_assess": idea_stages.novelty_assess_stage,
    "plan": plan_stages.plan_stage,
    "critique": plan_stages.critique_stage,
    "targeted_search": plan_stages.targeted_search_stage,
    "verify": plan_stages.verify_stage,
    "report": _report,
}


def _node(name: str, ctx: PipelineContext) -> Callable[[PipelineState], dict[str, Any]]:
    work = _WORK[name]

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
    if state.get("critique_revise") and state.get("critique_round", 0) < MAX_CRITIQUE_ROUNDS:
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
