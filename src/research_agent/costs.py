"""What a run cost, read from its trace: time per stage, model calls and tokens, API
requests, and how much earlier work it reused (kept LLM replies, extractions,
embeddings, search responses).

Time counts the stages' own running time, failed attempts included; a run resumed
after a stop also reports the time from its first stage to its last (``elapsed``).
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel

from research_agent.storage import Database
from research_agent.trace import TraceRecorder

LLM_SERVICES = ("google", "openrouter", "groq")  # their HTTP requests count under the models
STAGE_ROWS = 25


class StageCost(BaseModel):
    stage: str
    attempts: int = 0
    failed: int = 0
    seconds: float = 0.0
    llm_calls: int = 0
    api_requests: int = 0  # literature API requests sent (not from the cache)


class ModelCost(BaseModel):
    model: str  # provider:model, or "no model answered" (every model of the role was unavailable)
    role: str
    calls: int = 0
    failed: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0  # part of the output


class ApiCost(BaseModel):
    service: str
    requests: int = 0  # sent, retries included
    cached: int = 0  # answered from kept responses
    failed: int = 0


class Reuse(BaseModel):
    kind: str
    reused: int = 0
    asked: int = 0

    @property
    def rate(self) -> float | None:
        total = self.reused + self.asked
        return self.reused / total if total else None


class RunCost(BaseModel):
    seconds: float = 0.0
    elapsed: float = 0.0
    failed_attempts: int = 0
    stages: list[StageCost] = []
    models: list[ModelCost] = []
    apis: list[ApiCost] = []
    reuse: list[Reuse] = []

    def reuse_of(self, kinds: tuple[str, ...] | None = None, exclude: tuple[str, ...] = ()) -> Reuse:
        found = [r for r in self.reuse if (kinds is None or r.kind in kinds) and r.kind not in exclude]
        return Reuse(kind="+".join(r.kind for r in found), reused=sum(r.reused for r in found),
                     asked=sum(r.asked for r in found))

    def markdown(self) -> list[str]:
        if not self.stages:
            return ["_No stage has run._"]
        lines = [f"- **Time:** {_minutes(self.seconds)} in stages"
                 + (f", {_minutes(self.elapsed)} from the first stage to the last" if self.elapsed > self.seconds + 60 else "")
                 + (f"; {self.failed_attempts} stage attempt(s) failed and were run again" if self.failed_attempts else "")]
        if self.models:
            lines.append("- **Models:** " + "; ".join(
                f"{m.model} ({m.role}): {m.calls} call(s){f', {m.failed} failed' if m.failed else ''}, "
                f"{m.input_tokens:,} tokens in, {m.output_tokens:,} out"
                + (f" ({m.reasoning_tokens:,} thinking)" if m.reasoning_tokens else "") for m in self.models))
        apis = [a for a in self.apis if a.service not in LLM_SERVICES]
        if apis:
            lines.append("- **APIs:** " + "; ".join(
                f"{a.service} {a.requests} request(s)" + (f", {a.cached} from kept responses" if a.cached else "")
                + (f", {a.failed} failed" if a.failed else "") for a in apis))
        replies, embedding = self.reuse_of(exclude=("embedding",)), self.reuse_of(("embedding",))
        parts = []
        if replies.reused + replies.asked:
            parts.append(f"LLM replies {replies.reused} of {replies.reused + replies.asked} kept from earlier ({replies.rate:.0%})")
        if embedding.reused + embedding.asked:
            parts.append(f"embeddings {embedding.reused} of {embedding.reused + embedding.asked} ({embedding.rate:.0%})")
        cached = sum(a.cached for a in apis)
        if cached + sum(a.requests for a in apis):
            searched = sum(a.cached + a.requests for a in apis)
            parts.append(f"API responses {cached} of {searched} ({cached / searched:.0%})")
        if parts:
            lines.append("- **Reused:** " + "; ".join(parts))
        lines += ["", "| stage | runs | time | LLM calls | API requests |", "|---|---|---|---|---|"]
        for stage in self.stages[:STAGE_ROWS]:
            runs = f"{stage.attempts}" + (f" ({stage.failed} failed)" if stage.failed else "")
            lines.append(f"| {stage.stage} | {runs} | {_minutes(stage.seconds)} | {stage.llm_calls} | {stage.api_requests} |")
        return lines


def run_cost(db: Database, run_id: str) -> RunCost:
    events = TraceRecorder(db).events(run_id)
    cost = RunCost()
    if not events:
        return cost
    stages: dict[str, StageCost] = {}
    models: dict[str, ModelCost] = {}
    apis: dict[str, ApiCost] = {}
    reuse: dict[str, Reuse] = {}
    for event in events:
        step = stages.setdefault(event["step"], StageCost(stage=event["step"]))
        detail, agent = event["detail"], event["agent"]
        if event["activity_type"] == "step":
            step.attempts += 1
            step.failed += event["status"] == "error"
            step.seconds += _seconds(event)
            for kind, counts in (detail.get("kept") or {}).items():
                entry = reuse.setdefault(kind, Reuse(kind=kind))
                entry.reused += counts.get("reused", 0)
                entry.asked += counts.get("asked", 0)
        elif event["activity_type"] == "llm_call":
            step.llm_calls += 1
            name = f"{agent['provider']}:{agent['model']}" if event["status"] == "ok" and agent.get("model") else "no model answered"
            model = models.setdefault(name, ModelCost(model=name, role=agent.get("role", "")))
            model.calls += 1
            model.failed += event["status"] == "error"
            usage = detail.get("usage") or {}
            model.input_tokens += usage.get("input_tokens", 0)
            model.output_tokens += usage.get("output_tokens", 0)
            model.reasoning_tokens += usage.get("reasoning_tokens", 0)
        elif event["activity_type"] == "api_call":
            api = apis.setdefault(agent.get("name", "?"), ApiCost(service=agent.get("name", "?")))
            if detail.get("cached"):
                api.cached += 1
            else:
                sent = len(detail.get("sends") or []) or 1
                api.requests += sent
                if api.service not in LLM_SERVICES:
                    step.api_requests += sent
            api.failed += event["status"] == "error"
    cost.stages = list(stages.values())
    cost.models = sorted(models.values(), key=lambda m: (m.role, -m.calls))
    cost.apis = sorted(apis.values(), key=lambda a: a.service)
    cost.reuse = sorted(reuse.values(), key=lambda r: r.kind)
    cost.seconds = round(sum(s.seconds for s in cost.stages), 1)
    cost.failed_attempts = sum(s.failed for s in cost.stages)
    steps = [e for e in events if e["activity_type"] == "step"]
    if steps:
        cost.elapsed = round((_time(steps[-1]["ended_at"]) - _time(steps[0]["started_at"])).total_seconds(), 1)
    return cost


def _time(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp)


def _seconds(event: dict[str, Any]) -> float:
    return round((_time(event["ended_at"]) - _time(event["started_at"])).total_seconds(), 1)


def _minutes(seconds: float) -> str:
    return f"{seconds:.0f} s" if seconds < 120 else f"{seconds / 60:.1f} min"
