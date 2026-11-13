"""The run trace: an append-only, PROV-DM-shaped event log.

Each event is a PROV *activity*: one execution of a pipeline step, or an LLM or
API call made inside one. ``used`` and ``generated`` hold entity references
(see ``ref``), ``agent`` says who performed the activity, and ``dropped``
records what it considered but did not keep, with the reason, so evidence
omission can be measured later.

Calls made inside a step (``child_event``) attach to that step through a
context variable, so clients and the LLM interface need no run plumbing.
Outside a run they are not recorded. So do the counts of kept work a step reused
or had to ask for (``count``), which measure the caches.
"""

import json
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from research_agent.storage import Database, new_id, utc_now


def ref(kind: str, entity_id: str) -> str:
    """An entity reference, e.g. ``ref("sentence", "p_1:s2:3")``."""
    return f"{kind}:{entity_id}"


@dataclass
class Event:
    run_id: str
    step: str
    activity_type: str
    agent: dict[str, Any]
    used: list[str]
    prompt_version: str | None
    parent_event_id: str | None
    recorder: "TraceRecorder | None"  # None: not recorded (a call made outside a run)
    event_id: str = field(default_factory=lambda: new_id("ev"))
    generated: list[str] = field(default_factory=list)
    dropped: list[dict[str, str]] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=utc_now)

    def drop(self, entity: str, reason: str) -> None:
        self.dropped.append({"ref": entity, "reason": reason})


_current: ContextVar[Event | None] = ContextVar("current_trace_event", default=None)


@contextmanager
def _record(event: Event) -> Iterator[Event]:
    token = _current.set(event)
    status, error = "ok", None
    try:
        yield event
    except BaseException as exc:
        status, error = "error", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _current.reset(token)
        if event.recorder is not None:
            event.recorder._write(event, status, error)


class TraceRecorder:
    def __init__(self, db: Database) -> None:
        self.db = db

    @contextmanager
    def step(
        self, run_id: str, step: str, *, used: Iterable[str] = ()
    ) -> Iterator[Event]:
        """Record one execution of a pipeline step."""
        event = Event(
            run_id=run_id,
            step=step,
            activity_type="step",
            agent={"kind": "pipeline"},
            used=list(used),
            prompt_version=None,
            parent_event_id=None,
            recorder=self,
        )
        with _record(event):
            yield event

    def _write(self, event: Event, status: str, error: str | None) -> None:
        with self.db.transaction() as conn:
            seq = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM trace_events WHERE run_id = ?",
                (event.run_id,),
            ).fetchone()[0]
            attempt = None
            if event.activity_type == "step":
                attempt = conn.execute(
                    "SELECT COUNT(*) + 1 FROM trace_events"
                    " WHERE run_id = ? AND step = ? AND activity_type = 'step'",
                    (event.run_id, event.step),
                ).fetchone()[0]
            conn.execute(
                "INSERT INTO trace_events (event_id, run_id, seq, parent_event_id, step, attempt,"
                " activity_type, agent, prompt_version, used, generated, dropped, detail, status,"
                " error, started_at, ended_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event.event_id,
                    event.run_id,
                    seq,
                    event.parent_event_id,
                    event.step,
                    attempt,
                    event.activity_type,
                    json.dumps(event.agent),
                    event.prompt_version,
                    json.dumps(event.used),
                    json.dumps(event.generated),
                    json.dumps(event.dropped),
                    json.dumps(event.detail, default=str),
                    status,
                    error,
                    event.started_at,
                    utc_now(),
                ),
            )

    def events(self, run_id: str) -> list[dict[str, Any]]:
        """All events of a run in write order, with JSON fields decoded."""
        rows = self.db.query("SELECT * FROM trace_events WHERE run_id = ? ORDER BY seq", (run_id,))
        events = []
        for row in rows:
            event = dict(row)
            for key in ("agent", "used", "generated", "dropped", "detail"):
                event[key] = json.loads(event[key])
            events.append(event)
        return events


@contextmanager
def child_event(
    activity_type: str,
    *,
    agent: dict[str, Any],
    used: Iterable[str] = (),
    prompt_version: str | None = None,
) -> Iterator[Event]:
    """Record a call made inside the current step. Outside a run, nothing is recorded."""
    parent = _current.get()
    event = Event(
        run_id=parent.run_id if parent else "",
        step=parent.step if parent else "",
        activity_type=activity_type,
        agent=agent,
        used=list(used),
        prompt_version=prompt_version,
        parent_event_id=parent.event_id if parent else None,
        recorder=parent.recorder if parent else None,
    )
    with _record(event):
        yield event


def count(kind: str, *, reused: int = 0, asked: int = 0) -> None:
    """Count, on the current step, lookups of kept work of one kind (an LLM reply kind,
    extractions, embeddings): ``reused`` were found kept, ``asked`` had to be produced."""
    event = _current.get()
    if event is None:
        return
    kept = event.detail.setdefault("kept", {}).setdefault(kind, {"reused": 0, "asked": 0})
    kept["reused"] += reused
    kept["asked"] += asked
