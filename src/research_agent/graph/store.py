"""Kept LLM replies: chunk extractions and other decisions (name groupings, claim-pair labels,
the brief, relevance labels, ...).

A reply is reused when the model would be shown exactly the same thing again
(same prompt version, same input) and the model that gave it is still one the
role is configured with, so changing a role's models makes the work happen again.
"""

import hashlib
import json
import sqlite3
from typing import Any

from pydantic import BaseModel

from research_agent.llm import LLM, Prompt, Role
from research_agent.storage import Database, new_id, utc_now
from research_agent.trace import count


def digest(*parts: str) -> str:
    """A hash of the parts, kept apart so ("ab", "c") and ("a", "bc") differ."""
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(part.encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


class ExtractionStore:
    def __init__(self, db: Database, models: list[str]) -> None:
        self.db = db
        self.models = models  # the extraction role's models: replies from others are not reused

    def find(self, input_hash: str) -> sqlite3.Row | None:
        rows = self.db.query(
            f"SELECT * FROM extractions WHERE input_hash = ? AND model IN ({','.join('?' * len(self.models))})"
            " ORDER BY created_at DESC LIMIT 1",
            (input_hash, *self.models),
        )
        count("extraction", reused=len(rows[:1]), asked=1 - len(rows[:1]))
        return rows[0] if rows else None

    def add(
        self,
        *,
        paper_id: str,
        kind: str,
        chunk: int,
        sections: list[int],
        input_hash: str,
        prompt_version: str,
        model: str,
        output: dict[str, Any],
    ) -> str:
        extraction_id = new_id("x")
        self.db.execute(
            "INSERT INTO extractions (extraction_id, paper_id, kind, chunk, sections, input_hash,"
            " prompt_version, model, output, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (extraction_id, paper_id, kind, chunk, json.dumps(sections), input_hash, prompt_version,
             model, json.dumps(output), utc_now()),
        )
        return extraction_id


class JudgmentStore:
    def __init__(self, db: Database, models: list[str]) -> None:
        self.db = db
        self.models = models

    def find(self, kind: str, input_hash: str) -> Any | None:
        rows = self.db.query(
            f"SELECT output FROM judgments WHERE kind = ? AND input_hash = ?"
            f" AND model IN ({','.join('?' * len(self.models))}) ORDER BY created_at DESC LIMIT 1",
            (kind, input_hash, *self.models),
        )
        count(kind, reused=len(rows[:1]), asked=1 - len(rows[:1]))
        return json.loads(rows[0]["output"]) if rows else None

    def add(self, kind: str, input_hash: str, model: str, output: Any) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO judgments (kind, input_hash, model, output, created_at) VALUES (?, ?, ?, ?, ?)",
            (kind, input_hash, model, json.dumps(output), utc_now()),
        )


def kept_answer[T: BaseModel](llm: LLM, judgments: JudgmentStore, kind: str, role: Role, prompt: Prompt, output: type[T]) -> T:
    """The role's reply to ``prompt``, kept: the same prompt gets the same reply again."""
    input_hash = digest(prompt.id, prompt.system, prompt.user)
    kept = judgments.find(kind, input_hash)
    if kept is not None:
        return output.model_validate(kept)
    answer = llm.answer(role, prompt, output)
    judgments.add(kind, input_hash, answer.model, answer.output.model_dump())
    return answer.output
