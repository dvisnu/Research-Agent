"""Plans and their elements in SQLite.

Each element keeps three provenance records: its source (the evidence rows: sentences
and the quotes the cited facts rest on), its derivation (the step, round, model and
prompt that wrote it, and the origin of every fact it cites) and its decision (why it
was chosen, over which alternatives, and which critique points it answers), plus the
verifier's verdict on it.
"""

import json
from dataclasses import dataclass, field
from typing import Any

from research_agent.storage import Database, new_id, utc_now

FIELDS = ("hypothesis", "research_question", "method", "datasets", "baselines", "metrics", "conditions",
          "ablations", "expected_results", "risks")
SINGLE = ("hypothesis", "research_question")  # one element each; the rest are lists
ENTITY_FIELDS = {"datasets": "dataset", "baselines": "method", "metrics": "metric"}  # field -> the fact kind it lists


@dataclass
class Element:
    field: str
    position: int
    text: str
    name: str | None = None
    basis: str | None = None
    cites: list[str] = field(default_factory=list)
    evidence: list[dict[str, str]] = field(default_factory=list)  # {"sentence_id", "quote"}
    derivation: dict[str, Any] = field(default_factory=dict)
    decision: dict[str, Any] = field(default_factory=dict)
    verdict: str | None = None
    verification: dict[str, Any] | None = None
    status: str = "kept"
    element_id: str = field(default_factory=lambda: new_id("pe"))

    @property
    def label(self) -> str:
        return self.field if self.field in SINGLE else f"{self.field} {self.position}"


@dataclass
class Plan:
    plan_id: str
    run_id: str
    idea_id: str
    round: int
    status: str = "draft"
    menu: list[dict[str, Any]] = field(default_factory=list)
    critique: dict[str, Any] | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    elements: list[Element] = field(default_factory=list)

    def field_elements(self, name: str, *, removed: bool = False) -> list[Element]:
        return [e for e in self.elements if e.field == name and (removed or e.status != "removed")]


class PlanStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    def clear_run(self, run_id: str) -> None:
        """Forget a run's plans and plan papers, so Phase 5 can run on it again."""
        with self.db.transaction() as conn:
            conn.execute(
                "DELETE FROM evidence WHERE owner_kind = 'plan_element' AND owner_id IN"
                " (SELECT e.element_id FROM plan_elements e JOIN plans p USING (plan_id) WHERE p.run_id = ?)",
                (run_id,),
            )
            conn.execute("DELETE FROM plans WHERE run_id = ?", (run_id,))  # elements go with them
            conn.execute("DELETE FROM plan_papers WHERE run_id = ?", (run_id,))

    def add(self, plan: Plan) -> Plan:
        """Store a plan version, replacing one of the same round (a stage run again)."""
        now = utc_now()
        with self.db.transaction() as conn:
            old = conn.execute("SELECT plan_id FROM plans WHERE run_id = ? AND round = ?", (plan.run_id, plan.round)).fetchone()
            if old:
                conn.execute(
                    "DELETE FROM evidence WHERE owner_kind = 'plan_element' AND owner_id IN"
                    " (SELECT element_id FROM plan_elements WHERE plan_id = ?)", (old["plan_id"],))
                conn.execute("DELETE FROM plans WHERE plan_id = ?", (old["plan_id"],))
            conn.execute(
                "INSERT INTO plans (plan_id, run_id, idea_id, round, status, menu, critique, detail, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (plan.plan_id, plan.run_id, plan.idea_id, plan.round, plan.status, _json(plan.menu),
                 None if plan.critique is None else _json(plan.critique), _json(plan.detail), now),
            )
            for element in plan.elements:
                conn.execute(
                    "INSERT INTO plan_elements (element_id, plan_id, field, position, name, text, basis, cites,"
                    " derivation, decision, verdict, verification, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (element.element_id, plan.plan_id, element.field, element.position, element.name, element.text,
                     element.basis, _json(element.cites), _json(element.derivation), _json(element.decision),
                     element.verdict, None if element.verification is None else _json(element.verification),
                     element.status),
                )
                conn.executemany(
                    "INSERT OR IGNORE INTO evidence (owner_kind, owner_id, sentence_id, quote) VALUES ('plan_element', ?, ?, ?)",
                    [(element.element_id, e["sentence_id"], e["quote"]) for e in element.evidence],
                )
        return plan

    def versions(self, run_id: str) -> list[Plan]:
        return [self._plan(row) for row in self.db.query("SELECT * FROM plans WHERE run_id = ? ORDER BY round", (run_id,))]

    def latest(self, run_id: str) -> Plan | None:
        rows = self.db.query("SELECT * FROM plans WHERE run_id = ? ORDER BY round DESC LIMIT 1", (run_id,))
        return self._plan(rows[0]) if rows else None

    def _plan(self, row: Any) -> Plan:
        plan = Plan(row["plan_id"], row["run_id"], row["idea_id"], row["round"], row["status"], json.loads(row["menu"]),
                    json.loads(row["critique"]) if row["critique"] else None, json.loads(row["detail"]))
        evidence: dict[str, list[dict[str, str]]] = {}
        for e in self.db.query(
            "SELECT e.owner_id, e.sentence_id, e.quote FROM evidence e JOIN plan_elements p ON p.element_id = e.owner_id"
            " WHERE e.owner_kind = 'plan_element' AND p.plan_id = ? ORDER BY e.rowid", (plan.plan_id,),
        ):
            evidence.setdefault(e["owner_id"], []).append({"sentence_id": e["sentence_id"], "quote": e["quote"]})
        for e in self.db.query("SELECT * FROM plan_elements WHERE plan_id = ? ORDER BY rowid", (plan.plan_id,)):
            plan.elements.append(Element(
                field=e["field"], position=e["position"], text=e["text"], name=e["name"], basis=e["basis"],
                cites=json.loads(e["cites"]), evidence=evidence.get(e["element_id"], []),
                derivation=json.loads(e["derivation"]), decision=json.loads(e["decision"]), verdict=e["verdict"],
                verification=json.loads(e["verification"]) if e["verification"] else None, status=e["status"],
                element_id=e["element_id"],
            ))
        plan.elements.sort(key=lambda e: (FIELDS.index(e.field), e.position))
        return plan

    def set_critique(self, plan_id: str, critique: dict[str, Any]) -> None:
        self.db.execute("UPDATE plans SET critique = ? WHERE plan_id = ?", (_json(critique), plan_id))

    def set_status(self, plan_id: str, status: str) -> None:
        self.db.execute("UPDATE plans SET status = ? WHERE plan_id = ?", (status, plan_id))

    def set_detail(self, plan_id: str, detail: dict[str, Any]) -> None:
        self.db.execute("UPDATE plans SET detail = ? WHERE plan_id = ?", (_json(detail), plan_id))

    def set_verdicts(self, elements: list[Element]) -> None:
        with self.db.transaction() as conn:
            conn.executemany(
                "UPDATE plan_elements SET verdict = ?, verification = ?, status = ? WHERE element_id = ?",
                [(e.verdict, None if e.verification is None else _json(e.verification), e.status, e.element_id)
                 for e in elements],
            )


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)
