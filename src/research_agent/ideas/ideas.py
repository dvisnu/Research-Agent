"""Ideas: written from the top gaps by the reasoning model, stored with their facets.

An idea has the three facets the novelty check compares (purpose: the problem
it addresses; mechanism: how it works; evaluation: how it is tested), the gap
it answers, and the kind of move it makes: fixing a limitation the papers
report, combining methods, or borrowing a technique from another area.

The reasoning model writes IDEAS_WRITTEN ideas in one call; code drops those
naming no real gap and near-duplicates (of each other and of ideas rejected
earlier in the run), then picks IDEAS_CHECKED for the novelty check, one per
gap in gap order first. An idea rated 1-2 is revised: the model changes exactly
one facet, code keeps the other two from the previous version, and that version
is rejected and remembered so it isn't proposed again.
"""

import json
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from research_agent.embeddings import EmbeddingCache, TextEmbedder
from research_agent.graph.store import JudgmentStore, digest, kept_answer
from research_agent.llm import LLM, LLMOutputError, Prompt, Role
from research_agent.storage import Database, new_id, utc_now

IDEAS_WRITTEN = 6
IDEAS_CHECKED = 3
MAX_PER_GAP = 2
# Text-embedding similarity at which two ideas count as one, as for gaps: on the RAG dev
# topic distinct ideas reach 0.91 as whole texts.
IDEA_DUPLICATE = 0.94
# The same for a revised facet against earlier versions of its idea. A revision keeps the
# sentence and swaps the technique, so on 5 dev topics a different technique scored 0.87-0.97
# against the one it replaced; only a near-verbatim repeat (0.985 seen) is rejected unchecked,
# and the novelty check judges the rest.
REPEAT_IN_LINEAGE = 0.98
FACETS = ("purpose", "mechanism", "evaluation")
KINDS = {
    "fix_limitation": "fixes a limitation the papers report",
    "combine_methods": "combines methods in a way the papers have not",
    "borrow_technique": "borrows a technique from another area",
}


@dataclass
class Idea:
    idea_id: str
    run_id: str
    lineage: int
    round: int
    origin: str  # generated, revised, user, planted
    title: str
    purpose: str
    mechanism: str
    evaluation: str
    status: str = "pending"
    parent_id: str | None = None
    gap_node_id: str | None = None
    kind: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    level: int | None = None
    assessment: dict[str, Any] | None = None
    final_rank: int | None = None

    def text(self) -> str:
        """The idea as one passage: what is embedded, searched and shown to models."""
        return f"{self.title}. {self.purpose} {self.mechanism} {self.evaluation}"

    def block(self) -> list[str]:
        return [f"Title: {self.title}", f"Purpose: {self.purpose}", f"Mechanism: {self.mechanism}",
                f"Evaluation: {self.evaluation}"]

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Idea":
        return cls(
            idea_id=row["idea_id"], run_id=row["run_id"], lineage=row["lineage"], round=row["round"],
            origin=row["origin"], title=row["title"], purpose=row["purpose"], mechanism=row["mechanism"],
            evaluation=row["evaluation"], status=row["status"], parent_id=row["parent_id"],
            gap_node_id=row["gap_node_id"], kind=row["kind"], detail=json.loads(row["detail"]),
            level=row["level"], assessment=json.loads(row["assessment"]) if row["assessment"] else None,
            final_rank=row["final_rank"],
        )

    def summary(self) -> dict[str, Any]:
        """What the pipeline state keeps about an idea."""
        return {"idea_id": self.idea_id, "lineage": self.lineage, "round": self.round, "origin": self.origin,
                "title": self.title, "status": self.status, "level": self.level, "final_rank": self.final_rank}


class IdeaStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    def add(self, idea: Idea) -> Idea:
        self.db.execute(
            "INSERT INTO ideas (idea_id, run_id, lineage, round, parent_id, origin, gap_node_id, kind, title,"
            " purpose, mechanism, evaluation, detail, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (idea.idea_id, idea.run_id, idea.lineage, idea.round, idea.parent_id, idea.origin, idea.gap_node_id,
             idea.kind, idea.title, idea.purpose, idea.mechanism, idea.evaluation,
             json.dumps(idea.detail, ensure_ascii=False), idea.status, utc_now()),
        )
        return idea

    def get(self, idea_id: str) -> Idea:
        return Idea.from_row(self.db.query("SELECT * FROM ideas WHERE idea_id = ?", (idea_id,))[0])

    def for_run(self, run_id: str, *statuses: str) -> list[Idea]:
        sql, params = "SELECT * FROM ideas WHERE run_id = ?", (run_id,)
        if statuses:
            sql += f" AND status IN ({','.join('?' * len(statuses))})"
            params += statuses
        return [Idea.from_row(row) for row in self.db.query(sql + " ORDER BY lineage, round", params)]

    def set_status(self, idea_id: str, status: str) -> None:
        self.db.execute("UPDATE ideas SET status = ? WHERE idea_id = ?", (status, idea_id))

    def set_detail(self, idea: Idea) -> None:
        self.db.execute("UPDATE ideas SET detail = ? WHERE idea_id = ?",
                        (json.dumps(idea.detail, ensure_ascii=False), idea.idea_id))

    def set_assessment(self, idea_id: str, level: int, assessment: dict[str, Any]) -> None:
        self.db.execute(
            "UPDATE ideas SET status = 'assessed', level = ?, assessment = ? WHERE idea_id = ?",
            (level, json.dumps(assessment, ensure_ascii=False, default=str), idea_id),
        )

    def set_final_ranks(self, run_id: str, ranked: list[str]) -> None:
        with self.db.transaction() as conn:
            conn.execute("UPDATE ideas SET final_rank = NULL WHERE run_id = ?", (run_id,))
            conn.executemany("UPDATE ideas SET final_rank = ? WHERE idea_id = ?",
                             [(rank, idea_id) for rank, idea_id in enumerate(ranked, start=1)])

    def clear_run(self, run_id: str) -> None:
        """Forget a run's ideas, so Phase 4 can run on it again. Comparisons, candidates and
        plans go with their ideas; the comparisons' and plan elements' evidence is removed here."""
        with self.db.transaction() as conn:
            conn.execute(
                "DELETE FROM evidence WHERE owner_kind = 'plan_element' AND owner_id IN"
                " (SELECT e.element_id FROM plan_elements e JOIN plans p USING (plan_id) WHERE p.run_id = ?)",
                (run_id,),
            )
            conn.execute("DELETE FROM plans WHERE run_id = ?", (run_id,))
            conn.execute(
                "DELETE FROM evidence WHERE owner_kind = 'comparison' AND owner_id IN"
                " (SELECT c.comparison_id FROM comparisons c JOIN ideas i USING (idea_id) WHERE i.run_id = ?)",
                (run_id,),
            )
            conn.execute("UPDATE ideas SET parent_id = NULL WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM ideas WHERE run_id = ?", (run_id,))


# The gaps an idea can answer -------------------------------------------------------

def load_gaps(db: Database, run_id: str) -> list[dict[str, Any]]:
    """The run's stored gaps, best first, with their evidence and its papers' years."""
    gaps = []
    for row in db.query("SELECT node_id, name, attrs FROM nodes WHERE run_id = ? AND node_type = 'gap'", (run_id,)):
        attrs = json.loads(row["attrs"])
        evidence = db.query(
            "SELECT e.sentence_id, e.quote, p.paper_id, p.title, p.year FROM evidence e JOIN sentences s USING (sentence_id)"
            " JOIN sections c ON c.section_id = s.section_id JOIN papers p ON p.paper_id = c.paper_id"
            " WHERE e.owner_kind = 'node' AND e.owner_id = ? ORDER BY e.sentence_id",
            (row["node_id"],),
        )
        gaps.append({
            "node_id": row["node_id"], "rank": attrs["rank"], "kind": attrs["kind"], "gap": row["name"],
            "warrant": attrs.get("warrant"), "papers": len(attrs["papers"]),
            "evidence": [{"sentence_id": e["sentence_id"], "quote": e["quote"], "paper_id": e["paper_id"],
                          "paper": e["title"], "year": e["year"]} for e in evidence],
        })
    return sorted(gaps, key=lambda gap: gap["rank"])


def _paper_label(title: str, year: int | None) -> str:
    return f"{title}, {year}" if year else title


# Writing ideas ------------------------------------------------------------------------

class IdeaDraft(BaseModel):
    gap: int = Field(description="The number of the gap it answers")
    kind: Literal["fix_limitation", "combine_methods", "borrow_technique"]
    title: str = Field(description="At most 15 words")
    purpose: str = Field(description="One sentence: the problem it addresses")
    mechanism: str = Field(description="One or two sentences: the technique and how it works")
    evaluation: str = Field(description="One sentence: datasets or benchmarks, metrics and baselines")
    builds_on: list[str] = Field(description="Methods, datasets or papers from the context that it builds on")
    feasibility: str = Field(description="One sentence: why an academic lab could do it within a few months")


class IdeaDrafts(BaseModel):
    """The reasoning model's reply."""

    ideas: list[IdeaDraft] = Field(description=f"{IDEAS_WRITTEN} ideas")


_IDEAS_SYSTEM = f"""You propose research ideas in computer science and machine learning.

You are given a research question, the research gaps found in its literature (each with evidence quoted from the papers, with their years) and the closest existing work. Propose {IDEAS_WRITTEN} research ideas, each answering one of the gaps by one of these moves:
- fix_limitation: overcome a limitation or open problem the papers report;
- combine_methods: combine methods or findings from this literature in a way the papers have not;
- borrow_technique: bring in a technique from another area of computer science or machine learning that suits the problem.

Each idea must be:
- specific: name the technique and say how it works, not just a direction such as "use better retrieval";
- feasible for an academic lab within a few months: public datasets and benchmarks, open models or affordable APIs, modest compute;
- testable: name the datasets or benchmarks, metrics and baselines, preferably ones these papers use;
- new as far as you can tell: not what the closest work or the gap evidence already describes, and not one of the rejected ideas listed or a close variant of one.

Spread the ideas over the gaps, at most {MAX_PER_GAP} per gap, best gaps first. Write each facet in plain technical language without hype."""


def ideas_prompt(question: str, gaps: list[dict[str, Any]], related: list[dict[str, Any]], rejected: list[str]) -> Prompt:
    lines = [f"Research question: {question}", "", "Research gaps:"]
    for number, gap in enumerate(gaps, start=1):
        lines += ["", f"[{number}] ({gap['kind']}) {gap['gap']}"]
        if gap.get("warrant"):
            lines.append(f"Warrant: {gap['warrant']}")
        lines += [f'- "{e["quote"]}" ({_paper_label(e["paper"], e["year"])})' for e in gap["evidence"][:4]]
    if related:
        lines += ["", "Closest existing work:"]
        for paper in related:
            purpose = f": {paper['purpose']}" if paper.get("purpose") else ""
            lines.append(f"- {_paper_label(paper['title'], paper.get('year'))}{purpose}")
    if rejected:
        lines += ["", "Rejected ideas (too close to existing work; do not propose these or close variants):"]
        lines += [f"- {text}" for text in rejected]
    return Prompt("idea_writeup", 1, _IDEAS_SYSTEM, "\n".join(lines))


def write_ideas(
    llm: LLM, judgments: JudgmentStore, question: str, gaps: list[dict[str, Any]], related: list[dict[str, Any]],
    rejected: list[str],
) -> tuple[list[IdeaDraft], dict[str, Any]]:
    prompt = ideas_prompt(question, gaps, related, rejected)
    input_hash = digest(prompt.id, prompt.system, prompt.user)
    kept = judgments.find("idea_writeup", input_hash)
    stats: dict[str, Any] = {"reused": kept is not None}
    if kept is None:
        try:
            answer = llm.answer(Role.REASONING, prompt, IdeaDrafts)
        except LLMOutputError as exc:
            return [], {**stats, "error": str(exc)}
        kept = answer.output.model_dump()["ideas"]
        judgments.add("idea_writeup", input_hash, answer.model, kept)
    drafts = [IdeaDraft.model_validate(item) for item in kept]
    return drafts, {**stats, "written": len(drafts)}


def _normalised(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def duplicates(
    db: Database, text_embedder: TextEmbedder | None, texts: list[str], against: list[str],
    threshold: float = IDEA_DUPLICATE,
) -> tuple[list[int | None], dict[str, Any]]:
    """For each text, the index (in ``against`` + the texts before it) of an earlier text
    saying the same thing, or None. Text-embedding similarity of at least ``threshold``,
    or the same words when no embedding model is set or a vector is missing."""
    pool = [*against, *texts]
    vectors: dict[str, Any] = {}
    report: dict[str, Any] = {"status": "no embedding model set (RA_EMBEDDING): exact wording only"}
    if text_embedder is not None and texts:
        vectors, report = EmbeddingCache(db, text_embedder).vectors(pool)
    found: list[int | None] = []
    for i, text in enumerate(texts):
        position = len(against) + i
        match = None
        for j in range(position):
            other = pool[j]
            if j >= len(against) and found[j - len(against)] is not None:
                continue  # compare with kept texts only, so matches never chain
            if text in vectors and other in vectors:
                if float(vectors[text] @ vectors[other]) >= threshold:
                    match = j
                    break
            elif _normalised(text) == _normalised(other):
                match = j
                break
        found.append(match)
    return found, report


def select(drafts: list[IdeaDraft], gap_count: int, duplicate_of: list[int | None]) -> tuple[list[int], list[dict[str, Any]]]:
    """Indexes of the drafts to check: valid and not duplicates, one per gap in gap order,
    then a second per gap, up to IDEAS_CHECKED. Also the drafts dropped, with the reason."""
    dropped: list[dict[str, Any]] = []
    valid: list[int] = []
    for i, draft in enumerate(drafts):
        if not 1 <= draft.gap <= gap_count:
            dropped.append({"draft": i, "reason": f"names gap {draft.gap}, which does not exist"})
        elif duplicate_of[i] is not None:
            dropped.append({"draft": i, "reason": "repeats another or a rejected idea"})
        else:
            valid.append(i)
    chosen: list[int] = []
    for turn in range(MAX_PER_GAP):
        for gap in range(1, gap_count + 1):
            per_gap = [i for i in valid if drafts[i].gap == gap]
            if len(per_gap) > turn and len(chosen) < IDEAS_CHECKED:
                chosen.append(per_gap[turn])
    return chosen, dropped


# Revising an idea that is too close to existing work ------------------------------------

class Revision(BaseModel):
    idea: int = Field(description="The idea's number")
    changed: Literal["purpose", "mechanism", "evaluation"] = Field(description="The one facet changed")
    new_text: str = Field(description="The changed facet's new text, in the same style as before")
    title: str = Field(description="The revised idea's title, at most 15 words")
    why: str = Field(description="One sentence: how the changed facet now differs from every paper listed")
    feasibility: str = Field(description="One sentence: why an academic lab could still do it within a few months")


class Revisions(BaseModel):
    """The reasoning model's reply."""

    ideas: list[Revision] = Field(description="One revision per idea, in list order")


_REVISE_SYSTEM = """You revise research ideas in computer science and machine learning that turned out to be too close to existing work.

Each idea is given with its facets (purpose: the problem it addresses; mechanism: how it works; evaluation: how it is tested), the research gap it answers, and the existing papers closest to it, with each paper's own purpose, mechanism and evaluation and how they match the idea's. For each idea, change exactly one facet so that the idea differs from every paper listed in that facet, while still answering the same gap and staying specific, feasible and testable. Prefer changing the mechanism: a different technique for the same problem. The other two facets stay as they are. Do not propose any of the rejected ideas listed, or a close variant of one."""


def revise_prompt(items: list[dict[str, Any]], rejected: list[str]) -> Prompt:
    """``items``: {"idea": Idea, "gap": str | None, "papers": [{"title", "year", facet: {"paper", "match"}}]}."""
    lines: list[str] = []
    for number, item in enumerate(items, start=1):
        idea: Idea = item["idea"]
        if lines:
            lines.append("")
        lines += [f"[{number}]", *idea.block(), f"Gap it answers: {item['gap'] or 'none given'}", "Closest papers:"]
        for paper in item["papers"]:
            lines.append(f"- {_paper_label(paper['title'], paper.get('year'))}")
            for facet in FACETS:
                verdict = paper[facet]
                lines.append(f"  {facet} ({verdict['match'].replace('_', ' ')}): {verdict['paper'] or 'not stated'}")
    if rejected:
        lines += ["", "Rejected ideas (do not propose these or close variants):", *(f"- {text}" for text in rejected)]
    return Prompt("idea_revision", 1, _REVISE_SYSTEM, "\n".join(lines))


def revise_ideas(
    llm: LLM, judgments: JudgmentStore, items: list[dict[str, Any]], rejected: list[str]
) -> tuple[dict[int, Revision], dict[str, Any]]:
    """Revisions by item number (from 1)."""
    if not items:
        return {}, {"revised": 0}
    prompt = revise_prompt(items, rejected)
    input_hash = digest(prompt.id, prompt.system, prompt.user)
    kept = judgments.find("idea_revision", input_hash)
    stats: dict[str, Any] = {"reused": kept is not None}
    if kept is None:
        try:
            answer = llm.answer(Role.REASONING, prompt, Revisions)
        except LLMOutputError as exc:
            return {}, {**stats, "error": str(exc)}
        kept = answer.output.model_dump()["ideas"]
        judgments.add("idea_revision", input_hash, answer.model, kept)
    revisions = {r.idea: r for r in (Revision.model_validate(item) for item in kept) if 1 <= r.idea <= len(items)}
    return revisions, {**stats, "revised": len(revisions)}


def revised(parent: Idea, revision: Revision, round_: int) -> Idea:
    """The next version of ``parent``: the changed facet from the revision, the others kept."""
    facets = {facet: getattr(parent, facet) for facet in FACETS}
    facets[revision.changed] = revision.new_text
    return Idea(
        idea_id=new_id("i"), run_id=parent.run_id, lineage=parent.lineage, round=round_, origin="revised",
        title=revision.title, status="pending", parent_id=parent.idea_id, gap_node_id=parent.gap_node_id,
        kind=parent.kind,
        detail={**{k: v for k, v in parent.detail.items() if k in ("gap", "builds_on")},
                "feasibility": revision.feasibility, "changed": revision.changed, "why": revision.why},
        **facets,
    )


# A user's own idea, from free text -----------------------------------------------------

class IdeaFacets(BaseModel):
    """The extraction model's reply."""

    title: str = Field(description="A short title for the idea, at most 15 words")
    purpose: str = Field(description="One sentence: the problem it addresses")
    mechanism: str = Field(description="One or two sentences: the technique and how it works, as far as stated")
    evaluation: str = Field(description="One sentence: how it would be tested, as far as stated or implied")


_FACETS_SYSTEM = """You restate a research idea in computer science or machine learning as its facets: a short title, its purpose (the problem it addresses), its mechanism (how it works) and its evaluation (how it would be tested). Keep to what the text says or clearly implies; do not add techniques it does not mention. Where it says nothing about a facet, write what follows most directly from the rest, briefly."""


def idea_facets(llm: LLM, judgments: JudgmentStore, text: str) -> IdeaFacets:
    """The idea's facets; the same text gets the same facets again."""
    prompt = Prompt("idea_facets", 1, _FACETS_SYSTEM, f"Idea:\n{text}")
    return kept_answer(llm, judgments, "idea_facets", Role.EXTRACTION, prompt, IdeaFacets)
