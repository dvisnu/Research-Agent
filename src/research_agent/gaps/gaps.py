"""Research gaps: candidates from the graph, ranked by code, written up by the reasoning model.

Three kinds of candidate, each resting on evidence sentences already in the graph:
- stated: a limitation or future-work item a paper reports;
- structural: an empty cell of the landscape grid (a core method never reported
  on a core dataset) that the reasoning model judges worth studying;
- conflict: two findings from different papers that contradict each other in
  the same setting.

Ranking is code, so it can be reproduced: relevance to the question first, in
tiers of half a standard deviation of text-embedding similarity (Gemini, kept
in the embedding cache; SPECTER2 when any vector is missing), then evidence
strength (how many independent papers the evidence comes from), then raw
relevance. Before ranking, candidates of one kind that say the same thing are
merged by text similarity, so a gap raised by several papers counts them all. The reasoning model then writes each shortlisted candidate up as a
gap with a one-line warrant linking it to its evidence (a Toulmin
claim-grounds-warrant record, as in GAPMAP), drops the ones that aren't
research gaps, and folds duplicates into the earlier candidate. The kept gaps
are ranked again (a merged gap has more papers) and the top TOP_GAPS stored as
graph nodes, with their evidence and edges to what they rest on.
"""

import json
import math
import sqlite3
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
from pydantic import BaseModel, Field

from research_agent.graph.store import JudgmentStore, digest
from research_agent.literature.embed import Embedder
from research_agent.llm import LLM, LLMOutputError, Prompt, Role
from research_agent.storage import Database, new_id, parse_sentence_id, utc_now

TOP_GAPS = 5
SHORTLIST = 15  # candidates the reasoning model writes up
MAX_STRUCTURAL = 5  # empty cells it may pick
TIER = 0.5  # relevance tiers, in standard deviations
# Text-embedding similarity at which two candidates count as one gap. Within one topic,
# distinct gaps reach 0.90-0.93 (measured on the dev topics); clear paraphrases about 0.95.
DUPLICATE_SIMILARITY = 0.94
EVIDENCE_PER_SIDE = 2  # evidence sentences for a structural gap's method, and for its dataset


@dataclass
class Candidate:
    kind: str  # stated, structural, conflict
    text: str
    evidence: list[dict[str, str]]  # {"sentence", "quote"}
    sources: list[str]  # graph nodes it rests on
    detail: str = ""  # limitation / future_work, or why an empty cell matters
    relevance: float = 0.0
    tier: int = 0
    papers: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.papers |= {parse_sentence_id(e["sentence"])[0] for e in self.evidence}

    def key(self) -> tuple[int, int, float]:
        return (-self.tier, -len(self.papers), -self.relevance)


def _evidence(db: Database, kind: str, owner: str) -> list[dict[str, str]]:
    rows = db.query(
        "SELECT sentence_id, quote FROM evidence WHERE owner_kind = ? AND owner_id = ? ORDER BY sentence_id",
        (kind, owner),
    )
    return [{"sentence": row["sentence_id"], "quote": row["quote"]} for row in rows]


def stated_candidates(graph: nx.MultiDiGraph, db: Database) -> list[Candidate]:
    return [
        Candidate("stated", data["name"], _evidence(db, "node", node), [node], detail=data["node_type"])
        for node, data in graph.nodes(data=True)
        if data["node_type"] in ("limitation", "future_work")
    ]


def conflict_candidates(graph: nx.MultiDiGraph, db: Database) -> list[Candidate]:
    """One candidate per contradiction. The two findings are named in text order, not edge
    direction (which follows random node IDs), so a rebuilt graph gives the same candidate."""
    candidates = []
    for a, b, key, d in graph.edges(keys=True, data=True):
        if d["edge_type"] != "contradicts":
            continue
        first, second = sorted((a, b), key=lambda node: graph.nodes[node]["name"])
        candidates.append(Candidate(
            "conflict", f'"{graph.nodes[first]["name"]}" conflicts with "{graph.nodes[second]["name"]}"',
            _evidence(db, "edge", key), [first, second], detail=d["attrs"].get("reason", ""),
        ))
    return sorted(candidates, key=lambda c: c.text)


# Structural gaps: the reasoning model picks the empty cells worth studying.

class CellPick(BaseModel):
    pair: int = Field(description="The pair's number")
    reason: str = Field(description="One sentence: why testing this method on this dataset would be useful")


class CellPicks(BaseModel):
    """The reasoning model's reply."""

    pairs: list[CellPick] = Field(description=f"At most {MAX_STRUCTURAL} pairs, most worthwhile first; empty if none")


_CELLS_SYSTEM = f"""You look for research gaps in computer science and machine learning.

You are given a research question, the methods and the datasets most studied in its literature (with the tasks their papers work on), and the method-dataset pairs that no paper in this literature reports. Pick at most {MAX_STRUCTURAL} pairs worth studying: the method is meant for a task the dataset tests, and trying it there would tell us something useful for the research question. Skip pairs where the method does not apply to the dataset's task, and pairs that would be routine. Judge only from what is given."""


def _tasks(graph: nx.MultiDiGraph, node: str, edge_types: tuple[str, ...]) -> list[str]:
    """The tasks most often addressed by the papers linked to an entity."""
    counts: dict[str, int] = {}
    for paper, _, d in graph.in_edges(node, data=True):
        if d["edge_type"] in edge_types and graph.nodes[paper]["node_type"] == "paper":
            for _, task, e in graph.out_edges(paper, data=True):
                if e["edge_type"] == "addresses":
                    counts[graph.nodes[task]["name"]] = counts.get(graph.nodes[task]["name"], 0) + 1
    return sorted(counts, key=lambda t: (-counts[t], t))[:3]


def cells_prompt(question: str, graph: nx.MultiDiGraph, landscape: dict[str, Any], empty: list[tuple[int, int]]) -> Prompt:
    lines = [f"Research question: {question}", "", "Methods:"]
    for i, method in enumerate(landscape["methods"], start=1):
        tasks = ", ".join(_tasks(graph, method["node_id"], ("proposes", "uses"))) or "not stated"
        lines.append(f"[M{i}] {method['name']} ({method['papers']} papers; tasks: {tasks})")
    lines += ["", "Datasets:"]
    for j, dataset in enumerate(landscape["datasets"], start=1):
        tasks = ", ".join(_tasks(graph, dataset["node_id"], ("evaluates_on",))) or "not stated"
        lines.append(f"[D{j}] {dataset['name']} ({dataset['papers']} papers; tasks: {tasks})")
    lines += ["", "Pairs no paper reports:"]
    lines += [f"[{n}] M{i + 1} x D{j + 1}" for n, (i, j) in enumerate(empty, start=1)]
    return Prompt("gap_cells", 1, _CELLS_SYSTEM, "\n".join(lines))


def structural_candidates(
    llm: LLM, judgments: JudgmentStore, db: Database, run_id: str, graph: nx.MultiDiGraph,
    landscape: dict[str, Any], question: str,
) -> tuple[list[Candidate], dict[str, Any]]:
    """Empty grid cells the reasoning model picks as worth studying. A pick is dropped when
    some corpus paper names both the method and the dataset: the graph may simply have missed
    that paper's result (extraction keeps at most 10 items a list), so it isn't a safe gap."""
    empty = [(i, j) for i, row in enumerate(landscape["cells"]) for j, cell in enumerate(row) if cell["status"] == "empty"]
    if not empty:
        return [], {"empty_cells": 0}
    prompt = cells_prompt(question, graph, landscape, empty)
    input_hash = digest(prompt.id, prompt.system, prompt.user)
    picks = judgments.find("gap_cells", input_hash)
    reused = picks is not None
    if picks is None:
        try:
            answer = llm.answer(Role.REASONING, prompt, CellPicks)
        except LLMOutputError as exc:
            return [], {"empty_cells": len(empty), "error": str(exc)}
        picks = answer.output.model_dump()["pairs"]
        judgments.add("gap_cells", input_hash, answer.model, picks)
    candidates = []
    co_mentioned: list[dict[str, Any]] = []
    for pick in picks[:MAX_STRUCTURAL]:
        if not 1 <= pick["pair"] <= len(empty):
            continue
        i, j = empty[pick["pair"] - 1]
        method, dataset = landscape["methods"][i], landscape["datasets"][j]
        both = papers_mentioning_both(db, run_id, _names(graph, method["node_id"]), _names(graph, dataset["node_id"]))
        if both:  # the graph lacks the link, but a paper may well test the pair: not a safe gap
            co_mentioned.append({"method": method["name"], "dataset": dataset["name"], "papers": both[:3]})
            continue
        evidence = _side_evidence(db, graph, method["node_id"], ("proposes", "uses")) + _side_evidence(
            db, graph, dataset["node_id"], ("evaluates_on",))
        candidates.append(Candidate(
            "structural", f"No paper here reports {method['name']} on {dataset['name']}.", evidence,
            [method["node_id"], dataset["node_id"]], detail=pick["reason"],
        ))
    return candidates, {"empty_cells": len(empty), "picked": len(candidates), "reused": reused,
                        "dropped_co_mentioned": co_mentioned}


MIN_NAME_CHARS = 3  # shorter names (e.g. "QA") match too much text to count


def _names(graph: nx.MultiDiGraph, node: str) -> set[str]:
    data = graph.nodes[node]
    return {name for name in (data["name"], *data["aliases"]) if len(name.strip()) >= MIN_NAME_CHARS}


def papers_mentioning_both(db: Database, run_id: str, first: set[str], second: set[str]) -> list[str]:
    """Corpus papers of the run whose text names something from each set (case-insensitive)."""
    def papers(names: set[str]) -> set[str]:
        if not names:
            return set()
        rows = db.query(
            "SELECT DISTINCT c.paper_id FROM sentences t JOIN sections c ON c.section_id = t.section_id"
            " JOIN run_papers r ON r.paper_id = c.paper_id AND r.run_id = ? AND r.read_level IN ('full', 'abstract')"
            f" WHERE {' OR '.join('t.text LIKE ?' for _ in names)}",
            (run_id, *(f"%{name}%" for name in sorted(names))),
        )
        return {row["paper_id"] for row in rows}

    return sorted(papers(first) & papers(second))


def _side_evidence(db: Database, graph: nx.MultiDiGraph, node: str, edge_types: tuple[str, ...]) -> list[dict[str, str]]:
    """Sentences from up to EVIDENCE_PER_SIDE different papers that use the entity."""
    evidence = []
    for _, _, key, d in graph.in_edges(node, keys=True, data=True):
        if d["edge_type"] in edge_types and len(evidence) < EVIDENCE_PER_SIDE:
            evidence += _evidence(db, "edge", key)[:1]
    return evidence


# Ranking.

def score_relevance(
    candidates: list[Candidate], embedder: Embedder, question: str, vectors: dict[str, np.ndarray] | None = None
) -> str:
    """Each candidate's similarity to the question, and its relevance tier. Uses the text
    embedding ``vectors`` (keyed by text) when they cover the question and every candidate,
    and SPECTER2 otherwise, so the two models' scores never mix. Returns which was used."""
    if not candidates:
        return "none"
    texts = [c.text for c in candidates]
    if vectors and question in vectors and all(text in vectors for text in texts):
        relevance = np.stack([vectors[text] for text in texts]) @ vectors[question]
        used = "text embeddings"
    else:
        relevance = embedder.papers([(text, None) for text in texts]) @ embedder.query(question)
        used = "specter2"
    spread = relevance.std()
    z = (relevance - relevance.mean()) / spread if spread else np.zeros_like(relevance)
    for candidate, value, standard in zip(candidates, relevance, z):
        candidate.relevance = float(value)
        candidate.tier = math.floor(standard / TIER)
    return used


def merge_duplicates(candidates: list[Candidate], vectors: dict[str, np.ndarray]) -> tuple[list[Candidate], int]:
    """Fold candidates of the same kind that say the same thing (text similarity of at least
    DUPLICATE_SIMILARITY) into the most relevant of them, which takes their evidence and
    papers. Each candidate is compared with the kept ones only, so merges never chain.
    Candidates without a vector are kept as they are. Returns the candidates and the merge count."""
    kept: list[Candidate] = []
    merged = 0
    for candidate in sorted(candidates, key=lambda c: -c.relevance):
        vector = vectors.get(candidate.text)
        into = next(
            (k for k in kept if vector is not None and k.kind == candidate.kind and k.text in vectors
             and float(vectors[k.text] @ vector) >= DUPLICATE_SIMILARITY),
            None,
        )
        if into is None:
            kept.append(candidate)
            continue
        into.evidence += [e for e in candidate.evidence if e not in into.evidence]
        into.sources += [s for s in candidate.sources if s not in into.sources]
        into.papers |= candidate.papers
        merged += 1
    return kept, merged


def rank(candidates: list[Candidate]) -> list[Candidate]:
    return sorted(candidates, key=Candidate.key)


# The write-up.

class GapWriteup(BaseModel):
    candidate: int = Field(description="The candidate's number")
    gap: str = Field(description="One sentence: what is not yet known, solved or tested")
    warrant: str = Field(description="One sentence: how the evidence shows this gap")
    keep: bool = Field(description="False if this is not a research gap worth pursuing, or is a duplicate")
    duplicate_of: int | None = Field(description="The number of an earlier candidate saying the same thing, or null")


class GapWriteups(BaseModel):
    """The reasoning model's reply."""

    gaps: list[GapWriteup] = Field(description="One entry per candidate, in list order")


_WRITEUP_SYSTEM = """You write up research gaps for a literature review in computer science and machine learning.

You are given a research question and numbered gap candidates found in its literature. Each has a kind (stated: a limitation or future-work item a paper reports; structural: a method never reported on a dataset; conflict: findings that disagree) and the evidence sentences it rests on, quoted from the papers. For each candidate:
- gap: one sentence saying what is not yet known, solved or tested, in the field's terms and as specific as the evidence allows. Do not claim more than the evidence shows.
- warrant: one sentence saying how the evidence shows the gap, naming what the papers report.
- keep: false if it is not a research gap worth pursuing: an implementation detail, a limitation of one experiment's setup, something the evidence does not support, a problem the other candidates' evidence shows newer work has since addressed (each quote gives its paper's year), or a restatement of the research question's broad problem rather than a specific open issue within it. Also false if it repeats an earlier candidate; then give that candidate's number as duplicate_of.
Judge only from the evidence given."""


def writeup_prompt(question: str, shortlist: list[Candidate], titles: dict[str, str]) -> Prompt:
    """``titles`` names each paper as the model should see it, e.g. "Title, 2021"."""
    lines = [f"Research question: {question}"]
    for number, candidate in enumerate(shortlist, start=1):
        kind = f"{candidate.kind} ({candidate.detail})" if candidate.kind == "stated" else candidate.kind
        lines += ["", f"[{number}] {kind}: {candidate.text}"]
        if candidate.kind == "structural" and candidate.detail:
            lines.append(f"Why it may matter: {candidate.detail}")
        for evidence in candidate.evidence[:4]:
            paper = titles.get(parse_sentence_id(evidence["sentence"])[0], "a paper")
            lines.append(f'- "{evidence["quote"]}" ({paper})')
    return Prompt("gap_writeup", 2, _WRITEUP_SYSTEM, "\n".join(lines))


@dataclass
class Gap:
    candidate: Candidate
    gap: str
    warrant: str | None


def write_up(
    llm: LLM, judgments: JudgmentStore, question: str, shortlist: list[Candidate], titles: dict[str, str]
) -> tuple[list[Gap], dict[str, Any]]:
    """The kept gaps, duplicates folded into the earlier candidate, ranked again."""
    if not shortlist:
        return [], {"shortlist": 0}
    prompt = writeup_prompt(question, shortlist, titles)
    input_hash = digest(prompt.id, prompt.system, prompt.user)
    replies = judgments.find("gap_writeup", input_hash)
    stats: dict[str, Any] = {"shortlist": len(shortlist), "reused": replies is not None}
    if replies is None:
        try:
            answer = llm.answer(Role.REASONING, prompt, GapWriteups)
        except LLMOutputError as exc:  # keep the code ranking, without warrants
            stats["error"] = str(exc)
            return [Gap(c, c.text, None) for c in shortlist], stats
        replies = answer.output.model_dump()["gaps"]
        judgments.add("gap_writeup", input_hash, answer.model, replies)
    by_number = {reply["candidate"]: reply for reply in replies}
    kept: dict[int, Gap] = {}
    dropped = merged = 0
    for number, candidate in enumerate(shortlist, start=1):
        reply = by_number.get(number)
        if reply is None:
            kept[number] = Gap(candidate, candidate.text, None)
            continue
        target = reply.get("duplicate_of")
        if target in kept and target != number:
            into = kept[target].candidate
            into.evidence += [e for e in candidate.evidence if e not in into.evidence]
            into.sources += [s for s in candidate.sources if s not in into.sources]
            into.papers |= candidate.papers
            merged += 1
        elif reply["keep"]:
            kept[number] = Gap(candidate, reply["gap"], reply["warrant"])
        else:
            dropped += 1
    stats.update(kept=len(kept), dropped=dropped, merged=merged)
    return sorted(kept.values(), key=lambda g: g.candidate.key()), stats


# Storage: gaps are graph nodes of the run.

def store_gaps(db: Database, run_id: str, gaps: list[Gap]) -> list[str]:
    now = utc_now()
    ids = []
    with db.transaction() as conn:
        _clear_gaps(conn, run_id)
        for rank_, gap in enumerate(gaps, start=1):
            node_id = new_id("n")
            ids.append(node_id)
            c = gap.candidate
            attrs = {"rank": rank_, "kind": c.kind, "warrant": gap.warrant, "candidate": c.text, "detail": c.detail,
                     "relevance": round(c.relevance, 4), "tier": c.tier, "papers": sorted(c.papers)}
            conn.execute(
                "INSERT INTO nodes (node_id, node_type, name, attrs, created_at, run_id) VALUES (?, 'gap', ?, ?, ?, ?)",
                (node_id, gap.gap, json.dumps(attrs, ensure_ascii=False), now, run_id),
            )
            conn.executemany(
                "INSERT OR IGNORE INTO evidence (owner_kind, owner_id, sentence_id, quote) VALUES ('node', ?, ?, ?)",
                [(node_id, e["sentence"], e["quote"]) for e in c.evidence],
            )
            conn.executemany(
                "INSERT INTO edges (edge_id, edge_type, src_id, dst_id, attrs, created_at, run_id)"
                " VALUES (?, 'rests_on', ?, ?, '{}', ?, ?)",
                [(new_id("e"), node_id, source, now, run_id) for source in c.sources],
            )
    return ids


def _clear_gaps(conn: sqlite3.Connection, run_id: str) -> None:
    gaps = "SELECT node_id FROM nodes WHERE run_id = ? AND node_type = 'gap'"
    conn.execute(f"DELETE FROM evidence WHERE owner_kind = 'node' AND owner_id IN ({gaps})", (run_id,))
    conn.execute(f"DELETE FROM nodes WHERE node_id IN ({gaps})", (run_id,))  # their edges go with them

