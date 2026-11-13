"""Cross-paper links: which findings support or contradict each other.

Candidate pairs come from two places, always from different papers:
- similar text: each finding's nearest neighbours by embedding similarity;
- shared setting: results on the same dataset with the same metric.
The extraction model then labels each pair supports, contradicts or unrelated,
and says whether the two are about the same setting. A contradiction counts
only in the same setting: "A beats B on X" and "B beats A on Y" don't conflict.
Labels are kept, so a pair is judged once.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from research_agent.graph.store import JudgmentStore, digest
from research_agent.llm import LLM, LLMOutputError, Prompt, Role

NEIGHBOURS = 3  # similar findings considered per finding
MIN_SIMILARITY = 0.80
MAX_SIMILAR_PAIRS = 150
MAX_SHARED_PAIRS = 100
PAIRS_PER_CALL = 10

Relation = Literal["supports", "contradicts", "unrelated"]


@dataclass
class Finding:
    node_id: str
    paper_id: str
    paper_title: str
    kind: str  # result or claim
    text: str
    quote: str  # its first piece of evidence
    dataset: str | None = None  # merged node IDs, for results
    metric: str | None = None
    conditions: str | None = None


@dataclass
class Pair:
    a: Finding
    b: Finding
    similarity: float | None
    found_by: list[str] = field(default_factory=list)


def candidate_pairs(findings: list[Finding], vectors: dict[str, np.ndarray]) -> list[Pair]:
    """Pairs from different papers worth labelling, most promising first."""
    pairs: dict[tuple[str, str], Pair] = {}
    by_id = {f.node_id: f for f in findings}

    def add(a: Finding, b: Finding, similarity: float | None, source: str) -> None:
        key = (a.node_id, b.node_id) if a.node_id < b.node_id else (b.node_id, a.node_id)
        pair = pairs.setdefault(key, Pair(by_id[key[0]], by_id[key[1]], similarity))
        if source not in pair.found_by:
            pair.found_by.append(source)

    embedded = [f for f in findings if f.text in vectors]
    similar: list[tuple[float, Finding, Finding]] = []
    if len(embedded) > 1:
        matrix = np.stack([vectors[f.text] for f in embedded])
        scores = matrix @ matrix.T
        papers = np.array([f.paper_id for f in embedded])
        for i, finding in enumerate(embedded):
            row = np.where(papers == finding.paper_id, -1.0, scores[i])
            for j in np.argsort(-row)[:NEIGHBOURS]:
                if row[j] >= MIN_SIMILARITY:
                    similar.append((float(row[j]), finding, embedded[j]))
    similar.sort(key=lambda item: -item[0])
    for score, a, b in similar:
        if len(pairs) >= MAX_SIMILAR_PAIRS:
            break
        add(a, b, score, "similar_text")

    shared: list[tuple[float, Finding, Finding]] = []
    results = [f for f in findings if f.kind == "result" and f.dataset and f.metric]
    for i, a in enumerate(results):
        for b in results[i + 1 :]:
            if a.paper_id != b.paper_id and (a.dataset, a.metric) == (b.dataset, b.metric):
                score = float(vectors[a.text] @ vectors[b.text]) if a.text in vectors and b.text in vectors else 0.0
                shared.append((score, a, b))
    shared.sort(key=lambda item: -item[0])
    for score, a, b in shared[:MAX_SHARED_PAIRS]:
        add(a, b, score if a.text in vectors and b.text in vectors else None, "shared_setting")
    return list(pairs.values())


class PairLabel(BaseModel):
    pair: int = Field(description="The pair's number")
    relation: Relation
    same_setting: bool = Field(description="Whether both concern the same task, data, metric and conditions, as far as stated")
    reason: str = Field(description="One short sentence")


class PairLabels(BaseModel):
    """The extraction model's reply."""

    labels: list[PairLabel] = Field(description="One label per pair, in list order")


_SYSTEM = """You compare findings from different computer science and machine learning papers.

For each numbered pair, say how finding A relates to finding B:
- supports: they report the same finding, or one gives evidence for the other.
- contradicts: they cannot both be true in the same setting, such as opposite directions, incompatible numbers or conflicting conclusions.
- unrelated: neither, including findings about different things.
Also say whether the two concern the same setting: the same task, data, metric and conditions, as far as the text states. Results on different datasets or metrics are not the same setting. Judge only from the text given, and give one short reason."""


def _describe(finding: Finding) -> str:
    conditions = f" Conditions: {finding.conditions}." if finding.conditions else ""
    return f"{finding.text}{conditions} (from \"{finding.paper_title}\", which says: \"{finding.quote}\")"


def pair_prompt(pairs: list[Pair]) -> Prompt:
    lines: list[str] = []
    for number, pair in enumerate(pairs, start=1):
        lines += [f"[{number}]", f"A: {_describe(pair.a)}", f"B: {_describe(pair.b)}", ""]
    return Prompt("claim_pairs", 1, _SYSTEM, "\n".join(lines).strip())


def _pair_hash(pair: Pair) -> str:
    a, b = sorted((_describe(pair.a), _describe(pair.b)))
    return digest(pair_prompt([]).id, a, b)


def label_pairs(llm: LLM, judgments: JudgmentStore, pairs: list[Pair]) -> tuple[list[tuple[Pair, dict]], dict[str, int]]:
    """Each pair with its label (relation, same_setting, reason), and counts of calls
    made and labels reused. A pair the model leaves out gets no label."""
    labelled: list[tuple[Pair, dict]] = []
    stats = {"calls": 0, "reused": 0, "unlabelled": 0}
    pending: list[Pair] = []
    for pair in pairs:
        label = judgments.find("claim_pair", _pair_hash(pair))
        if label is None:
            pending.append(pair)
        else:
            labelled.append((pair, label))
            stats["reused"] += 1
    for start in range(0, len(pending), PAIRS_PER_CALL):
        batch = pending[start : start + PAIRS_PER_CALL]
        try:
            answer = llm.answer(Role.EXTRACTION, pair_prompt(batch), PairLabels)
        except LLMOutputError:
            stats["unlabelled"] += len(batch)
            continue
        stats["calls"] += 1
        by_number = {label.pair: label for label in answer.output.labels}
        for number, pair in enumerate(batch, start=1):
            label = by_number.get(number)
            if label is None:
                stats["unlabelled"] += 1
                continue
            data = label.model_dump(exclude={"pair"})
            judgments.add("claim_pair", _pair_hash(pair), answer.model, data)
            labelled.append((pair, data))
    return labelled, stats
