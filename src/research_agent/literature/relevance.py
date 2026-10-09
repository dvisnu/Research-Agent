"""The LLM relevance check: screens the best-ranked candidates before the corpus is cut.

The ranking rewards well-cited, well-linked papers, which can lift famous work on
another problem. The extraction model reads each candidate's title and abstract
and labels it core, related or off-topic. Only off-topic papers are dropped, so
the check removes noise without reordering the ranking.
"""

import sqlite3
from typing import Literal

from pydantic import BaseModel, Field

from research_agent.literature.brief import ResearchBrief
from research_agent.llm import LLM, Prompt, Role

BATCH = 15  # candidates per call
Label = Literal["core", "related", "off_topic"]


class Judgement(BaseModel):
    paper: int = Field(description="The candidate's number in the list")
    label: Label
    reason: str = Field(description="One short sentence saying why")


class Judgements(BaseModel):
    """The extraction model's reply."""

    judgements: list[Judgement] = Field(description="One judgement per candidate, in list order")


_SYSTEM = """You screen candidate papers for a literature review in computer science and machine learning.

You are given a research brief and a numbered list of candidate papers. Label every candidate:
- core: directly about the research problem, or proposes or evaluates a method, dataset or benchmark central to it.
- related: useful context for the problem: a foundational method or model it builds on, a closely neighbouring problem, a survey, or a likely baseline.
- off_topic: about a different problem; a paper on the research problem would have no reason to cite it.

Foundational work that the problem builds on is related or core, never off_topic, even when its wording differs from the brief. If a candidate has no abstract, judge it by its title. When unsure, choose related."""


def relevance_prompt(brief: ResearchBrief, batch: list[sqlite3.Row]) -> Prompt:
    lines = [f"Research question or idea: {brief.question}", f"Problem: {brief.problem}", "", "Candidates:"]
    for number, row in enumerate(batch, start=1):
        year = f" ({row['year']})" if row["year"] else ""
        lines += ["", f"[{number}] {row['title']}{year}", f"Abstract: {row['abstract'] or 'none available'}"]
    return Prompt(name="relevance_check", version=1, system=_SYSTEM, user="\n".join(lines))


def screen(
    llm: LLM, brief: ResearchBrief, candidates: list[sqlite3.Row], *, want: int, limit: int
) -> tuple[dict[str, Judgement], int]:
    """Judge candidates in batches, best-ranked first, until ``want`` of them are
    not off-topic or ``limit`` have been screened.

    Returns the judgements by paper ID, and how many candidates were screened. A
    candidate the reply leaves out has no judgement and counts as kept.
    """
    judged: dict[str, Judgement] = {}
    kept = screened = 0
    end = min(limit, len(candidates))
    while kept < want and screened < end:
        batch = candidates[screened : min(screened + BATCH, end)]
        screened += len(batch)
        reply = llm.structured(Role.EXTRACTION, relevance_prompt(brief, batch), Judgements)
        by_number = {judgement.paper: judgement for judgement in reply.judgements}
        for number, row in enumerate(batch, start=1):
            judgement = by_number.get(number)
            if judgement is not None:
                judged[row["paper_id"]] = judgement
            kept += judgement is None or judgement.label != "off_topic"
    return judged, screened
