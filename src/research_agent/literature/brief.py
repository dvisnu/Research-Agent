"""The research brief: what a run searches for.

The first search angle is the question as written: its content words. The
reasoning model adds a problem statement, two or three angles in the field's own
vocabulary and, for an idea, its facets (purpose / mechanism / evaluation). Each
angle becomes one query per search channel, so the model's angles can widen the
search but never lose what the user's own words find. Candidates are ranked
against the question as written.
"""

import re
from typing import Literal

from pydantic import BaseModel, Field

from research_agent.graph.store import JudgmentStore, kept_answer
from research_agent.llm import LLM, Prompt, Role

# arXiv ANDs every term of a query, so its query uses only an angle's first few
# keywords: single words for the question's own angle, phrases for the model's.
QUESTION_ARXIV_TERMS = 4
ARXIV_TERMS = 2
# Function words, plus words too generic to narrow a literature search.
_STOPWORDS = set(
    """a an and are as at be been but by can could did do does doing for from had has have how i if in
    into is it its may might more most my no not of on or our should so some such than that the their
    them then there these they this those to under up us was we were what when where which while who
    why will with would you your each other between about over across without within via both
    improve improving improved better reduce reducing make making use using used new way ways
    approach approaches method methods system systems model models based work works existing
    good well want like help handle still close gap beyond only few many much very""".split()
)
# Characters that arXiv's query syntax would read as operators.
_UNSAFE = re.compile(r"[^\w\s\-+.]")


class Facets(BaseModel):
    purpose: str = Field(description="What the idea is for: the problem or need it addresses")
    mechanism: str = Field(description="How it would work: the technique or change it proposes")
    evaluation: str = Field(description="How it would be tested: tasks, datasets or metrics, as far as the idea implies them")


class AngleDraft(BaseModel):
    label: str = Field(description="A few words naming this angle")
    keywords: list[str] = Field(
        min_length=1, max_length=5,
        description="2 to 4 search terms, most specific first. Each is 1 to 3 words, worded as paper titles and abstracts word it",
    )
    passage: str = Field(description="One sentence that the abstract of a highly relevant paper might contain")


class BriefDraft(BaseModel):
    """The reasoning model's reply."""

    input_kind: Literal["question", "idea"] = Field(
        description="question: the user asks something; idea: the user proposes something to build or try"
    )
    problem: str = Field(description="The research problem in one or two sentences, in the field's own terms")
    facets: Facets | None = Field(None, description="Only for an idea; null for a question")
    angles: list[AngleDraft] = Field(min_length=1, max_length=4, description="2 or 3 distinct angles to search from")


_SYSTEM = """You plan literature searches for research in computer science and machine learning.

You are given a research question, or a rough idea the user wants to develop. Write a research brief:
- Say whether it is a question or an idea, and state the underlying research problem in one or two sentences, in the terms the field uses.
- For an idea, describe its facets: purpose, mechanism and evaluation. Leave facets null for a question.
- Give 2 or 3 search angles that together cover the literature a researcher would need, for example the problem itself, the main family of methods for it, and how it is evaluated. The angles must differ, not rephrase one another. The user's own words are already searched, so the angles should add the field's vocabulary.
- For each angle, give 2 to 4 keywords as they appear in paper titles and abstracts. Prefer the established names of tasks, methods and benchmarks over everyday words. The first angle's first keyword is the name the literature gives the problem itself, the term its survey papers use. Then give one sentence that a relevant paper's abstract might contain.

Do not invent paper titles, authors or results."""


def brief_prompt(question: str) -> Prompt:
    return Prompt(name="research_brief", version=2, system=_SYSTEM, user=f"Research question or idea:\n{question}")


class SearchAngle(BaseModel):
    label: str
    keywords: list[str]
    keyword_query: str  # for keyword search (Semantic Scholar, OpenAlex)
    arxiv_query: str  # arXiv search syntax
    passage_query: str  # natural language, for snippet search


class ResearchBrief(BaseModel):
    question: str
    input_kind: Literal["question", "idea"]
    problem: str
    facets: Facets | None
    angles: list[SearchAngle]
    rank_query: str  # what candidates are ranked against: the question as written


def write_brief(llm: LLM, question: str, judgments: JudgmentStore) -> ResearchBrief:
    """The brief for ``question``; asking the same question again gets the same brief."""
    draft = kept_answer(llm, judgments, "brief", Role.REASONING, brief_prompt(question), BriefDraft)
    return ResearchBrief(
        question=question,
        input_kind=draft.input_kind,
        problem=draft.problem,
        facets=draft.facets,
        angles=[question_angle(question), *(make_angle(angle) for angle in draft.angles)],
        rank_query=" ".join(question.split()),
    )


def question_angle(question: str) -> SearchAngle:
    """The question's content words, in order."""
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*[A-Za-z0-9]|[A-Za-z0-9]", question)
    keywords: list[str] = []
    for word in words:
        lowered = word.lower()
        if lowered not in _STOPWORDS and len(lowered) > 1 and lowered not in keywords:
            keywords.append(lowered)
    return SearchAngle(
        label="the question as written",
        keywords=keywords,
        keyword_query=" ".join(keywords),
        arxiv_query=_arxiv_query(keywords, QUESTION_ARXIV_TERMS),
        passage_query=" ".join(question.split()),
    )


def make_angle(draft: AngleDraft) -> SearchAngle:
    keywords: list[str] = []
    for keyword in draft.keywords:
        cleaned = " ".join(_UNSAFE.sub(" ", keyword).split())
        if cleaned and cleaned.lower() not in (k.lower() for k in keywords):
            keywords.append(cleaned)
    return SearchAngle(
        label=draft.label,
        keywords=keywords,
        keyword_query=" ".join(keywords),
        arxiv_query=_arxiv_query(keywords, ARXIV_TERMS),
        passage_query=" ".join(draft.passage.split()),
    )


def _arxiv_query(keywords: list[str], terms: int) -> str:
    return " AND ".join(
        f'all:"{k.replace("-", " ")}"' if " " in k or "-" in k else f"all:{k}" for k in keywords[:terms]
    )
