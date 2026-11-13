"""Paper extraction: the extraction model fills a fixed form for each part of a
paper, citing every item's evidence as a sentence address and an exact quote.

Full-read papers are cut into chunks of whole sections, about 7k tokens each;
appendices are left out. The first chunk also asks for the paper's facets
(purpose / mechanism / evaluation). Abstract-read papers get a shorter form,
several abstracts per call. Replies are kept (``ExtractionStore``) with their
addresses resolved to full sentence IDs; the quote check runs when they are read.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from research_agent.llm import Prompt
from research_agent.storage import Database, parse_sentence_id

CHUNK_CHARS = 28_000  # sentence text per chunk: about 7k tokens
ABSTRACT_BATCH = 6  # abstracts per call
ABSTRACT_SENTENCES = 15  # read from the first section when a paper has no Abstract section
_APPENDIX = re.compile(r"^\s*(?:Appendix|Appendices|Supplementary)\b", re.IGNORECASE)
_ABSTRACT = re.compile(r"^\s*abstract\s*$", re.IGNORECASE)


# The form -------------------------------------------------------------------

class Evidence(BaseModel):
    sentence: str = Field(description="The address in brackets before the sentence")
    quote: str = Field(description="Words copied exactly from that one sentence")


class Mention(BaseModel):
    name: str
    evidence: list[Evidence] = Field(min_length=1)


class MethodMention(Mention):
    role: Literal["proposed", "used"]


class Statement(BaseModel):
    text: str = Field(description="One self-contained sentence")
    evidence: list[Evidence] = Field(min_length=1)


class Result(Statement):
    method: str | None = Field(description="The method the result is about, or null")
    dataset: str | None = Field(description="The dataset or benchmark, or null")
    metric: str | None = Field(description="The metric, or null")
    conditions: str | None = Field(description="The setting it holds in, or null")


class Facets(BaseModel):
    purpose: Statement | None
    mechanism: Statement | None
    evaluation: Statement | None


class ChunkExtraction(BaseModel):
    """The form for one part of a paper."""

    tasks: list[Mention]
    methods: list[MethodMention]
    datasets: list[Mention]
    metrics: list[Mention]
    results: list[Result]
    claims: list[Statement]
    limitations: list[Statement]
    assumptions: list[Statement]
    future_work: list[Statement]


class OpeningChunkExtraction(ChunkExtraction):
    """The form for the first part of a paper, which also has the paper's facets."""

    facets: Facets


class AbstractExtraction(BaseModel):
    paper: int = Field(description="The paper's number in the list")
    tasks: list[Mention]
    methods: list[MethodMention]
    datasets: list[Mention]
    metrics: list[Mention]
    results: list[Result]
    claims: list[Statement]
    facets: Facets


class AbstractExtractions(BaseModel):
    """The extraction model's reply for a batch of abstracts."""

    papers: list[AbstractExtraction] = Field(description="One form per paper, in list order")


# Prompts --------------------------------------------------------------------

_FIELDS = """- tasks: the research tasks or problems the paper itself works on.
- methods: methods, models, algorithms and architectures. role is "proposed" if this paper introduces it, "used" if it is existing work the paper uses, builds on or compares with.
- datasets: named datasets and benchmarks.
- metrics: named evaluation metrics.
- results: the paper's own experimental findings, each with the method, dataset, metric and conditions it applies to, as far as stated. Keep the numbers.
- claims: other conclusions the authors state or argue for, such as observations, analyses and explanations. Not results, not background."""

_MORE_FIELDS = """
- limitations: weaknesses, failure cases or open issues the authors report, in their own method or in earlier work.
- assumptions: assumptions their method or analysis relies on, when stated.
- future_work: directions or open problems the authors suggest."""

_FACET_FIELDS = """- facets: the paper's purpose (the problem or need it addresses), mechanism (how its approach works) and evaluation (how it is tested: tasks, datasets, metrics), each one sentence with evidence, or null when the text does not say."""

_RULES = """Rules:
- Use only what the text says, never outside knowledge. Leave a list empty when the text has nothing for it.
- Every item needs evidence: the address of each sentence it rests on, with a quote copied word for word from that sentence. A quote is a phrase or the whole sentence, at most about 30 words, and never joins words from two sentences: cite two sentences as two evidence entries. Do not correct or reword anything inside a quote.
- Write each result, claim, limitation, assumption, future-work item and facet as one self-contained sentence that names what it is about, so it can be read without the paper. One fact per item: split compound statements.
- Name methods, datasets, metrics and tasks as the paper does, using its short name or acronym when it defines one. List each once.
- List only the tasks, methods, datasets and metrics the paper itself works with: what it proposes, uses, compares against or evaluates on (in a survey, what it reviews). Skip those it names only when describing other papers, in related work or in examples.
- Use specific names. A task is "open-domain question answering", not "understanding" or "NLP". Skip umbrella terms such as "large language models", "deep learning" or "neural networks" unless the paper studies that thing itself. In a result, leave dataset or metric null rather than describing it ("several benchmarks", "technical domains").
- Leave out results and claims that the text attributes to other papers, unless this paper confirms or disputes them.
- Be selective: at most 10 items in any list, the most important first."""

_INTRO = (
    "You extract structured facts from computer science and machine learning papers, "
    "for a knowledge graph in which every fact points to the sentence it came from."
)

FULL_SYSTEM = f"""{_INTRO}

You are given one part of a paper. Each sentence follows its address in brackets, e.g. [s3:12]. Fill in the form from this part only:
{_FIELDS}{_MORE_FIELDS}

{_RULES}"""

_OPENING_NOTE = """This part is the start of the paper, so also fill in:
""" + _FACET_FIELDS

ABSTRACT_SYSTEM = f"""{_INTRO}

You are given several numbered paper abstracts. Each sentence follows its address in brackets: the paper's number, a colon, then the sentence's number, e.g. [2:0]. Fill in one form per paper, with its number, from its abstract only:
{_FIELDS}
{_FACET_FIELDS}

{_RULES}"""

FULL_PROMPT = ("paper_extraction", 2)
ABSTRACT_PROMPT = ("abstract_extraction", 2)


# Paper text -----------------------------------------------------------------

@dataclass(frozen=True)
class Line:
    sentence_id: str
    text: str


@dataclass
class Section:
    idx: int
    heading: str | None
    lines: list[Line]


def load_sections(db: Database, paper_id: str) -> list[Section]:
    """The paper's current text (earlier texts' sections are kept, retired, for their evidence)."""
    rows = db.query(
        "SELECT s.idx AS section, s.heading, t.sentence_id, t.text FROM sections s"
        " JOIN sentences t ON t.section_id = s.section_id WHERE s.paper_id = ? AND s.retired_by IS NULL"
        " ORDER BY s.idx, t.idx",
        (paper_id,),
    )
    sections: list[Section] = []
    for row in rows:
        if not sections or sections[-1].idx != row["section"]:
            sections.append(Section(row["section"], row["heading"], []))
        sections[-1].lines.append(Line(row["sentence_id"], row["text"]))
    return sections


@dataclass
class Block:
    heading: str | None
    continued: bool  # the section began in the previous chunk
    lines: list[Line]


@dataclass
class Chunk:
    index: int
    blocks: list[Block] = field(default_factory=list)
    sections: list[int] = field(default_factory=list)

    @property
    def lines(self) -> list[Line]:
        return [line for block in self.blocks for line in block.lines]

    def size(self) -> int:
        return _chars(self.lines)

    def add(self, block: Block, section: int) -> None:
        self.blocks.append(block)
        if section not in self.sections:
            self.sections.append(section)


def chunk_sections(sections: list[Section], limit: int = CHUNK_CHARS) -> list[Chunk]:
    """Whole sections in reading order, up to ``limit`` characters a chunk, without the
    appendices. Only a section longer than a chunk is split, between sentences."""
    chunks: list[Chunk] = []
    for section in sections:
        if section.heading and _APPENDIX.match(section.heading):
            break
        for n, piece in enumerate(_pieces(section.lines, limit)):
            if not chunks or chunks[-1].size() + _chars(piece) > limit:
                chunks.append(Chunk(len(chunks)))
            chunks[-1].add(Block(section.heading, n > 0, piece), section.idx)
    return chunks


def _pieces(lines: list[Line], limit: int) -> list[list[Line]]:
    pieces: list[list[Line]] = [[]]
    for line in lines:
        if pieces[-1] and _chars(pieces[-1]) + len(line.text) > limit:
            pieces.append([])
        pieces[-1].append(line)
    return [piece for piece in pieces if piece]


def _chars(lines: list[Line]) -> int:
    return sum(len(line.text) for line in lines)


def short_label(sentence_id: str) -> str:
    """How a full paper's sentence is labelled for the model: s<section>:<sentence>."""
    _, section, sentence = parse_sentence_id(sentence_id)
    return f"s{section}:{sentence}"


def chunk_prompt(title: str, chunk: Chunk, total: int) -> Prompt:
    lines = [f"Paper: {title}", f"Part {chunk.index + 1} of {total}."]
    if chunk.index == 0:
        lines.append(_OPENING_NOTE)
    for block in chunk.blocks:
        lines += ["", f"## {block.heading or 'Untitled section'}{' (continued)' if block.continued else ''}"]
        lines += [f"[{short_label(line.sentence_id)}] {line.text}" for line in block.lines]
    return Prompt(*FULL_PROMPT, FULL_SYSTEM, "\n".join(lines))


@dataclass
class AbstractItem:
    paper_id: str
    title: str
    year: int | None
    section: int
    lines: list[Line]


def abstract_item(paper_id: str, title: str, year: int | None, sections: list[Section]) -> AbstractItem | None:
    """The paper's Abstract section, or else the start of its first section."""
    if not sections:
        return None
    section = next((s for s in sections if s.heading and _ABSTRACT.match(s.heading)), None)
    if section is not None:
        return AbstractItem(paper_id, title, year, section.idx, section.lines)
    return AbstractItem(paper_id, title, year, sections[0].idx, sections[0].lines[:ABSTRACT_SENTENCES])


def abstract_block(number: int, item: AbstractItem) -> list[str]:
    year = f" ({item.year})" if item.year else ""
    return [f"[{number}] {item.title}{year}", *(f"[{number}:{n}] {line.text}" for n, line in enumerate(item.lines))]


def abstract_prompt(items: list[AbstractItem]) -> Prompt:
    lines: list[str] = []
    for number, item in enumerate(items, start=1):
        if lines:
            lines.append("")
        lines += abstract_block(number, item)
    return Prompt(*ABSTRACT_PROMPT, ABSTRACT_SYSTEM, "\n".join(lines))


# Addresses ------------------------------------------------------------------

_FULL_LABEL = re.compile(r"^\[?\s*s?\s*(\d+)\s*:\s*(\d+)\s*\]?$", re.IGNORECASE)
_ABSTRACT_LABEL = re.compile(r"^\[?\s*(\d+)\s*:\s*(\d+)\s*\]?$")


def full_resolver(paper_id: str) -> Callable[[str], str | None]:
    def resolve(label: str) -> str | None:
        match = _FULL_LABEL.match(label.strip())
        return f"{paper_id}:s{match[1]}:{match[2]}" if match else None

    return resolve


def abstract_resolver(number: int, item: AbstractItem) -> Callable[[str], str | None]:
    def resolve(label: str) -> str | None:
        match = _ABSTRACT_LABEL.match(label.strip())
        if match and int(match[1]) == number and int(match[2]) < len(item.lines):
            return item.lines[int(match[2])].sentence_id
        return None

    return resolve


def resolve_addresses(output: Any, resolve: Callable[[str], str | None]) -> Any:
    """The reply with each evidence label replaced by its full sentence ID. A label
    that names no sentence is kept as given, for the quote check to deal with."""
    if isinstance(output, list):
        return [resolve_addresses(item, resolve) for item in output]
    if isinstance(output, dict):
        if set(output) == {"sentence", "quote"}:
            return {"sentence": resolve(output["sentence"]) or output["sentence"], "quote": output["quote"]}
        return {key: resolve_addresses(value, resolve) for key, value in output.items()}
    return output
