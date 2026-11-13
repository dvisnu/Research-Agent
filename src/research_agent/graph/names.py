"""Name merging for tasks, methods, datasets and metrics.

Two steps. Code first gives each name a key that ignores case, punctuation,
spacing and generic words ("ResNet-50", "ResNet50" and "the ResNet 50 model"
share one). Then the extraction model sees the distinct names of one kind
together and groups those that name the same thing (an acronym and its full
name, "residual network" and "ResNet"), with different versions or sizes kept
apart. Embeddings are not used here: the free tier counts every text as a
request, and the model sees synonyms that string similarity misses.
"""

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from research_agent.graph.store import JudgmentStore, digest
from research_agent.llm import LLM, LLMOutputError, Prompt, Role

MAX_NAMES_PER_CALL = 250

_LEADING = re.compile(r"^(?:the|a|an)\s+")
_TRAILING = re.compile(
    r"\s+(?:dataset|datasets|data set|benchmark|benchmarks|corpus|model|models|method|methods|"
    r"framework|algorithm|approach|architecture|score|metric|task)$"
)


def name_key(name: str) -> str:
    text = unicodedata.normalize("NFKC", name).casefold()
    text = " ".join(re.sub(r"[^\w\s]", " ", text).split())
    for pattern in (_LEADING, _TRAILING):
        shorter = pattern.sub("", text)
        text = shorter or text
    return re.sub(r"[\W_]+", "", text) or name.strip().casefold()


_VAGUE_START = re.compile(
    r"^(?:various|several|many|multiple|all|some|different|relevant|other|these|those|existing|"
    r"standard|common|diverse|numerous|the following|a (?:variety|range|number|set) of)\b",
    re.IGNORECASE,
)
_UMBRELLA_TERMS = (
    "large language models", "LLMs", "LLM", "language models", "deep learning", "neural networks",
    "machine learning", "NLP", "natural language processing", "artificial intelligence", "AI", "understanding",
)
# A dataset or metric "name" in lower case ending in one of these describes, not names: "technical domains".
_GENERIC_PLURALS = {"domains", "datasets", "benchmarks", "tasks", "settings", "environments", "languages",
                    "metrics", "measures", "scores", "corpora", "collections"}
_TRAILING_ACRONYM = re.compile(r"\s*\([^()]*\)\s*$")


def vague_name(kind: str, name: str) -> bool:
    """A description rather than a name ("several benchmarks", "technical domains"), or, for
    tasks and methods, an umbrella term ("large language models (LLMs)"). Such names get no node."""
    text = name.strip()
    if not text or _VAGUE_START.match(text):
        return True
    if kind in ("dataset", "metric"):
        words = text.split()
        return text.islower() and words[-1] in _GENERIC_PLURALS
    if kind in ("task", "method"):
        return name_key(text) in _UMBRELLA or name_key(_TRAILING_ACRONYM.sub("", text) or text) in _UMBRELLA
    return False


@dataclass
class Entity:
    """One merged name: what to call it, and every way it was written."""

    name: str
    keys: set[str] = field(default_factory=set)
    aliases: Counter[str] = field(default_factory=Counter)


class NameGroup(BaseModel):
    names: list[int] = Field(min_length=2, description="The numbers of the names that refer to the same thing")
    canonical: int = Field(description="The number of the clearest name in the group")


class NameGroups(BaseModel):
    """The extraction model's reply."""

    groups: list[NameGroup] = Field(description="Only groups of two or more names; leave every other name out")


_SYSTEM = """You tidy the names in a knowledge graph built from computer science and machine learning papers.

You are given a numbered list of {kind} names written by different papers. Group the names that refer to the same {kind_one}: spelling variants, an abbreviation and its full name, or a name with and without a generic word. Do not group different things: different versions or sizes (GPT-3 and GPT-4, BERT-base and BERT-large), a method and its extension or variant, a dataset and its subset, or two related but distinct {kind}. When unsure, leave the names apart. For each group, give the number of its clearest name."""

_KINDS = {"task": "tasks", "method": "methods", "dataset": "datasets", "metric": "metrics"}


def name_prompt(kind: str, names: list[str]) -> Prompt:
    listing = "\n".join(f"[{n}] {name}" for n, name in enumerate(names, start=1))
    system = _SYSTEM.format(kind=_KINDS[kind], kind_one=kind)
    return Prompt("name_merge", 1, system, f"{_KINDS[kind].capitalize()}:\n{listing}")


def merge_names(
    llm: LLM, judgments: JudgmentStore, kind: str, mentions: Counter[str]
) -> tuple[dict[str, Entity], dict[str, int]]:
    """Merge the names of one kind. Returns each name key's entity, and counts:
    LLM calls made, calls reused, names merged by key and by the model."""
    by_key: dict[str, Counter[str]] = {}
    for name, count in mentions.items():
        by_key.setdefault(name_key(name), Counter())[name] += count
    entities = {key: Entity(_most_common(names), {key}, names) for key, names in by_key.items()}
    stats = {"calls": 0, "reused": 0, "by_key": sum(len(names) - 1 for names in by_key.values()), "by_model": 0}

    keys = sorted(by_key)
    for start in range(0, len(keys), MAX_NAMES_PER_CALL):
        part = keys[start : start + MAX_NAMES_PER_CALL]
        if len(part) < 2:
            continue
        names = [entities[key].name for key in part]
        prompt = name_prompt(kind, names)
        input_hash = digest(prompt.id, prompt.system, prompt.user)
        groups = judgments.find("name_groups", input_hash)
        if groups is None:
            try:
                answer = llm.answer(Role.EXTRACTION, prompt, NameGroups)
            except LLMOutputError:
                continue  # merged by key only
            groups = answer.output.model_dump()["groups"]
            judgments.add("name_groups", input_hash, answer.model, groups)
            stats["calls"] += 1
        else:
            stats["reused"] += 1
        stats["by_model"] += _apply_groups(entities, part, groups)
    return entities, stats


def _apply_groups(entities: dict[str, Entity], keys: list[str], groups: list[dict]) -> int:
    """Merge each valid group into one entity; returns how many entities were merged away.
    Numbers outside the list, and names already placed in an earlier group, are ignored."""
    merged = 0
    placed: set[int] = set()
    for group in groups:
        members = [n for n in dict.fromkeys(group["names"]) if 1 <= n <= len(keys) and n not in placed]
        if len(members) < 2:
            continue
        placed.update(members)
        canonical = group["canonical"] if group["canonical"] in members else members[0]
        target = entities[keys[canonical - 1]]
        for number in members:
            source = entities[keys[number - 1]]
            if source is target:
                continue
            target.keys |= source.keys
            target.aliases.update(source.aliases)
            for key in source.keys:
                entities[key] = target
            merged += 1
    return merged


def _most_common(names: Counter[str]) -> str:
    """The most frequent spelling; ties go to the shorter, then the alphabetically first."""
    return min(names, key=lambda name: (-names[name], len(name), name))


_UMBRELLA = {name_key(term) for term in _UMBRELLA_TERMS}
