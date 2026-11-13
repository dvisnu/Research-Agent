"""Fakes for pipeline tests: literature sources that answer from canned data,
through the real client classes, cheap embedders, and an LLM that answers the
pipeline's prompts the way a cooperative model would."""

import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from research_agent.clients.cache import ResponseCache
from research_agent.clients import Arxiv, OpenAlex, SemanticScholar, Unpaywall
from research_agent.config import Settings
from research_agent.llm import LLM, Completion
from research_agent.literature.sources import ArxivWeb, Downloader, Sources
from research_agent.pipeline import Services
from research_agent.plans.facts import KIND_TITLES

EMPTY_FEED = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'


class FakeEmbedder:
    """Hashed bag of words: texts sharing words get similar vectors."""

    DIM = 256

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.DIM)
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIM] += 1
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def papers(self, items: list[tuple[str, str | None]]) -> np.ndarray:
        return np.array([self._vector(f"{title} {abstract or ''}") for title, abstract in items]).reshape(
            len(items), self.DIM
        )

    def query(self, text: str) -> np.ndarray:
        return self._vector(text)


class FakeTextEmbedder(FakeEmbedder):
    """The text embedder for claims: the same hashed bag of words. Records what it was sent."""

    model_id = "fake:bow@256"

    def __init__(self) -> None:
        self.sent: list[str] = []

    def embed(self, texts: list[str]) -> np.ndarray:
        self.sent += texts
        return np.array([self._vector(text) for text in texts]).reshape(len(texts), self.DIM)


_FORM_LISTS = ("tasks", "methods", "datasets", "metrics", "results", "claims", "limitations", "assumptions", "future_work")
_NO_FACETS = {"purpose": None, "mechanism": None, "evaluation": None}


def first_sentence_claim(sentences: list[tuple[str, str]]) -> list[dict]:
    """A claim made of the first sentence, quoting its first six words."""
    if not sentences:
        return []
    label, text = sentences[0]
    return [{"text": text, "evidence": [{"sentence": label, "quote": " ".join(text.split()[:6])}]}]


class FakeModel:
    """Answers every prompt of the pipeline; records every call.

    The brief has ``angles`` angles, keyed on the question's longer words. The
    relevance check calls a candidate off-topic when its title contains one of
    ``off_topic``, and core otherwise. A paper extraction reports one claim, the
    first sentence shown, unless ``extract`` (prompt, output name) gives the
    reply. Name merging groups nothing, and every claim pair supports. A plan cites
    the first listed facts of each kind (a revision also the first new result, method
    or dataset, for critique point 1); the critique makes no point, and the verifier
    finds every claim supported.
    """

    def __init__(
        self,
        *,
        angles: int = 1,
        off_topic: tuple[str, ...] = (),
        extract: Callable[[str, str], dict] | None = None,
    ) -> None:
        self.angles = angles
        self.off_topic = off_topic
        self.extract = extract
        self.calls: list[dict[str, Any]] = []

    def complete(self, *, model, system, messages, json_schema) -> Completion:
        self.calls.append({"model": model, "output": json_schema["title"], "prompt": messages[0].content})
        prompt = messages[0].content
        if json_schema["title"] == "BriefDraft":
            question = prompt.split("\n", 1)[1]
            words = [w for w in re.findall(r"[A-Za-z][A-Za-z\-]+", question) if len(w) > 3] or ["research"]
            angles = [
                {"label": f"angle {i}", "keywords": words[i * 2 : i * 2 + 4] or words[:2], "passage": question}
                for i in range(self.angles)
            ]
            reply = {"input_kind": "question", "problem": question, "facets": None, "angles": angles}
        elif json_schema["title"] == "Judgements":
            reply = {"judgements": [
                {"paper": int(number), "reason": "fake",
                 "label": "off_topic" if any(t in title for t in self.off_topic) else "core"}
                for number, title in re.findall(r"^\[(\d+)\] (.*)$", prompt, re.MULTILINE)
            ]}
        elif self.extract is not None and json_schema["title"] in (
            "ChunkExtraction", "OpeningChunkExtraction", "AbstractExtractions", "NameGroups", "PairLabels",
            "CellPicks", "GapWriteups", "IdeaDrafts", "Revisions", "NoveltyQueries", "CloseWork", "PaperComparison",
            "IdeaFacets", "PlanDraft", "PlanRevision", "PlanCritique", "Verdicts",
        ) and (custom := self.extract(prompt, json_schema["title"])) is not None:
            reply = custom
        elif json_schema["title"] in ("ChunkExtraction", "OpeningChunkExtraction"):
            reply = {key: [] for key in _FORM_LISTS}
            reply["claims"] = first_sentence_claim(re.findall(r"^\[(s\d+:\d+)\] (.*)$", prompt, re.MULTILINE))
            if json_schema["title"] == "OpeningChunkExtraction":
                reply["facets"] = _NO_FACETS
        elif json_schema["title"] == "AbstractExtractions":
            by_paper: dict[int, list[tuple[str, str]]] = {}
            for number, sentence, text in re.findall(r"^\[(\d+):(\d+)\] (.*)$", prompt, re.MULTILINE):
                by_paper.setdefault(int(number), []).append((f"{number}:{sentence}", text))
            reply = {"papers": [
                {"paper": number, **{key: [] for key in _FORM_LISTS[:6]}, "claims": first_sentence_claim(sentences),
                 "facets": _NO_FACETS}
                for number, sentences in by_paper.items()
            ]}
        elif json_schema["title"] == "NameGroups":
            reply = {"groups": []}
        elif json_schema["title"] == "PairLabels":
            reply = {"labels": [
                {"pair": int(number), "relation": "supports", "same_setting": True, "reason": "fake"}
                for number in re.findall(r"^\[(\d+)\]$", prompt, re.MULTILINE)
            ]}
        elif json_schema["title"] == "CellPicks":
            pairs = re.findall(r"^\[(\d+)\] M\d+ x D\d+$", prompt, re.MULTILINE)
            reply = {"pairs": [{"pair": int(pairs[0]), "reason": "fake"}] if pairs else []}
        elif json_schema["title"] == "GapWriteups":
            reply = {"gaps": [
                {"candidate": int(number), "gap": f"Gap: {text}", "warrant": "fake warrant", "keep": True,
                 "duplicate_of": None}
                for number, text in re.findall(r"^\[(\d+)\] \w+[^:]*: (.*)$", prompt, re.MULTILINE)
            ]}
        elif json_schema["title"] == "IdeaDrafts":
            reply = {"ideas": [
                {"gap": int(number), "kind": "fix_limitation", "title": f"Idea for gap {number}",
                 "purpose": f"Address gap {number}: {text}", "mechanism": f"Mechanism {number} for the gap.",
                 "evaluation": "Evaluate on GLUE with accuracy against LoRA.", "builds_on": ["LoRA"],
                 "feasibility": "fake"}
                for number, text in re.findall(r"^\[(\d+)\] \(\w+\) (.*)$", prompt, re.MULTILINE)
            ]}
        elif json_schema["title"] == "Revisions":
            reply = {"ideas": [
                {"idea": int(number), "changed": "mechanism", "new_text": f"Revised mechanism {number}.",
                 "title": f"Revised idea {number}", "why": "fake", "feasibility": "fake"}
                for number in re.findall(r"^\[(\d+)\]$", prompt, re.MULTILINE)
            ]}
        elif json_schema["title"] == "NoveltyQueries":
            title = re.search(r"^Title: (.*)$", prompt, re.MULTILINE)[1]
            reply = {"keywords": [{"terms": title.split()[:3]}], "titles": [title]}
        elif json_schema["title"] == "CloseWork":
            numbers = re.findall(r"^\[(\d+)\] ", prompt, re.MULTILINE)
            reply = {"papers": [{"paper": int(n), "match": "one_facet"} for n in numbers[:10]]}
        elif json_schema["title"] == "PaperComparison":
            sentences = re.findall(r"^\[((?:s\d+:)?\d+)\] (.*)$", prompt, re.MULTILINE)
            label, text = sentences[0] if sentences else ("0", "")
            evidence = [{"sentence": label, "quote": " ".join(text.split()[:5])}]
            facet = {"match": "different", "paper": text, "evidence": evidence}
            reply = {"purpose": facet, "mechanism": facet, "evaluation": facet, "solves_gap": "no",
                     "gap_evidence": [], "difference": "fake difference"}
        elif json_schema["title"] == "IdeaFacets":
            text = prompt.split("\n", 1)[1]
            reply = {"title": text[:60], "purpose": text, "mechanism": text, "evaluation": "fake evaluation"}
        elif json_schema["title"] in ("PlanDraft", "PlanRevision"):
            reply = fake_plan(prompt, revision=json_schema["title"] == "PlanRevision")
        elif json_schema["title"] == "PlanCritique":
            reply = {"answers_gap": "yes", "points": []}
        elif json_schema["title"] == "Verdicts":
            reply = {"verdicts": [{"claim": int(n), "label": "supported", "reason": "fake"}
                                  for n in re.findall(r"^\[(\d+)\] Plan element", prompt, re.MULTILINE)]}
        else:
            raise AssertionError(f"FakeModel has no answer for {json_schema['title']}")
        return Completion(json.dumps(reply), usage={"input_tokens": 100, "output_tokens": 20})


def listed_facts(prompt: str) -> dict[str, list[tuple[str, str, bool]]]:
    """The facts a plan prompt lists, by kind: (number, line, marked new)."""
    kinds = {title: kind for kind, title in KIND_TITLES.items()}
    found: dict[str, list[tuple[str, str, bool]]] = {}
    kind = None
    for line in prompt.splitlines():
        if line.endswith(":") and line[:-1] in kinds:
            kind = kinds[line[:-1]]
        elif kind and (match := re.match(r"^\[(F\d+)\]( \(new\))? (.*)$", line)):
            found.setdefault(kind, []).append((match[1], match[3], bool(match[2])))
    return found


def fake_plan(prompt: str, revision: bool = False) -> dict:
    facts = listed_facts(prompt)

    def first(kind: str, n: int = 0) -> list[tuple[str, str, bool]]:
        return facts.get(kind, [])[n : n + 1]

    def item(text: str, cited: list[tuple[str, str, bool]], name: str = "", issues: tuple[int, ...] = ()) -> dict:
        return {"name": name, "text": text, "cites": [c[0] for c in cited],
                "basis": f"The sources say: {cited[0][1][:80]}" if cited else "", "decision": "fake decision",
                "issues": list(issues)}

    def named(kind: str, n: int = 0) -> list[dict]:
        return [item(f"{c[1].split(':')[0]}: use it.", [c], c[1].split(":")[0]) for c in first(kind, n)]

    gap = first("gap")
    plan = {
        "hypothesis": item("The mechanism beats the baselines.", gap),
        "research_question": item("Does the mechanism close the gap?", gap),
        "method": [item("Build on the closest work.", first("approach") or first("method")),
                   item("A step new to this idea.", [])],
        "datasets": named("dataset"), "baselines": named("method") + named("method", 1), "metrics": named("metric"),
        "conditions": [item("Same backbone for every method.", first("dataset"))],
        "ablations": [item("Remove the new step.", first("method"))],
        "expected_results": [item("Better than the reported result.", first("result"))],
        "risks": [item("The reported limitation may hold.", first("limitation"))],
    }
    if revision:
        new = [c for kind in ("result", "method", "dataset") for c in facts.get(kind, []) if c[2]][:1]
        if new:
            plan["expected_results"].append(item("Better than the newly found result.", new, issues=(1,)))
        points = re.findall(r"^\[(\d+)\] \(", prompt, re.MULTILINE)
        plan["answers"] = [{"point": int(n), "action": "fixed", "note": "fake"} for n in points]
    return plan


def fake_llm(settings: Settings, model: FakeModel | None = None) -> LLM:
    roles = {"llm_reasoning": "fake:reasoner", "llm_extraction": "fake:extractor", "llm_verifier": "fake:verifier"}
    return LLM(settings.model_copy(update=roles), providers={"fake": model or FakeModel()})


class Router:
    """Answers requests by host and path prefix; anything unrouted gets an empty result."""

    def __init__(self) -> None:
        self.routes: list[tuple[str, str, Callable[[httpx.Request], httpx.Response]]] = []
        self.requests: list[httpx.Request] = []

    def add(self, host: str, path_prefix: str, response: Any) -> None:
        handler = response if callable(response) else (lambda request, r=response: _respond(r))
        self.routes.insert(0, (host, path_prefix, handler))

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        matches = [
            (len(prefix), handler)
            for host, prefix, handler in self.routes
            if request.url.host == host and request.url.path.startswith(prefix)
        ]
        if matches:  # the longest prefix wins; among equals, the latest added
            return max(matches, key=lambda match: match[0])[1](request)
        if request.url.host == "export.arxiv.org":
            return httpx.Response(200, text=EMPTY_FEED)
        if request.url.host == "api.openalex.org":
            return httpx.Response(200, json={"results": []})
        if request.url.host == "api.semanticscholar.org":
            if request.url.path.endswith("/batch"):
                return httpx.Response(200, json=[None] * len(json.loads(request.content)["ids"]))
            return httpx.Response(200, json={"data": [], "recommendedPapers": []})
        return httpx.Response(404)


def _respond(value: Any) -> httpx.Response:
    if isinstance(value, httpx.Response):
        return value
    if isinstance(value, str):
        return httpx.Response(200, text=value)
    if isinstance(value, bytes):
        return httpx.Response(200, content=value)
    return httpx.Response(200, json=value)


def fake_services(
    settings: Settings, tmp_path: Path, router: Router | None = None, *, s2: bool = True, model: FakeModel | None = None,
    cache: bool = False,
) -> Services:
    """Fake sources and models. ``cache`` keeps the search APIs' responses, as a real run does."""
    router = router or Router()
    common = {"transport": httpx.MockTransport(router), "sleep": lambda seconds: None}
    kept = ResponseCache(settings.responses_path, settings.response_cache_days) if cache else None
    search = {**common, "cache": kept}
    sources = Sources(
        arxiv=Arxiv(**search),
        arxiv_web=ArxivWeb(**common),
        openalex=OpenAlex(settings, **search),
        downloader=Downloader(**common),
        s2=SemanticScholar(settings, rate_file=tmp_path / "s2-rate", **search) if s2 else None,
        unpaywall=Unpaywall(settings, **search) if settings.contact_email else None,
        cache=kept,
    )
    return Services(
        sources=sources, embedder=FakeEmbedder(), llm=fake_llm(settings, model), text_embedder=FakeTextEmbedder()
    )
