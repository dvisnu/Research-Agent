"""Fakes for pipeline tests: literature sources that answer from canned data,
through the real client classes, a cheap embedder, and an LLM that answers the
pipeline's prompts the way a cooperative model would."""

import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from research_agent.clients import Arxiv, OpenAlex, SemanticScholar, Unpaywall
from research_agent.config import Settings
from research_agent.llm import LLM, Completion
from research_agent.literature.sources import ArxivWeb, Downloader, Sources
from research_agent.pipeline import Services

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


class FakeModel:
    """Answers the research brief and the relevance check; records every call.

    The brief has ``angles`` angles, keyed on the question's longer words. The
    relevance check calls a candidate off-topic when its title contains one of
    ``off_topic``, and core otherwise.
    """

    def __init__(self, *, angles: int = 1, off_topic: tuple[str, ...] = ()) -> None:
        self.angles = angles
        self.off_topic = off_topic
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
        else:
            raise AssertionError(f"FakeModel has no answer for {json_schema['title']}")
        return Completion(json.dumps(reply), usage={"input_tokens": 100, "output_tokens": 20})


def fake_llm(settings: Settings, model: FakeModel | None = None) -> LLM:
    roles = {"llm_reasoning": "fake:reasoner", "llm_extraction": "fake:extractor"}
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
    settings: Settings, tmp_path: Path, router: Router | None = None, *, s2: bool = True, model: FakeModel | None = None
) -> Services:
    router = router or Router()
    common = {"transport": httpx.MockTransport(router), "sleep": lambda seconds: None}
    sources = Sources(
        arxiv=Arxiv(**common),
        arxiv_web=ArxivWeb(**common),
        openalex=OpenAlex(settings, **common),
        downloader=Downloader(**common),
        s2=SemanticScholar(settings, rate_file=tmp_path / "s2-rate", **common) if s2 else None,
        unpaywall=Unpaywall(settings, **common) if settings.contact_email else None,
    )
    return Services(sources=sources, embedder=FakeEmbedder(), llm=fake_llm(settings, model))
