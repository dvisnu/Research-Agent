import json

import httpx
import numpy as np
import pytest
from pydantic import SecretStr

from fakes import FakeTextEmbedder
from research_agent.embeddings import (
    EmbeddingCache,
    EmbeddingUnavailable,
    GeminiEmbeddings,
    MinuteWindow,
    text_embedder,
)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def test_the_window_holds_a_minutes_requests_under_the_limit():
    clock = Clock()
    window = MinuteWindow(90, 27_000, clock=clock, sleep=clock.sleep)
    window.admit(60, 100)
    clock.now = 10
    window.admit(30, 100)  # 90 in the window: still fits
    assert clock.slept == []
    window.admit(5, 100)  # waits until the first 60 leave the window
    assert clock.slept == [50]


def test_the_window_also_limits_tokens():
    clock = Clock()
    window = MinuteWindow(90, 1_000, clock=clock, sleep=clock.sleep)
    window.admit(1, 900)
    window.admit(1, 200)
    assert clock.slept == [60]


def test_the_window_does_not_spin_when_sleep_does_not_move_the_clock():
    window = MinuteWindow(1, 100, clock=lambda: 0.0, sleep=lambda seconds: None)
    window.admit(1, 1)
    window.admit(1, 1)  # returns: the wait is counted as served


def _gemini(handler, **options) -> GeminiEmbeddings:
    return GeminiEmbeddings("gemini-embedding-2", 4, "key", transport=httpx.MockTransport(handler),
                            sleep=lambda seconds: None, **options)


def test_gemini_batches_texts_with_the_task_prefix_and_dimensions():
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        assert request.url.path == "/v1beta/models/gemini-embedding-2:batchEmbedContents"
        assert request.headers["x-goog-api-key"] == "key"
        return httpx.Response(200, json={"embeddings": [{"values": [3, 4, 0, 0]} for _ in body["requests"]]})

    embedder = _gemini(handler)
    vectors = embedder.embed([f"claim {n}" for n in range(95)])

    assert vectors.shape == (95, 4)
    assert np.allclose(vectors[0], [0.6, 0.8, 0, 0])  # unit length
    assert [len(body["requests"]) for body in bodies] == [90, 5]  # 90 a minute at most
    first = bodies[0]["requests"][0]
    assert first["content"]["parts"][0]["text"] == "task: sentence similarity | query: claim 0"
    assert first["output_dimensionality"] == 4 and first["model"] == "models/gemini-embedding-2"
    assert embedder.model_id == "google:gemini-embedding-2@4"


def test_a_used_up_daily_quota_makes_gemini_unavailable_for_the_day():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "Quota exceeded for EmbedContentRequestsPerDayPerProject"}})

    with pytest.raises(EmbeddingUnavailable) as failure:
        _gemini(handler).embed(["a claim"])
    assert failure.value.lasting


def test_embedding_settings_build_the_named_model(settings):
    assert text_embedder(settings) is None
    configured = settings.model_copy(update={"embedding": "google:gemini-embedding-2", "google_api_key": SecretStr("k")})
    assert text_embedder(configured).model_id == "google:gemini-embedding-2@768"


def test_the_cache_sends_each_text_once(db):
    embedder = FakeTextEmbedder()
    cache = EmbeddingCache(db, embedder)
    first, report = cache.vectors(["alpha beta", "gamma", "alpha beta"])
    assert embedder.sent == ["alpha beta", "gamma"] and report["sent"] == 2 and report["missing"] == 0
    again, report = EmbeddingCache(db, embedder).vectors(["gamma", "delta"])
    assert embedder.sent[2:] == ["delta"]
    assert report["reused"] == 1 and report["sent"] == 1
    assert np.allclose(again["gamma"], first["gamma"])


def test_the_cache_returns_what_it_has_when_the_model_is_unavailable(db):
    class Down(FakeTextEmbedder):
        def embed(self, texts):
            raise EmbeddingUnavailable("quota used up", lasting=True)

    EmbeddingCache(db, FakeTextEmbedder()).vectors(["kept earlier"])
    vectors, report = EmbeddingCache(db, Down()).vectors(["kept earlier", "new text"])
    assert set(vectors) == {"kept earlier"}
    assert report["missing"] == 1 and report["unavailable"] == "quota used up"
