"""Text embeddings, used to find claims in different papers that say similar things.

``RA_EMBEDDING=provider:model`` picks the model and ``RA_EMBEDDING_DIMS`` the
vector size. Every vector is kept in SQLite under a hash of its text, so a text
is sent at most once: Google's free tier counts each text in a batch as one
request (100 a minute, 30k tokens a minute, 1,000 a day). When the model can't
serve, callers get vectors for the texts already kept and go on without the rest.
"""

import hashlib
import time
from collections import deque
from collections.abc import Callable
from typing import Any, Protocol

import httpx
import numpy as np

from research_agent.clients.http import ApiClient, RetryableStatus
from research_agent.config import Settings
from research_agent.llm_providers import ATTEMPTS, FIRST_WAIT, daily_quota, error_text
from research_agent.storage import Database, utc_now
from research_agent.trace import count

BATCH = 100  # texts per request: the API's maximum


class EmbeddingUnavailable(RuntimeError):
    """The model can't serve for now. ``lasting``: its daily quota is used up."""

    def __init__(self, message: str, *, lasting: bool = False) -> None:
        super().__init__(message)
        self.lasting = lasting


class TextEmbedder(Protocol):
    model_id: str  # "provider:model@dimensions": vectors from different models never mix

    def embed(self, texts: list[str]) -> np.ndarray:
        """Unit vectors, one row per text. Raise EmbeddingUnavailable when the model can't serve."""
        ...


class MinuteWindow:
    """Keeps what is sent in any 60 seconds under a request limit and a token limit."""

    def __init__(
        self,
        requests: int,
        tokens: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.requests = requests
        self.tokens = tokens
        self._clock = clock
        self._sleep = sleep
        self._sent: deque[tuple[float, int, int]] = deque()  # (time, requests, tokens)

    def admit(self, requests: int, tokens: int) -> None:
        """Block until ``requests`` and ``tokens`` fit in the window, then count them in."""
        now = self._clock()
        self._expire(now)
        while self._sent and not self._fits(requests, tokens):
            oldest = self._sent[0][0]
            if oldest + 60 > now:
                self._sleep(oldest + 60 - now)
            now = max(self._clock(), oldest + 60)
            self._expire(now)
        self._sent.append((now, requests, tokens))

    def _expire(self, now: float) -> None:
        while self._sent and self._sent[0][0] <= now - 60:
            self._sent.popleft()

    def _fits(self, requests: int, tokens: int) -> bool:
        return (
            sum(sent[1] for sent in self._sent) + requests <= self.requests
            and sum(sent[2] for sent in self._sent) + tokens <= self.tokens
        )


class GeminiEmbeddings:
    """Gemini embedding models through the Gemini API's batchEmbedContents."""

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    # gemini-embedding-2 takes no task type; the task goes in the text instead.
    TASK_PREFIX = "task: sentence similarity | query: "
    # 90% of the free tier's per-minute limits (100 requests, 30k tokens).
    REQUESTS_PER_MINUTE = 90
    TOKENS_PER_MINUTE = 27_000

    def __init__(self, model: str, dims: int, api_key: str, **client_options: Any) -> None:
        self.model = model
        self.dims = dims
        self.model_id = f"google:{model}@{dims}"
        self._window = MinuteWindow(
            self.REQUESTS_PER_MINUTE, self.TOKENS_PER_MINUTE, sleep=client_options.get("sleep", time.sleep)
        )
        self._client = ApiClient(
            service="google_embeddings",
            base_url=self.BASE_URL,
            headers={"x-goog-api-key": api_key},
            timeout=120.0,
            max_attempts=ATTEMPTS,
            first_wait=FIRST_WAIT,
            **client_options,
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for batch in self._batches([self.TASK_PREFIX + text for text in texts]):
            self._window.admit(len(batch), sum(_tokens(text) for text in batch))
            body = {"requests": [
                {"model": f"models/{self.model}", "content": {"parts": [{"text": text}]},
                 "output_dimensionality": self.dims}
                for text in batch
            ]}
            try:
                response = self._client.post(f"/models/{self.model}:batchEmbedContents", json=body)
            except RetryableStatus as exc:
                raise EmbeddingUnavailable(
                    f"HTTP {exc.response.status_code} after retries: {error_text(exc.response)}",
                    lasting=exc.response.status_code == 429 and daily_quota(exc.response),
                ) from exc
            except httpx.TransportError as exc:
                raise EmbeddingUnavailable(f"{type(exc).__name__} after retries") from exc
            rows += [embedding["values"] for embedding in response.json()["embeddings"]]
        return _unit(np.array(rows, dtype=np.float32).reshape(len(texts), self.dims))

    def _batches(self, texts: list[str]) -> list[list[str]]:
        """Up to BATCH texts each, and few enough tokens to fit one minute's budget."""
        size = min(BATCH, self.REQUESTS_PER_MINUTE)
        batches: list[list[str]] = []
        tokens = 0
        for text in texts:
            if not batches or len(batches[-1]) == size or tokens + _tokens(text) > self.TOKENS_PER_MINUTE:
                batches.append([])
                tokens = 0
            batches[-1].append(text)
            tokens += _tokens(text)
        return batches

    def close(self) -> None:
        self._client.close()


def _tokens(text: str) -> int:
    """A generous token estimate, for pacing: real counts run nearer 4 characters a token."""
    return len(text) // 3 + 1


def _unit(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


def text_embedder(settings: Settings) -> TextEmbedder | None:
    """The embedding model ``RA_EMBEDDING`` names, or None when it is unset."""
    if settings.embedding is None:
        return None
    provider, _, model = settings.embedding.strip().partition(":")
    if provider != "google":
        raise ValueError(f"Unknown embedding provider {provider!r} (known: google).")
    if settings.google_api_key is None:
        raise ValueError("No API key for google. Set RA_GOOGLE_API_KEY in .env.")
    return GeminiEmbeddings(model, settings.embedding_dims, settings.google_api_key.get_secret_value())


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """Vectors kept in SQLite; only texts never embedded before are sent."""

    def __init__(self, db: Database, embedder: TextEmbedder) -> None:
        self.db = db
        self.embedder = embedder

    def vectors(self, texts: list[str]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Vectors for as many of ``texts`` as possible, and what happened: how many were
        kept from earlier, how many were sent, and why any are missing."""
        wanted = {text_hash(text): text for text in dict.fromkeys(texts)}
        found: dict[str, np.ndarray] = {}
        hashes = list(wanted)
        for start in range(0, len(hashes), 500):
            part = hashes[start : start + 500]
            rows = self.db.query(
                f"SELECT text_hash, vector FROM embeddings WHERE model = ? AND text_hash IN ({','.join('?' * len(part))})",
                (self.embedder.model_id, *part),
            )
            for row in rows:
                found[wanted[row["text_hash"]]] = np.frombuffer(row["vector"], dtype=np.float32)
        report: dict[str, Any] = {"model": self.embedder.model_id, "reused": len(found), "sent": 0}
        missing = [text for text in wanted.values() if text not in found]
        for start in range(0, len(missing), BATCH):
            batch = missing[start : start + BATCH]
            try:
                vectors = self.embedder.embed(batch)
            except EmbeddingUnavailable as exc:
                report["unavailable"] = str(exc)
                break
            now = utc_now()
            with self.db.transaction() as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO embeddings (model, text_hash, vector, created_at) VALUES (?, ?, ?, ?)",
                    [(self.embedder.model_id, text_hash(text), vector.astype(np.float32).tobytes(), now)
                     for text, vector in zip(batch, vectors)],
                )
            found.update(zip(batch, vectors))
            report["sent"] += len(batch)
        report["missing"] = len(wanted) - len(found)
        count("embedding", reused=report["reused"], asked=len(missing))
        return found, report
