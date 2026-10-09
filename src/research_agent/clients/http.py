"""Shared HTTP layer for the literature APIs: retry with backoff, per-host pacing, tracing.

Responses are not cached.
"""

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any, Protocol, Self

import httpx
from filelock import FileLock
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from research_agent.storage import utc_now
from research_agent.trace import child_event

USER_AGENT = "research-agent/0.1"
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
# Never written to the trace.
_PRIVATE_PARAMS = frozenset({"email", "mailto", "api_key"})


class RetryableStatus(httpx.HTTPStatusError):
    def __init__(self, response: httpx.Response) -> None:
        super().__init__(
            f"HTTP {response.status_code} from {response.request.url.host}",
            request=response.request,
            response=response,
        )
        self.retry_after = _retry_after_seconds(response)


def _retry_after_seconds(response: httpx.Response) -> float | None:
    try:
        return max(0.0, float(response.headers["Retry-After"]))
    except (KeyError, ValueError):  # absent, or the HTTP-date form
        return None


class Pacer(Protocol):
    def slot(self) -> AbstractContextManager[None]:
        """Wrap one request: block until it may be sent, then send it inside."""
        ...


class _Pacer:
    """Keeps at least ``interval`` seconds between request starts within this process."""

    def __init__(self, interval: float, sleep: Callable[[float], None]) -> None:
        self.interval = interval
        self._sleep = sleep
        self._next = 0.0
        self._lock = threading.Lock()

    @contextmanager
    def slot(self) -> Iterator[None]:
        with self._lock:
            now = time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + self.interval
        if delay > 0:
            self._sleep(delay)
        yield


class FilePacer:
    """One request at a time across every process and thread on this machine,
    each sent at least ``interval`` seconds after the previous one finished.

    A file lock is held for the whole request, and the time the last request
    finished is kept in ``path``. Spacing from the previous *response* rather
    than the previous send means two requests can never reach the server less
    than ``interval`` apart, however long either spends in transit.
    """

    def __init__(self, path: Path, interval: float, sleep: Callable[[float], None]) -> None:
        self.path = path
        self.interval = interval
        self._sleep = sleep
        self._lock = FileLock(f"{path}.lock")

    @contextmanager
    def slot(self) -> Iterator[None]:
        with self._lock:
            try:
                last_end = float(self.path.read_text(encoding="utf-8"))
            except (FileNotFoundError, ValueError):
                last_end = 0.0
            delay = last_end + self.interval - time.time()
            if delay > 0:
                self._sleep(min(delay, self.interval))
            try:
                yield
            finally:
                self.path.write_text(repr(time.time()), encoding="utf-8")


class ApiClient:
    """One literature API: its host, its pacing, and its name in the trace."""

    def __init__(
        self,
        *,
        service: str,
        base_url: str,
        min_interval: float = 0.0,
        headers: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
        timeout: float = 30.0,
        max_attempts: int = 5,
        max_wait: float = 60.0,
        first_wait: float = 1.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        pacer: Pacer | None = None,
    ) -> None:
        self.service = service
        self._http = httpx.Client(
            base_url=base_url,
            headers={"User-Agent": USER_AGENT, **(headers or {})},
            params=params,
            timeout=timeout,
            transport=transport,
            follow_redirects=True,
        )
        self._pacer = pacer or _Pacer(min_interval, sleep)
        self._sleep = sleep
        self._max_attempts = max_attempts
        self._max_wait = max_wait
        self._backoff = wait_exponential_jitter(initial=first_wait, max=max_wait)

    def get(
        self, path: str, *, params: dict[str, Any] | None = None, allow_404: bool = False
    ) -> httpx.Response | None:
        return self._request("GET", path, params=params, allow_404=allow_404)

    def post(
        self, path: str, *, json: Any, params: dict[str, Any] | None = None
    ) -> httpx.Response:
        response = self._request("POST", path, params=params, json=json)
        assert response is not None
        return response

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any = None,
        allow_404: bool = False,
    ) -> httpx.Response | None:
        public_params = {k: v for k, v in (params or {}).items() if k not in _PRIVATE_PARAMS}
        with child_event("api_call", agent={"kind": "api", "name": self.service}) as event:
            event.detail.update(method=method, path=path, params=public_params)
            retrying = Retrying(
                stop=stop_after_attempt(self._max_attempts),
                wait=self._wait,
                retry=retry_if_exception(self._worth_retrying),
                sleep=self._sleep,
                reraise=True,
            )
            sends: list[dict[str, Any]] = []
            event.detail["sends"] = sends  # one entry per attempt: when it was sent, and the outcome
            for attempt in retrying:
                with attempt:
                    event.detail["attempts"] = attempt.retry_state.attempt_number
                    with self._pacer.slot():
                        sends.append({"sent_at": utc_now()})
                        try:
                            response = self._http.request(method, path, params=params, json=json)
                        except httpx.TransportError as exc:
                            sends[-1]["error"] = type(exc).__name__
                            raise
                        sends[-1]["done_at"] = utc_now()
                    sends[-1]["status"] = event.detail["status"] = response.status_code
                    if response.status_code in RETRY_STATUSES:
                        raise RetryableStatus(response)
            if allow_404 and response.status_code == 404:
                return None
            response.raise_for_status()
            return response

    def _worth_retrying(self, exc: BaseException) -> bool:
        """Retry transport errors and 429/5xx, unless the server asks for a wait longer
        than ``max_wait`` (a used-up daily budget, say): retrying sooner can't succeed."""
        if isinstance(exc, RetryableStatus):
            return exc.retry_after is None or exc.retry_after <= self._max_wait
        return isinstance(exc, httpx.TransportError)

    def _wait(self, retry_state: Any) -> float:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        if isinstance(exc, RetryableStatus) and exc.retry_after is not None:
            return min(exc.retry_after, self._max_wait)
        return self._backoff(retry_state)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
