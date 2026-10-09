import subprocess
import sys
import time

import httpx
import pytest

from research_agent.clients import semantic_scholar
from research_agent.clients.http import ApiClient, FilePacer


def _client(handler, sleeps: list[float], **kwargs) -> ApiClient:
    return ApiClient(
        service="test",
        base_url="https://api.example.org",
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
        **kwargs,
    )


def test_429_is_retried_after_the_retry_after_delay(recorder):
    responses = iter([httpx.Response(429, headers={"Retry-After": "2"}), httpx.Response(200, json={})])
    sleeps: list[float] = []
    with recorder.step("run_test", "search"):
        response = _client(lambda request: next(responses), sleeps).get("/x")
    assert response.status_code == 200
    assert sleeps == [2.0]
    call = next(e for e in recorder.events("run_test") if e["activity_type"] == "api_call")
    assert call["detail"]["attempts"] == 2
    assert call["detail"]["status"] == 200
    assert call["agent"] == {"kind": "api", "name": "test"}


def test_a_retry_after_longer_than_max_wait_is_not_waited_for():
    sleeps: list[float] = []
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(429, headers={"Retry-After": "27291"})

    with pytest.raises(httpx.HTTPStatusError):
        _client(handler, sleeps).get("/x")
    assert len(calls) == 1 and sleeps == []


def test_server_errors_are_retried_until_attempts_run_out():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503)

    with pytest.raises(httpx.HTTPStatusError):
        _client(handler, [], max_attempts=3).get("/x")
    assert len(calls) == 3


def test_transport_errors_are_retried():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200)

    assert _client(handler, []).get("/x").status_code == 200
    assert len(calls) == 2


def test_404_is_not_retried():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(404)

    client = _client(handler, [])
    assert client.get("/x", allow_404=True) is None
    with pytest.raises(httpx.HTTPStatusError):
        client.get("/x")
    assert len(calls) == 2


def test_requests_are_paced():
    sleeps: list[float] = []
    client = _client(lambda request: httpx.Response(200), sleeps, min_interval=3.0)
    client.get("/a")
    client.get("/b")
    assert len(sleeps) == 1 and 2.5 < sleeps[0] <= 3.0


def test_private_params_stay_out_of_the_trace(recorder):
    with recorder.step("run_test", "search"):
        _client(lambda request: httpx.Response(200), []).get(
            "/x", params={"q": "lora", "email": "me@example.org"}
        )
    call = next(e for e in recorder.events("run_test") if e["activity_type"] == "api_call")
    assert call["detail"]["params"] == {"q": "lora"}


def test_file_pacer_spaces_separate_instances(tmp_path):
    # Two instances on one file stand in for two clients or processes.
    sleeps: list[float] = []
    first = FilePacer(tmp_path / "rate", 1.1, sleeps.append)
    second = FilePacer(tmp_path / "rate", 1.1, sleeps.append)
    with first.slot():
        pass
    with second.slot():
        pass
    assert len(sleeps) == 1 and 1.0 < sleeps[0] <= 1.1


_WORKER = """
import sys, time
from pathlib import Path
from research_agent.clients.http import FilePacer
pacer = FilePacer(Path(sys.argv[1]), 0.3, time.sleep)
for _ in range(3):
    with pacer.slot():
        start = time.time()
        time.sleep(0.1)  # the request in flight
        print(start, time.time(), flush=True)
"""


def test_file_pacer_serialises_requests_across_processes(tmp_path):
    workers = [
        subprocess.Popen(
            [sys.executable, "-c", _WORKER, str(tmp_path / "rate")], stdout=subprocess.PIPE, text=True
        )
        for _ in range(3)
    ]
    output = "".join(w.communicate(timeout=60)[0] for w in workers)
    requests = sorted(tuple(map(float, line.split())) for line in output.splitlines())
    assert len(requests) == 9
    # Each request starts at least the interval after the previous one finished.
    gaps = [start - prev_end for (_, prev_end), (start, _) in zip(requests, requests[1:])]
    assert min(gaps) >= 0.29, gaps


def test_semantic_scholar_is_paced_machine_wide(settings, tmp_path):
    sleeps: list[float] = []
    clients = [
        semantic_scholar.SemanticScholar(
            settings,
            rate_file=tmp_path / "s2",
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"data": []})),
            sleep=sleeps.append,
        )
        for _ in range(2)
    ]
    clients[0].search("a")
    clients[1].search("b")
    assert semantic_scholar.MIN_INTERVAL >= 1.0
    assert len(sleeps) == 1 and sleeps[0] > 1.0
