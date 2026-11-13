"""The web UI's server: what it shows of a run, how it starts and stops runs, and whom it answers."""

import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from research_agent.storage import Database
from research_agent.trace import TraceRecorder, child_event
from research_agent.web import views
from research_agent.web.app import HEADER, create_app
from research_agent.web.jobs import Jobs
from test_resume import _run

# Stands in for `python -m research_agent`: notes its arguments, and a run keeps going a while.
FAKE_AGENT = """
import json, os, sys, time
with open(os.environ["FAKE_AGENT_OUT"], "a", encoding="utf-8") as out:
    out.write(json.dumps(sys.argv[1:]) + "\\n")
time.sleep(30 if sys.argv[1] == "run" else 0)
"""


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    """A run of the whole pipeline on fakes, through to a verified plan."""
    settings, run_id, _ = _run(tmp_path_factory.mktemp("web"), None)
    return settings, run_id


def _client(settings, jobs=None) -> TestClient:
    return TestClient(create_app(settings, jobs), base_url="http://127.0.0.1")


def test_the_ui_shows_a_finished_run_down_to_its_sentences(finished):
    settings, run_id = finished
    client = _client(settings)

    [run] = client.get("/api/runs").json()
    assert run["run_id"] == run_id and run["state"] == "completed" and run["best_idea"]["title"]
    assert run["coverage"]["elements"] > 0 and not run["started_here"]

    progress = client.get(f"/api/runs/{run_id}/progress").json()
    assert progress["current"] is None
    stages = {s["name"]: s for s in progress["stages"]}
    assert stages["extract"]["status"] == stages["verify"]["status"] == "done" and stages["extract"]["llm_calls"]

    report = client.get(f"/api/runs/{run_id}/report").json()
    assert report["final"] and report["research_gaps"] and report["ideas"]
    papers = client.get(f"/api/runs/{run_id}/papers").json()
    assert {p["read_level"] for p in papers} == {"full", "abstract"} and papers[0]["rank"] == 1

    # A plan element traces through its facts to sentences, sections and papers.
    plan = report["experiment_plan"]
    element = next(e for e in [plan["hypothesis"], *plan["method"], *plan["metrics"]] if e and e["sources"])
    chain = client.get(f"/api/runs/{run_id}/plan/elements/{element['element_id']}").json()
    evidence = [e for fact in chain["facts"] for e in fact["evidence"]]
    assert evidence and all(e["sentence"]["paper"]["title"] for e in evidence)
    sentence = client.get(f"/api/sentences/{evidence[0]['sentence_id']}").json()
    assert sentence["text"] in [s["text"] for s in sentence["context"]]

    graph = client.get(f"/api/runs/{run_id}/graph").json()
    node = next(n for n in graph["nodes"] if n["evidence"])
    found = client.get(f"/api/graph/node/{node['id']}").json()
    assert found["evidence"][0]["sentence"]["text"] and found["links"]
    assert client.get(f"/api/runs/{run_id}/cost").json()["stages"]

    details = client.get(f"/api/runs/{run_id}").json()
    assert {"report.md", "report.json", "provenance.json"} <= set(details["files"])
    assert client.get(f"/api/runs/{run_id}/files/report.md").text.startswith("#")
    assert client.get(f"/api/runs/{run_id}/files/agent.db").status_code == 404
    assert client.get("/api/runs/run_0123456789abcdef").status_code == 404
    assert client.post(f"/api/runs/{run_id}/resume", headers={HEADER: "1"}).status_code == 409  # it has completed
    assert client.get(f"/runs/{run_id}/plan").headers["content-type"].startswith("text/html")


def test_only_local_pages_can_use_it(finished):
    settings, run_id = finished
    client = _client(settings)
    # DNS rebinding: a page on another site, whose name now points at 127.0.0.1.
    assert client.get("/api/runs", headers={"host": "attacker.example:8000"}).status_code == 403
    # A form on another site can post, but cannot add the header.
    assert client.post(f"/api/runs/{run_id}/resume").status_code == 403
    assert client.post("/api/runs", data={"question": "q"}).status_code == 403


def test_an_older_runs_report_comes_in_the_current_shape(settings, db):
    # A Phase 1 run's report: no gaps, ideas or graph yet, and a plan whose parts are null.
    run_id = "run_0123456789abcdef"
    db.create_run(run_id, "q")
    db.set_run_status(run_id, "completed")
    (settings.runs_dir / run_id).mkdir(parents=True)
    plan = {"hypothesis": None, "method": None, "datasets": []}
    old = {"run_id": run_id, "question": "q", "generated_at": "2026-10-06", "experiment_plan": plan}
    (settings.runs_dir / run_id / "report.json").write_text(json.dumps(old), encoding="utf-8")

    report = _client(settings).get(f"/api/runs/{run_id}/report").json()
    assert report["final"] and report["ideas"] == report["research_gaps"] == report["related_work"] == []
    assert report["knowledge_graph"] is None and report["experiment_plan"]["method"] == []
    assert report["experiment_plan"]["provenance_coverage"] == {} and report["experiment_plan"]["hypothesis"] is None


def test_runs_start_in_a_child_process_one_at_a_time_and_stop_resumably(settings, tmp_path, monkeypatch):
    script = tmp_path / "fake_agent.py"
    script.write_text(FAKE_AGENT, encoding="utf-8")
    calls = tmp_path / "calls.jsonl"
    monkeypatch.setenv("FAKE_AGENT_OUT", str(calls))
    jobs = Jobs(settings, command=[sys.executable, str(script)])
    client = _client(settings, jobs)
    send = {HEADER: "1"}

    assert client.post("/api/runs", data={"question": "  "}, headers=send).status_code == 422
    bad = client.post("/api/runs", data={"question": "q"}, files={"uploads": ("notes.txt", b"x")}, headers=send)
    assert bad.status_code == 422

    started = client.post("/api/runs", headers=send, data={"question": "-Does LoRA  match full fine-tuning?",
                                                           "refresh": "true"},
                          files={"uploads": ("my paper.pdf", b"%PDF-1.4 test")})
    run_id = started.json()["run_id"]
    upload = settings.data_dir / "uploads" / run_id / "my_paper.pdf"
    assert upload.read_bytes() == b"%PDF-1.4 test"
    second = client.post("/api/runs", data={"question": "another"}, headers=send)
    assert second.status_code == 409 and run_id in second.json()["detail"]
    assert client.get("/api/status").json()["active"] == run_id

    with Database(settings.db_path) as db:  # the child would create the run; the fake doesn't
        db.create_run(run_id, "Does LoRA match full fine-tuning?")
    for _ in range(100):  # until the child has started and noted its arguments
        if calls.exists() and calls.read_text(encoding="utf-8"):
            break
        time.sleep(0.1)
    assert client.get(f"/api/runs/{run_id}").json()["state"] == "running"
    assert client.post(f"/api/runs/{run_id}/stop", headers=send).status_code == 200
    stopped = client.get(f"/api/runs/{run_id}").json()
    assert stopped["state"] == "failed" and stopped["error"] == "Stopped from the web UI."
    assert jobs.active() is None

    assert client.post(f"/api/runs/{run_id}/resume", headers=send).status_code == 200
    jobs.get(run_id).process.wait(timeout=30)
    run_args, resume_args = [json.loads(line) for line in calls.read_text(encoding="utf-8").splitlines()]
    # The question goes after "--", so one starting with "-" is not taken for an option.
    assert run_args == ["run", "--run-id", run_id, f"--upload={upload.resolve()}", "--refresh", "--",
                        "-Does LoRA match full fine-tuning?"]
    assert resume_args == ["resume", run_id]
    assert "=== run" in (settings.runs_dir / run_id / "console.log").read_text(encoding="utf-8")


def test_the_stage_running_now_is_the_one_whose_calls_have_no_step_yet(db, recorder):
    with recorder.step("run_test", "brief"):
        pass
    time.sleep(0.01)
    with recorder.step("run_test", "search"):
        with child_event("api_call", agent={"kind": "api", "name": "semantic_scholar"}):
            with child_event("api_call", agent={"kind": "api", "name": "inner"}):
                pass
        now = views.progress(db, "run_test")
    assert now["current"]["stage"] == "search" and now["current"]["calls"] == 2
    stages = {s["name"]: s["status"] for s in now["stages"]}
    assert stages["brief"] == "done" and stages["search"] == "running" and stages["rank"] == "pending"
    after = views.progress(db, "run_test")
    assert after["current"] is None and after["last_finished"] == "search"


@pytest.mark.parametrize(("error", "quota", "keys"), [
    ("LLMUnavailable: No model for the verifier role is available (groq:openai/gpt-oss-120b: HTTP 429 after retries:"
     " Rate limit reached ... tokens per day (TPD))", True, ["RA_GROQ_API_KEY"]),
    ("LLMUnavailable: No model for the extraction role is available (google:gemini-3.5-flash-lite: quota exhausted;"
     " google:gemini-3.1-flash-lite: quota exhausted)", True, ["RA_GOOGLE_API_KEY"]),
    ("UploadError: could not read uploaded file x.pdf", False, []),
])
def test_a_stopped_run_says_which_key_ran_out(error, quota, keys):
    found = views.diagnose(error)
    assert found["quota"] is quota and found["keys"] == keys


def test_without_a_built_app_the_page_says_how_to_build_it(settings, monkeypatch):
    monkeypatch.setattr("research_agent.web.app.STATIC", Path(settings.data_dir) / "no-build")
    page = _client(settings).get("/")
    assert "npm run build" in page.text
