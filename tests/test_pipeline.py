import json
from collections import Counter

import pytest

from research_agent.pipeline import (
    STAGE_NAMES,
    CannotResume,
    RunError,
    RunOptions,
    resume_run,
    start_run,
)
from research_agent.storage import Database
from research_agent.trace import TraceRecorder

DEFAULT_PATH = [name for name in STAGE_NAMES if name != "targeted_search"]


def _steps(settings, run_id):
    with Database(settings.db_path) as db:
        return [e for e in TraceRecorder(db).events(run_id) if e["activity_type"] == "step"]


def _run_status(settings, run_id):
    with Database(settings.db_path) as db:
        return db.get_run(run_id)["status"]


def test_run_passes_every_stage_and_writes_an_empty_report(settings, services):
    result = start_run(settings, "How can RAG handle conflicting documents?", services=services)

    assert result.stages_run == DEFAULT_PATH
    assert [e["step"] for e in _steps(settings, result.run_id)] == DEFAULT_PATH
    assert all(e["status"] == "ok" for e in _steps(settings, result.run_id))
    assert _run_status(settings, result.run_id) == "completed"

    report = json.loads((result.report_dir / "report.json").read_text(encoding="utf-8"))
    assert report["question"] == "How can RAG handle conflicting documents?"
    assert report["novel_evidence_report"]["proposed_idea"] is None
    assert report["experiment_plan"]["baselines"] == []
    markdown = (result.report_dir / "report.md").read_text(encoding="utf-8")
    assert "## Novel Evidence Report" in markdown and "## Experiment Plan" in markdown


def test_brief_uses_the_question(settings, services):
    result = start_run(settings, "q", services=services)
    brief = _steps(settings, result.run_id)[0]
    assert brief["used"] == [f"question:{result.run_id}"]


def test_failed_run_resumes_from_the_failed_stage(settings, services):
    with pytest.raises(RunError) as failure:
        start_run(settings, "q", RunOptions(fail_at="gaps"), services)
    run_id = failure.value.run_id
    assert _run_status(settings, run_id) == "failed"
    assert _steps(settings, run_id)[-1]["status"] == "error"

    result = resume_run(settings, run_id, services)

    assert result.stages_run == DEFAULT_PATH
    assert _run_status(settings, run_id) == "completed"
    counts = Counter(e["step"] for e in _steps(settings, run_id))
    assert counts["gaps"] == 2  # the failed attempt and the resumed one
    assert all(counts[name] == 1 for name in DEFAULT_PATH if name != "gaps")


def test_completed_or_unknown_runs_cannot_be_resumed(settings, services):
    result = start_run(settings, "q", services=services)
    with pytest.raises(CannotResume, match="already completed"):
        resume_run(settings, result.run_id, services)
    with pytest.raises(CannotResume, match="No run"):
        resume_run(settings, "run_missing", services)


def test_forced_loops_take_each_loop_once(settings, services):
    result = start_run(settings, "q", RunOptions(force_loops=True), services)
    counts = Counter(result.stages_run)
    assert counts["ideate"] == counts["novelty_search"] == counts["novelty_assess"] == 2
    assert counts["plan"] == counts["critique"] == 2
    assert counts["targeted_search"] == 1
    assert result.stages_run[-2:] == ["verify", "report"]
