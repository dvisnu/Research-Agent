import re
import tomllib
from pathlib import Path

from typer.testing import CliRunner

from research_agent.cli import app
from research_agent.pipeline import runner as pipeline_runner
from fakes import fake_services

TOPICS = Path(__file__).parents[1] / "data" / "dev_topics.toml"


def test_cli_run_then_inspect(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no .env here, and var/ lands in tmp_path
    monkeypatch.setattr(pipeline_runner, "default_services", lambda settings: fake_services(settings, tmp_path))
    runner = CliRunner()

    result = runner.invoke(app, ["run", "Does [bracketed] text survive?"])
    assert result.exit_code == 0, result.output
    assert "Novel Evidence Report" in result.output
    run_id = re.search(r"Run (run_\w+) completed", result.output)[1]

    trace = runner.invoke(app, ["inspect", run_id])
    assert trace.exit_code == 0, trace.output
    assert "brief" in trace.output and "report" in trace.output

    listing = runner.invoke(app, ["inspect"])
    assert run_id in listing.output


def test_cli_rejects_unknown_fail_stage(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["run", "q", "--fail-at", "nope"])
    assert result.exit_code == 2


def test_a_key_in_a_role_variable_is_not_echoed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RA_LLM_REASONING", "sk-or-v1-secret")
    result = CliRunner().invoke(app, ["run", "q"])
    assert result.exit_code == 2
    assert "RA_LLM_REASONING" in result.output and "secret" not in result.output


def test_dev_topics_are_well_formed():
    topics = tomllib.loads(TOPICS.read_text(encoding="utf-8"))["topic"]
    assert 10 <= len(topics) <= 20
    assert len({t["id"] for t in topics}) == len(topics)
    for topic in topics:
        assert topic["input"] in {"question", "idea"}
        assert topic["text"].strip()
        assert len(topic["key_papers"]) >= 3
        assert all(p["arxiv"] and p["title"] for p in topic["key_papers"])
