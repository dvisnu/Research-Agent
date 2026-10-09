import pytest

from research_agent.trace import child_event, ref


def test_step_records_entities(recorder):
    with recorder.step("run_test", "search", used=[ref("question", "run_test")]) as event:
        event.generated.append(ref("paper", "p_1"))
        event.drop(ref("paper", "p_2"), "below relevance threshold")

    [row] = recorder.events("run_test")
    assert row["step"] == "search"
    assert row["activity_type"] == "step"
    assert row["attempt"] == 1
    assert row["status"] == "ok"
    assert row["used"] == ["question:run_test"]
    assert row["generated"] == ["paper:p_1"]
    assert row["dropped"] == [{"ref": "paper:p_2", "reason": "below relevance threshold"}]


def test_child_events_attach_to_the_current_step(recorder):
    with recorder.step("run_test", "search") as step:
        with child_event("api_call", agent={"kind": "api", "name": "arxiv"}) as call:
            call.detail["status"] = 200

    events = {e["activity_type"]: e for e in recorder.events("run_test")}
    assert events["api_call"]["parent_event_id"] == step.event_id
    assert events["api_call"]["step"] == "search"
    assert events["api_call"]["attempt"] is None
    assert events["api_call"]["detail"] == {"status": 200}


def test_child_event_outside_a_run_is_not_recorded(db, recorder):
    with child_event("api_call", agent={"kind": "api", "name": "arxiv"}):
        pass
    assert db.query("SELECT COUNT(*) FROM trace_events")[0][0] == 0


def test_failed_step_is_recorded_and_reraised(recorder):
    with pytest.raises(ValueError):
        with recorder.step("run_test", "rank"):
            raise ValueError("boom")
    [row] = recorder.events("run_test")
    assert row["status"] == "error"
    assert row["error"] == "ValueError: boom"


def test_attempts_count_executions_of_a_step(recorder):
    for _ in range(3):
        with recorder.step("run_test", "ideate"):
            pass
    with recorder.step("run_test", "plan"):
        pass
    attempts = [(e["step"], e["attempt"], e["seq"]) for e in recorder.events("run_test")]
    assert attempts == [("ideate", 1, 1), ("ideate", 2, 2), ("ideate", 3, 3), ("plan", 1, 4)]
