import pytest
from pydantic import BaseModel, ValidationError

from research_agent import llm
from research_agent.config import Settings
from research_agent.llm import LLM, Completion, LLMConfigError, LLMOutputError, LLMUnavailable, MAX_STRIKES, Prompt, Role


class Brief(BaseModel):
    problem: str
    keywords: list[str]


PROMPT = Prompt(name="brief", version=1, system="You write research briefs.", user="Question: ...")


class FakeProvider:
    """Replies with queued texts and records what it was sent."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []

    def complete(self, *, model, system, messages, json_schema):
        self.calls.append({"model": model, "system": system, "messages": messages})
        return Completion(self.replies.pop(0), usage={"input_tokens": 10, "output_tokens": 5})


@pytest.fixture
def fake(monkeypatch, settings):
    provider = FakeProvider([])
    monkeypatch.setitem(llm._PROVIDERS, "fake", lambda _settings: provider)
    return provider


def _llm(settings, **roles) -> LLM:
    return LLM(settings.model_copy(update=roles))


def test_valid_reply_is_parsed(settings, fake):
    fake.replies = ['{"problem": "p", "keywords": ["a", "b"]}']
    brief = _llm(settings, llm_reasoning="fake:model-r").structured(Role.REASONING, PROMPT, Brief)
    assert brief == Brief(problem="p", keywords=["a", "b"])
    assert fake.calls[0]["model"] == "model-r"
    # The prompt's system text, then the schema the reply must match.
    assert fake.calls[0]["system"].startswith(PROMPT.system + "\n\nReply with only a JSON object")
    assert '"required": ["problem", "keywords"]' in fake.calls[0]["system"]


def test_invalid_reply_is_sent_back_once(settings, fake):
    fake.replies = ['{"problem": "p"}', '{"problem": "p", "keywords": []}']
    brief = _llm(settings, llm_reasoning="fake:m").structured(Role.REASONING, PROMPT, Brief)
    assert brief.keywords == []
    retry_messages = fake.calls[1]["messages"]
    assert [m.role for m in retry_messages] == ["user", "assistant", "user"]
    assert "keywords" in retry_messages[-1].content


def test_gives_up_after_max_attempts(settings, fake):
    fake.replies = ["not json", "still not json"]
    with pytest.raises(LLMOutputError, match="brief@v1"):
        _llm(settings, llm_reasoning="fake:m").structured(Role.REASONING, PROMPT, Brief)


def test_unset_role_names_the_variable_to_set(settings):
    with pytest.raises(LLMConfigError, match="RA_LLM_EXTRACTION"):
        LLM(settings).structured(Role.EXTRACTION, PROMPT, Brief)


def test_unknown_provider_is_reported(settings):
    with pytest.raises(LLMConfigError, match="nobody"):
        _llm(settings, llm_reasoning="nobody:m").structured(Role.REASONING, PROMPT, Brief)


def test_call_inside_a_step_is_traced(settings, fake, recorder):
    fake.replies = ["{}", '{"problem": "p", "keywords": ["k"]}']
    with recorder.step("run_test", "brief") as step:
        _llm(settings, llm_reasoning="fake:m").structured(Role.REASONING, PROMPT, Brief)

    call = next(e for e in recorder.events("run_test") if e["activity_type"] == "llm_call")
    assert call["parent_event_id"] == step.event_id
    assert call["prompt_version"] == "brief@v1"
    assert call["agent"] == {"kind": "llm", "role": "reasoning", "provider": "fake", "model": "m"}
    assert call["detail"]["attempts"] == 2
    assert call["detail"]["usage"] == {"input_tokens": 20, "output_tokens": 10}


@pytest.mark.parametrize("spec", ["no-colon", ":model", "provider:"])
def test_malformed_model_spec_is_rejected(spec):
    with pytest.raises(ValidationError, match="provider:model"):
        Settings(_env_file=None, llm_reasoning=spec)


def test_verifier_must_differ_from_reasoning():
    with pytest.raises(ValidationError, match="verifier"):
        Settings(_env_file=None, llm_reasoning="a:m", llm_verifier="a:m")
    with pytest.raises(ValidationError, match="verifier"):
        Settings(_env_file=None, llm_reasoning="a:m, b:m", llm_verifier="b:m")
    Settings(_env_file=None, llm_reasoning="a:m", llm_verifier="b:m")


def test_a_role_can_list_fallback_models():
    settings = Settings(_env_file=None, llm_extraction="google:gemini-x, openrouter:nvidia/model:free")
    assert LLM(settings).resolve(Role.EXTRACTION) == [("google", "gemini-x"), ("openrouter", "nvidia/model:free")]
    with pytest.raises(ValidationError, match="provider:model"):
        Settings(_env_file=None, llm_extraction="google:gemini-x, no-colon")


def test_fenced_or_wrapped_json_is_accepted(settings, fake):
    fake.replies = ['Here it is:\n```json\n{"problem": "p", "keywords": []}\n```']
    assert _llm(settings, llm_reasoning="fake:m").structured(Role.REASONING, PROMPT, Brief).problem == "p"


def test_an_empty_reply_is_not_echoed_back(settings, fake):
    fake.replies = ["", '{"problem": "p", "keywords": []}']
    _llm(settings, llm_reasoning="fake:m").structured(Role.REASONING, PROMPT, Brief)
    assert [m.role for m in fake.calls[1]["messages"]] == ["user", "user"]
    assert "empty" in fake.calls[1]["messages"][-1].content


class QuotaProvider:
    """Unavailable for the models in ``exhausted`` (lastingly, unless ``lasting`` is
    false); otherwise replies validly."""

    def __init__(self, exhausted: set[str], lasting: bool = True) -> None:
        self.exhausted = exhausted
        self.lasting = lasting
        self.calls: list[str] = []

    def complete(self, *, model, system, messages, json_schema):
        self.calls.append(model)
        if model in self.exhausted:
            raise LLMUnavailable("HTTP 429 after retries", lasting=self.lasting)
        return Completion('{"problem": "p", "keywords": []}')


def test_a_passing_outage_moves_only_that_call_to_the_next_model(settings):
    provider = QuotaProvider(exhausted={"first"}, lasting=False)
    model = LLM(settings.model_copy(update={"llm_extraction": "q:first,q:second"}), providers={"q": provider})
    model.structured(Role.EXTRACTION, PROMPT, Brief)
    model.structured(Role.EXTRACTION, PROMPT, Brief)
    assert provider.calls == ["first", "second", "first", "second"]


def test_a_model_failing_call_after_call_is_dropped_for_the_session(settings):
    provider = QuotaProvider(exhausted={"first"}, lasting=False)
    model = LLM(settings.model_copy(update={"llm_extraction": "q:first,q:second"}), providers={"q": provider})
    for _ in range(MAX_STRIKES + 2):
        model.structured(Role.EXTRACTION, PROMPT, Brief)
    assert provider.calls == ["first", "second"] * MAX_STRIKES + ["second", "second"]

    # A success in between starts the count again.
    flaky = QuotaProvider(exhausted=set(), lasting=False)
    model = LLM(settings.model_copy(update={"llm_extraction": "q:first,q:second"}), providers={"q": flaky})
    for down in (True, True, False, True, True):
        flaky.exhausted = {"first"} if down else set()
        model.structured(Role.EXTRACTION, PROMPT, Brief)
    assert flaky.calls.count("first") == 5


def test_an_unavailable_model_hands_over_to_the_next_for_the_session(settings, recorder):
    provider = QuotaProvider(exhausted={"first"})
    model = LLM(settings.model_copy(update={"llm_extraction": "q:first,q:second"}), providers={"q": provider})
    with recorder.step("run_test", "rank"):
        model.structured(Role.EXTRACTION, PROMPT, Brief)
        model.structured(Role.EXTRACTION, PROMPT, Brief)

    assert provider.calls == ["first", "second", "second"]  # "first" is not retried later in the session
    calls = [e for e in recorder.events("run_test") if e["activity_type"] == "llm_call"]
    assert [c["agent"]["model"] for c in calls] == ["second", "second"]
    assert calls[0]["detail"]["unavailable"] == ["q:first: HTTP 429 after retries"]
    assert calls[1]["detail"]["unavailable"] == ["q:first: earlier in this session"]


def test_a_role_with_every_model_unavailable_fails(settings):
    provider = QuotaProvider(exhausted={"first", "second"})
    model = LLM(settings.model_copy(update={"llm_extraction": "q:first,q:second"}), providers={"q": provider})
    with pytest.raises(LLMUnavailable, match="No model for the extraction role"):
        model.structured(Role.EXTRACTION, PROMPT, Brief)
