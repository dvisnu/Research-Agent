"""The OpenAI-compatible providers, against a mock HTTP transport."""

import json

import httpx
import pytest
from pydantic import BaseModel

from research_agent.llm import LLM, LLMConfigError, LLMError, LLMUnavailable, Message, Prompt, Role
from research_agent.llm_providers import ENDPOINTS, ChatCompletions, inline_refs


class Inner(BaseModel):
    name: str


class Outer(BaseModel):
    items: list[Inner]
    best: Inner | None = None


def _reply(content: str, **usage) -> httpx.Response:
    return httpx.Response(200, json={
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 7, **usage},
    })


def _provider(name: str, handler) -> tuple[ChatCompletions, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    provider = ChatCompletions(
        ENDPOINTS[name], "secret-key", transport=httpx.MockTransport(record), sleep=lambda seconds: None
    )
    return provider, requests


def _complete(provider: ChatCompletions):
    return provider.complete(
        model="m", system="Be brief.", messages=[Message("user", "Hi")], json_schema=Outer.model_json_schema()
    )


def test_google_request_carries_the_system_prompt_key_and_schema():
    provider, requests = _provider("google", lambda r: _reply('{"items": []}'))
    completion = _complete(provider)

    request = requests[0]
    assert str(request.url) == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    assert request.headers["Authorization"] == "Bearer secret-key"
    body = json.loads(request.content)
    assert body["messages"] == [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]
    schema = body["response_format"]["json_schema"]["schema"]
    assert "$defs" not in json.dumps(schema)  # references are inlined
    assert schema["properties"]["items"]["items"]["properties"]["name"]["type"] == "string"
    assert completion.text == '{"items": []}'
    assert completion.usage == {"input_tokens": 12, "output_tokens": 7}


def test_openrouter_sends_no_response_format_and_reports_thinking_tokens():
    provider, requests = _provider(
        "openrouter", lambda r: _reply("{}", completion_tokens_details={"reasoning_tokens": 5})
    )
    completion = _complete(provider)
    assert "response_format" not in json.loads(requests[0].content)
    assert str(requests[0].url) == "https://openrouter.ai/api/v1/chat/completions"
    assert completion.usage["reasoning_tokens"] == 5


@pytest.mark.parametrize(("message", "lasting"), [
    ("Rate limit reached on tokens per day (TPD): Limit 200000", True),
    ("Rate limit exceeded: free-models-per-day", True),
    ("Quota exceeded for quota id GenerateRequestsPerDayPerProjectPerModel-FreeTier", True),
    ("Rate limit reached on tokens per minute (TPM): Limit 8000", False),
])
def test_a_429_that_outlasts_the_retries_makes_the_model_unavailable(message, lasting):
    provider, requests = _provider("groq", lambda r: httpx.Response(429, json={"error": {"message": message}}))
    with pytest.raises(LLMUnavailable, match="HTTP 429 after retries") as caught:
        _complete(provider)
    assert len(requests) == 4
    assert caught.value.lasting is lasting  # only a used-up daily quota rules the model out for the session


def test_an_upstream_error_inside_a_200_is_retried():
    replies = iter([
        httpx.Response(200, json={"error": {"code": 502, "message": "Service temporarily overloaded"}}),
        _reply('{"items": []}'),
    ])
    provider, requests = _provider("openrouter", lambda r: next(replies))
    assert _complete(provider).text == '{"items": []}'
    assert len(requests) == 2


def test_a_lasting_upstream_error_makes_the_model_unavailable_for_now():
    provider, requests = _provider(
        "openrouter", lambda r: httpx.Response(200, json={"error": {"message": "Service temporarily overloaded"}})
    )
    with pytest.raises(LLMUnavailable, match="overloaded") as caught:
        _complete(provider)
    assert len(requests) == 3 and not caught.value.lasting


def test_a_rejected_key_names_the_setting():
    provider, requests = _provider("google", lambda r: httpx.Response(401, json={"error": {"message": "bad key"}}))
    with pytest.raises(LLMError, match="RA_GOOGLE_API_KEY") as caught:
        _complete(provider)
    assert not isinstance(caught.value, LLMUnavailable)
    assert len(requests) == 1  # not retried


def test_a_missing_key_names_the_setting(settings):
    llm = LLM(settings.model_copy(update={"llm_reasoning": "openrouter:nvidia/x:free"}))
    with pytest.raises(LLMConfigError, match="RA_OPENROUTER_API_KEY"):
        llm.structured(Role.REASONING, Prompt("p", 1, "s", "u"), Inner)


def test_inline_refs_keeps_sibling_keywords():
    schema = {"$defs": {"A": {"type": "object"}}, "properties": {"a": {"$ref": "#/$defs/A", "description": "d"}}}
    assert inline_refs(schema) == {"properties": {"a": {"type": "object", "description": "d"}}}
