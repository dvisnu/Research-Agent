"""LLM providers, all reached through OpenAI-compatible chat completions:
Google AI Studio (Gemini and Gemma), OpenRouter and Groq.

Each provider is paced to its free tier's requests per minute. The shared HTTP
layer retries 429 and 5xx with backoff long enough to clear a per-minute limit;
an upstream failure that OpenRouter reports inside a 200 is retried here. What
outlasts the retries is raised as LLMUnavailable, so the role falls back to its
next model; a used-up daily quota is marked lasting, so the role stops trying
that model for the rest of the session.
"""

import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from research_agent.clients.http import ApiClient, RetryableStatus
from research_agent.config import Settings
from research_agent.llm import (
    Completion,
    LLMConfigError,
    LLMError,
    LLMUnavailable,
    Message,
    ProviderFactory,
)

# Reasoning models can think for minutes before answering.
TIMEOUT = 600.0
# HTTP retries wait about 8, 16 and 32 s: enough for a per-minute limit to reset.
ATTEMPTS = 4
FIRST_WAIT = 8.0
# Waits before retrying an upstream failure reported inside a 200.
UPSTREAM_WAITS = (15.0, 45.0)
# How the providers word a used-up daily quota (Google: "...PerDay...", OpenRouter:
# "free-models-per-day", Groq: "tokens per day (TPD)").
_DAILY = re.compile(r"per.?day|daily|\bRPD\b|\bTPD\b", re.IGNORECASE)


@dataclass(frozen=True)
class Endpoint:
    name: str
    base_url: str
    key_setting: str  # the Settings field holding the API key
    min_interval: float  # seconds between requests: 60 / the free tier's requests per minute
    max_tokens: int  # reply budget, thinking included
    schema_output: bool  # also send the JSON schema as response_format, for constrained output


ENDPOINTS = {
    endpoint.name: endpoint
    for endpoint in (
        # Gemini Flash-Lite allows 15 requests a minute (Gemma 30).
        Endpoint("google", "https://generativelanguage.googleapis.com/v1beta/openai",
                 "google_api_key", 60 / 15, 16_384, True),
        # Free models allow 5 requests a minute. Not every free endpoint honours
        # response_format, so the schema in the prompt does the work.
        Endpoint("openrouter", "https://openrouter.ai/api/v1",
                 "openrouter_api_key", 60 / 5, 32_768, False),
        # 30 requests a minute, but 8k tokens a minute, and Groq counts max_tokens
        # against that, so the reply budget stays small.
        Endpoint("groq", "https://api.groq.com/openai/v1",
                 "groq_api_key", 60 / 30, 4_096, True),
    )
}


class ChatCompletions:
    """One provider's /chat/completions endpoint."""

    def __init__(self, endpoint: Endpoint, api_key: str, **client_options: Any) -> None:
        self.endpoint = endpoint
        self._sleep = client_options.get("sleep", time.sleep)
        self._client = ApiClient(
            service=endpoint.name,
            base_url=endpoint.base_url,
            min_interval=endpoint.min_interval,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=TIMEOUT,
            max_attempts=ATTEMPTS,
            first_wait=FIRST_WAIT,
            **client_options,
        )

    def complete(
        self, *, model: str, system: str, messages: list[Message], json_schema: dict[str, Any]
    ) -> Completion:
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": self.endpoint.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                *({"role": m.role, "content": m.content} for m in messages),
            ],
        }
        if self.endpoint.schema_output:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": json_schema.get("title", "output"), "schema": inline_refs(json_schema)},
            }
        for wait in (*UPSTREAM_WAITS, None):
            response = self._post(body, model)
            data = response.json()
            if "error" not in data:  # OpenRouter reports upstream failures inside a 200
                break
            if wait is None:
                raise LLMUnavailable(
                    f"provider error after retries: {_error_text(response)}", lasting=_daily(response)
                )
            self._sleep(wait)
        message = data["choices"][0]["message"]
        reported = data.get("usage") or {}
        usage = {
            "input_tokens": reported.get("prompt_tokens", 0),
            "output_tokens": reported.get("completion_tokens", 0),
        }
        thinking = (reported.get("completion_tokens_details") or {}).get("reasoning_tokens")
        if thinking:
            usage["reasoning_tokens"] = thinking
        return Completion(message.get("content") or "", usage=usage)

    def _post(self, body: dict[str, Any], model: str) -> httpx.Response:
        try:
            return self._client.post("/chat/completions", json=body)
        except RetryableStatus as exc:
            raise LLMUnavailable(
                f"HTTP {exc.response.status_code} after retries: {_error_text(exc.response)}",
                lasting=exc.response.status_code == 429 and _daily(exc.response),
            ) from exc
        except httpx.TransportError as exc:
            raise LLMUnavailable(f"{type(exc).__name__} after retries") from exc
        except httpx.HTTPStatusError as exc:
            hint = f" Check RA_{self.endpoint.key_setting.upper()}." if exc.response.status_code in (401, 403) else ""
            raise LLMError(
                f"{self.endpoint.name} rejected the request for {model}: "
                f"HTTP {exc.response.status_code}: {_error_text(exc.response)}.{hint}"
            ) from exc

    def close(self) -> None:
        self._client.close()


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """The schema with each $ref replaced by its definition: not every endpoint resolves $defs."""
    definitions = schema.get("$defs", {})

    def inline(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                target = inline(definitions[node["$ref"].rsplit("/", 1)[-1]])
                return {**target, **{k: inline(v) for k, v in node.items() if k != "$ref"}}
            return {k: inline(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [inline(item) for item in node]
        return node

    return inline(schema)


def _daily(response: httpx.Response) -> bool:
    """Whether the error is a used-up daily quota, which won't recover within the session."""
    return bool(_DAILY.search(response.text))


def _error_text(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text[:300]
    error = data.get("error") if isinstance(data, dict) else None
    if isinstance(error, dict):
        return str(error.get("message") or error)[:300]
    return str(error or response.text)[:300]


def _factory(endpoint: Endpoint) -> ProviderFactory:
    def build(settings: Settings) -> ChatCompletions:
        key = getattr(settings, endpoint.key_setting)
        if key is None:
            raise LLMConfigError(
                f"No API key for {endpoint.name}. Set RA_{endpoint.key_setting.upper()} in .env."
            )
        return ChatCompletions(endpoint, key.get_secret_value())

    return build


BUILTIN: dict[str, ProviderFactory] = {name: _factory(endpoint) for name, endpoint in ENDPOINTS.items()}
