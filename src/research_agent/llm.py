"""The one way the agent calls an LLM.

Three roles, each mapped to one or more models in config
(``RA_LLM_<ROLE>=provider:model[,provider:model...]``):

- extraction: cheap, high-volume extraction
- reasoning: briefs, gaps, ideas, plans
- verifier: checks and judging; a different provider or model family from reasoning

Callers name a role, a versioned prompt and a Pydantic output type, and get a
validated object back. A role's later models are fallbacks: when a model is
unavailable (its quota used up, or its provider down), the role moves to the
next one for the rest of the session, and the trace records which model answered. So
does a model whose calls fail ``MAX_STRIKES`` times in a row (a provider down, a
dead endpoint), instead of being retried for a minute on every call.
Providers are registered by name (built-in ones in ``llm_providers``), so
swapping one is a config change.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ValidationError

from research_agent.config import Settings, model_specs
from research_agent.trace import child_event


class Role(StrEnum):
    EXTRACTION = "extraction"
    REASONING = "reasoning"
    VERIFIER = "verifier"


@dataclass(frozen=True)
class Prompt:
    name: str
    version: int
    system: str
    user: str

    @property
    def id(self) -> str:
        """Recorded in the trace, so every output can be tied to the prompt wording."""
        return f"{self.name}@v{self.version}"


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class Answer[T: BaseModel]:
    output: T
    model: str  # "provider:model" that answered


@dataclass
class Completion:
    text: str
    usage: dict[str, int] = field(default_factory=dict)  # token counts, as the provider reports them


class Provider(Protocol):
    def complete(
        self, *, model: str, system: str, messages: list[Message], json_schema: dict[str, Any]
    ) -> Completion:
        """Return text that should be one JSON object matching ``json_schema``.

        Raise LLMUnavailable when the model can't serve requests for now.
        """
        ...


ProviderFactory = Callable[[Settings], Provider]

_PROVIDERS: dict[str, ProviderFactory] = {}


def register_provider(name: str, factory: ProviderFactory) -> None:
    _PROVIDERS[name] = factory


def _factories() -> dict[str, ProviderFactory]:
    from research_agent.llm_providers import BUILTIN  # imported here: it imports this module

    return {**BUILTIN, **_PROVIDERS}


class LLMError(RuntimeError):
    pass


class LLMConfigError(LLMError):
    pass


class LLMOutputError(LLMError):
    pass


class LLMUnavailable(LLMError):
    """The model can't serve requests for now: rate limit or quota used up, or provider down.

    ``lasting`` means it won't recover within the session (a daily quota is used
    up), so the role stops trying it; otherwise only this call moves on.
    """

    def __init__(self, message: str, *, lasting: bool = False) -> None:
        super().__init__(message)
        self.lasting = lasting


MAX_STRIKES = 3  # calls in a row a model may fail before the role stops trying it for the session

_SCHEMA_INSTRUCTION = "\n\nReply with only a JSON object that matches this JSON Schema:\n{schema}"
_RETRY_MESSAGE = (
    "That reply did not match the required schema:\n{error}\n"
    "Reply again with only a JSON object that matches the schema."
)
_EMPTY_RETRY_MESSAGE = "Your reply was empty. Reply with only a JSON object that matches the schema."


class LLM:
    def __init__(self, settings: Settings, providers: dict[str, Provider] | None = None) -> None:
        self.settings = settings
        self._providers: dict[str, Provider] = dict(providers or {})
        self._unavailable: dict[tuple[str, str], str] = {}  # models that failed this session, and why
        self._strikes: dict[tuple[str, str], int] = {}  # each model's failed calls in a row

    def resolve(self, role: Role) -> list[tuple[str, str]]:
        """The (provider, model) pairs configured for a role, in fallback order."""
        specs = model_specs(getattr(self.settings, f"llm_{role}"))
        if not specs:
            raise LLMConfigError(
                f"No model is set for the {role} role. "
                f"Set RA_LLM_{role.upper()}=provider:model in .env."
            )
        return [(provider, model) for provider, _, model in (spec.partition(":") for spec in specs)]

    def _provider(self, name: str) -> Provider:
        if name not in self._providers:
            factories = _factories()
            factory = factories.get(name)
            if factory is None:
                registered = ", ".join(sorted(factories)) or "none"
                raise LLMConfigError(f"Unknown LLM provider {name!r} (registered: {registered}).")
            self._providers[name] = factory(self.settings)
        return self._providers[name]

    def models(self, role: Role) -> list[str]:
        """The role's models as "provider:model" strings, in fallback order."""
        return [f"{provider}:{model}" for provider, model in self.resolve(role)]

    def structured[T: BaseModel](
        self, role: Role, prompt: Prompt, output: type[T], *, max_attempts: int = 2
    ) -> T:
        """Call the role's model and validate its reply as ``output``.

        An invalid reply is sent back with the validation error, up to
        ``max_attempts`` calls per model. An unavailable model hands over to the
        role's next one.
        """
        return self.answer(role, prompt, output, max_attempts=max_attempts).output

    def answer[T: BaseModel](
        self, role: Role, prompt: Prompt, output: type[T], *, max_attempts: int = 2
    ) -> Answer[T]:
        """Like ``structured``, but also says which model answered, for callers that keep replies."""
        chain = self.resolve(role)
        schema = output.model_json_schema()
        system = prompt.system + _SCHEMA_INSTRUCTION.format(schema=json.dumps(schema))
        agent: dict[str, Any] = {"kind": "llm", "role": str(role)}

        with child_event("llm_call", agent=agent, prompt_version=prompt.id) as event:
            usage: dict[str, int] = {}
            event.detail.update(output=output.__name__, usage=usage)
            for provider_name, model in chain:
                if (provider_name, model) in self._unavailable:
                    event.detail.setdefault("unavailable", []).append(f"{provider_name}:{model}: earlier in this session")
                    continue
                agent.update(provider=provider_name, model=model)
                key = (provider_name, model)
                try:
                    reply = self._call(
                        self._provider(provider_name), model, system, prompt, output, schema,
                        max_attempts, event.detail, usage,
                    )
                    self._strikes.pop(key, None)
                    return Answer(reply, f"{provider_name}:{model}")
                except LLMUnavailable as exc:
                    self._strikes[key] = strikes = self._strikes.get(key, 0) + 1
                    if exc.lasting:
                        self._unavailable[key] = str(exc)
                    elif strikes >= MAX_STRIKES:
                        self._unavailable[key] = f"failed {strikes} calls in a row, the last: {exc}"
                    event.detail.setdefault("unavailable", []).append(f"{provider_name}:{model}: {exc}")
            reasons = "; ".join(event.detail["unavailable"])
            raise LLMUnavailable(f"No model for the {role} role is available ({reasons}).")

    def _call[T: BaseModel](
        self,
        provider: Provider,
        model: str,
        system: str,
        prompt: Prompt,
        output: type[T],
        schema: dict[str, Any],
        max_attempts: int,
        detail: dict[str, Any],
        usage: dict[str, int],
    ) -> T:
        messages = [Message("user", prompt.user)]
        for attempt in range(1, max_attempts + 1):
            detail["attempts"] = attempt
            completion = provider.complete(model=model, system=system, messages=messages, json_schema=schema)
            for key, count in completion.usage.items():
                usage[key] = usage.get(key, 0) + count
            try:
                return output.model_validate_json(_json_text(completion.text))
            except ValidationError as exc:
                error = exc
                if completion.text.strip():
                    messages = [
                        *messages,
                        Message("assistant", completion.text),
                        Message("user", _RETRY_MESSAGE.format(error=exc)),
                    ]
                else:
                    messages = [*messages, Message("user", _EMPTY_RETRY_MESSAGE)]
        raise LLMOutputError(
            f"{prompt.id}: reply did not match {output.__name__} after {max_attempts} attempts"
        ) from error

    def close(self) -> None:
        for provider in self._providers.values():
            close = getattr(provider, "close", None)
            if close is not None:
                close()


def _json_text(text: str) -> str:
    """The JSON object in a reply, without the code fences or prose some models wrap it in."""
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if 0 <= start < end else text
