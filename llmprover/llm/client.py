"""LLM API backends — one implementation per provider.

Internal prompts use ``PromptPart`` / ``PromptMessage`` with optional cache
breakpoints. Each backend translates those markers only when the provider
supports them:

- OpenAI GPT-5.6+: ``prompt_cache_breakpoint`` + ``prompt_cache_options.mode=explicit``
- Anthropic: ``cache_control`` on content blocks (5-minute ephemeral TTL)
- Mistral: no explicit breakpoints; optional ``prompt_cache_key`` only
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Self, Sequence

from anthropic import Anthropic
from mistralai.client import Mistral
from openai import OpenAI

from llmprover.llm.model_prices import price_for
from llmprover.llm.prompting import PromptMessage


@dataclass(frozen=True)
class ModelTokenUsage:
    """Token counts attributed to a single model.

    ``input_tokens`` is the total input volume for display/aggregation
    (OpenAI/Mistral ``prompt_tokens``; Anthropic sum of exclusive input +
    cache write + cache read). Cache categories are also stored separately
    so ``cost_usd()`` can bill each exclusive bucket.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0

    def __add__(self, other: ModelTokenUsage) -> ModelTokenUsage:
        return ModelTokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
        )

    def regular_input_tokens(self) -> int:
        """Input tokens billed at the standard input rate (never negative)."""
        return max(
            0,
            self.input_tokens - self.cache_read_tokens - self.cache_write_tokens,
        )


@dataclass(frozen=True)
class TokenUsage:
    """Token counts for one or more LLM calls, keyed by model name.

    This structure is required by `Orchestrator` to compute per-model cost.
    """

    by_model: Mapping[str, ModelTokenUsage] = field(default_factory=dict)

    def __init__(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        reasoning_tokens: int = 0,
        *,
        model: str = "",
        cache_write_tokens: int = 0,
        cache_read_tokens: int = 0,
        by_model: Mapping[str, ModelTokenUsage] | None = None,
    ) -> None:
        if by_model is None:
            if (
                input_tokens
                or output_tokens
                or reasoning_tokens
                or cache_write_tokens
                or cache_read_tokens
                or model
            ):
                by_model = {
                    model: ModelTokenUsage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        reasoning_tokens=reasoning_tokens,
                        cache_write_tokens=cache_write_tokens,
                        cache_read_tokens=cache_read_tokens,
                    )
                }
            else:
                by_model = {}
        object.__setattr__(self, "by_model", MappingProxyType(dict(by_model)))

    @property
    def input_tokens(self) -> int:
        return sum(usage.input_tokens for usage in self.by_model.values())

    @property
    def output_tokens(self) -> int:
        return sum(usage.output_tokens for usage in self.by_model.values())

    @property
    def reasoning_tokens(self) -> int:
        return sum(usage.reasoning_tokens for usage in self.by_model.values())

    @property
    def cache_write_tokens(self) -> int:
        return sum(usage.cache_write_tokens for usage in self.by_model.values())

    @property
    def cache_read_tokens(self) -> int:
        return sum(usage.cache_read_tokens for usage in self.by_model.values())

    def __add__(self, other: TokenUsage) -> TokenUsage:
        merged: dict[str, ModelTokenUsage] = dict(self.by_model)
        for name, usage in other.by_model.items():
            merged[name] = merged.get(name, ModelTokenUsage()) + usage
        return TokenUsage(by_model=merged)

    def cost_usd(self) -> float:
        """Dollar cost from provider-reported categories and list prices.

        For each model::

            regular = input - cache_read - cache_write  (floored at 0)
            cost = regular * input_price
                 + cache_write * cache_write_price   (if priced; else input)
                 + cache_read * cache_read_price     (if priced; else input)
                 + output * output_price

        When a cache price column is ``None``, that category is billed at the
        regular input rate (and not double-counted via ``regular``).
        """
        total = 0.0
        for model, counts in self.by_model.items():
            price = price_for(model)
            write_priced = price.cache_write_usd_per_million is not None
            read_priced = price.cache_read_usd_per_million is not None
            # Unpriced cache categories stay in the regular-input bucket.
            regular = max(
                0,
                counts.input_tokens
                - (counts.cache_write_tokens if write_priced else 0)
                - (counts.cache_read_tokens if read_priced else 0),
            )
            total += regular * price.input_usd_per_million / 1_000_000
            if write_priced:
                total += (
                    counts.cache_write_tokens
                    * price.cache_write_usd_per_million
                    / 1_000_000
                )
            if read_priced:
                total += (
                    counts.cache_read_tokens
                    * price.cache_read_usd_per_million
                    / 1_000_000
                )
            total += counts.output_tokens * price.output_usd_per_million / 1_000_000
        return total


@dataclass(frozen=True)
class CompletionResult:
    """Assistant text plus token usage for a single completion."""

    text: str
    usage: TokenUsage = TokenUsage()


class LLMClient(ABC):
    """Abstract chat-completion backend."""

    DEFAULT_API_KEY_ENV: str | None = None
    DEFAULT_MODEL: str | None = None

    def __init__(self, model: str) -> None:
        self.model = model

    @classmethod
    @abstractmethod
    def from_api_key(cls, api_key: str, *, model: str | None = None) -> Self:
        """Create a backend from an explicit API key."""

    @classmethod
    def from_env(
        cls,
        env_var: str | None = None,
        *,
        model: str | None = None,
        **kwargs,
    ) -> Self:
        """Create a backend from an environment variable.

        Uses ``DEFAULT_API_KEY_ENV`` when *env_var* is omitted (e.g. ``OPENAI_API_KEY``).
        Extra *kwargs* are forwarded to ``from_api_key`` (e.g. ``reasoning_effort``).
        """
        key_name = env_var or cls.DEFAULT_API_KEY_ENV
        api_key = os.environ.get(key_name)
        if not api_key:
            raise ValueError(f"API key not found in environment variable {key_name!r}")
        return cls.from_api_key(api_key, model=model, **kwargs)

    @abstractmethod
    def complete(self, messages: Sequence[PromptMessage]) -> CompletionResult:
        """Run a chat completion and return text plus token usage."""


def _usage_int(value: object) -> int:
    """Best-effort numeric conversion for SDK response fields."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, int(value))


def openai_supports_explicit_cache(model: str) -> bool:
    """True for GPT-5.6 and later OpenAI model families."""
    match = re.match(r"^gpt-(\d+)(?:\.(\d+))?", model)
    if match is None:
        return False
    major = int(match.group(1))
    minor = int(match.group(2) or 0)
    return (major, minor) >= (5, 6)


def _to_openai_messages(
    messages: Sequence[PromptMessage], *, explicit_cache: bool
) -> list[dict]:
    """Translate internal messages for Chat Completions."""
    api_messages: list[dict] = []
    for message in messages:
        if not explicit_cache:
            api_messages.append(
                {"role": message.role, "content": message.joined_text()}
            )
            continue
        content: list[dict] = []
        for part in message.parts:
            block: dict = {"type": "text", "text": part.text}
            if part.cache_breakpoint:
                block["prompt_cache_breakpoint"] = {"mode": "explicit"}
            content.append(block)
        api_messages.append({"role": message.role, "content": content})
    return api_messages


def _to_anthropic_payload(
    messages: Sequence[PromptMessage],
) -> tuple[list[dict] | None, list[dict]]:
    """Split system blocks (with cache_control) from non-system messages."""
    system_blocks: list[dict] = []
    api_messages: list[dict] = []
    for message in messages:
        blocks: list[dict] = []
        for part in message.parts:
            block: dict = {"type": "text", "text": part.text}
            if part.cache_breakpoint:
                block["cache_control"] = {"type": "ephemeral"}
            blocks.append(block)
        if message.role == "system":
            system_blocks.extend(blocks)
        else:
            api_messages.append({"role": message.role, "content": blocks})
    return (system_blocks or None), api_messages


def _to_mistral_messages(messages: Sequence[PromptMessage]) -> list[dict]:
    """Mistral has no explicit breakpoints; flatten parts to plain strings."""
    return [
        {"role": message.role, "content": message.joined_text()} for message in messages
    ]


def _openai_style_usage(response: object, *, model: str) -> TokenUsage:
    """Parse OpenAI-compatible usage (also used as a base for Mistral)."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return TokenUsage()

    prompt_tokens = _usage_int(getattr(usage, "prompt_tokens", 0))
    completion_tokens = _usage_int(getattr(usage, "completion_tokens", 0))

    completion_details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = 0
    if completion_details is not None:
        reasoning_tokens = _usage_int(
            getattr(completion_details, "reasoning_tokens", 0)
        )

    prompt_details = getattr(usage, "prompt_tokens_details", None)
    cache_read = 0
    cache_write = 0
    if prompt_details is not None:
        cache_read = _usage_int(getattr(prompt_details, "cached_tokens", 0))
        cache_write = _usage_int(getattr(prompt_details, "cache_write_tokens", 0))

    return TokenUsage(
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens,
        model=model,
        cache_write_tokens=cache_write,
        cache_read_tokens=cache_read,
    )


def _anthropic_usage(response: object, *, model: str) -> TokenUsage:
    """Parse Anthropic usage (input_tokens is exclusive of cache categories)."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return TokenUsage()

    exclusive_input = _usage_int(getattr(usage, "input_tokens", 0))
    cache_write = _usage_int(getattr(usage, "cache_creation_input_tokens", 0))
    cache_read = _usage_int(getattr(usage, "cache_read_input_tokens", 0))
    output_tokens = _usage_int(getattr(usage, "output_tokens", 0))
    # Normalize to total input for aggregation/logging; cost subtracts again.
    total_input = exclusive_input + cache_write + cache_read
    return TokenUsage(
        input_tokens=total_input,
        output_tokens=output_tokens,
        model=model,
        cache_write_tokens=cache_write,
        cache_read_tokens=cache_read,
    )


class OpenAIClient(LLMClient):
    """OpenAI chat models (GPT)."""

    DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
    DEFAULT_MODEL = "gpt-5-nano"

    def __init__(
        self,
        model: str,
        client: OpenAI,
        *,
        reasoning_effort: str | None = None,
        prompt_cache_key: str | None = None,
    ) -> None:
        super().__init__(model)
        self.client = client
        self.reasoning_effort = reasoning_effort
        self.prompt_cache_key = prompt_cache_key or f"llmprover:{model}"

    @classmethod
    def from_api_key(
        cls,
        api_key: str,
        *,
        model: str | None = None,
        reasoning_effort: str | None = None,
        prompt_cache_key: str | None = None,
    ) -> OpenAIClient:
        return cls(
            model=model or cls.DEFAULT_MODEL,
            client=OpenAI(api_key=api_key),
            reasoning_effort=reasoning_effort,
            prompt_cache_key=prompt_cache_key,
        )

    def complete(self, messages: Sequence[PromptMessage]) -> CompletionResult:
        has_breakpoints = any(
            part.cache_breakpoint for message in messages for part in message.parts
        )
        explicit = openai_supports_explicit_cache(self.model) and has_breakpoints
        api_messages = _to_openai_messages(messages, explicit_cache=explicit)
        kwargs: dict = {"model": self.model, "messages": api_messages}
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if explicit:
            kwargs["prompt_cache_key"] = self.prompt_cache_key
            kwargs["prompt_cache_options"] = {"mode": "explicit"}
        response = self.client.chat.completions.create(**kwargs)
        return CompletionResult(
            text=response.choices[0].message.content or "",
            usage=_openai_style_usage(response, model=self.model),
        )


class MistralAIClient(LLMClient):
    """Mistral models via the official ``mistralai`` SDK.

    Mistral supports automatic prefix caching with ``prompt_cache_key`` and
    reports ``prompt_tokens_details.cached_tokens``. It has no explicit
    content-block breakpoints; internal breakpoints are ignored on the wire.
    """

    DEFAULT_MODEL = "mistral-large-latest"
    DEFAULT_API_KEY_ENV = "MISTRAL_API_KEY"

    def __init__(
        self,
        model: str,
        client: Mistral,
        *,
        prompt_cache_key: str | None = None,
    ) -> None:
        super().__init__(model)
        self.client = client
        self.prompt_cache_key = prompt_cache_key or f"llmprover:{model}"

    @classmethod
    def from_api_key(
        cls,
        api_key: str,
        *,
        model: str | None = None,
        prompt_cache_key: str | None = None,
    ) -> MistralAIClient:
        return cls(
            model=model or cls.DEFAULT_MODEL,
            client=Mistral(api_key=api_key),
            prompt_cache_key=prompt_cache_key,
        )

    def complete(self, messages: Sequence[PromptMessage]) -> CompletionResult:
        response = self.client.chat.complete(
            model=self.model,
            messages=_to_mistral_messages(messages),
            prompt_cache_key=self.prompt_cache_key,
        )
        return CompletionResult(
            text=response.choices[0].message.content or "",
            usage=_openai_style_usage(response, model=self.model),
        )


class AnthropicClient(LLMClient):
    """Anthropic Claude models."""

    DEFAULT_MODEL = "claude-haiku-4-5"
    DEFAULT_API_KEY_ENV = "ANTHROPIC_API_KEY"

    def __init__(self, model: str, client: object) -> None:
        super().__init__(model)
        self.client = client

    @classmethod
    def from_api_key(cls, api_key: str, *, model: str | None = None) -> AnthropicClient:
        return cls(model=model or cls.DEFAULT_MODEL, client=Anthropic(api_key=api_key))

    def complete(self, messages: Sequence[PromptMessage]) -> CompletionResult:
        system_blocks, api_messages = _to_anthropic_payload(messages)

        kwargs: dict = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": 4096,
        }
        if system_blocks is not None:
            kwargs["system"] = system_blocks

        response = self.client.messages.create(**kwargs)

        parts: list[str] = []
        for block in response.content:
            if block.type == "text":
                parts.append(block.text)

        return CompletionResult(
            text="".join(parts),
            usage=_anthropic_usage(response, model=self.model),
        )
