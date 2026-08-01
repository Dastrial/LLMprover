"""LLM API backends — one implementation per provider."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self

from anthropic import Anthropic
from mistralai.client import Mistral
from openai import OpenAI


@dataclass(frozen=True)
class TokenUsage:
    """Token counts for one or more LLM calls."""

    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


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
    def from_env(cls, env_var: str | None = None, *, model: str | None = None) -> Self:
        """Create a backend from an environment variable.

        Uses ``DEFAULT_API_KEY_ENV`` when *env_var* is omitted (e.g. ``OPENAI_API_KEY``).
        """
        key_name = env_var or cls.DEFAULT_API_KEY_ENV
        api_key = os.environ.get(key_name)
        if not api_key:
            raise ValueError(f"API key not found in environment variable {key_name!r}")
        return cls.from_api_key(api_key, model=model)

    @abstractmethod
    def complete(self, messages: list[dict[str, str]]) -> CompletionResult:
        """Run a chat completion and return text plus token usage."""


def _openai_style_usage(response: object) -> TokenUsage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
    )


class OpenAIClient(LLMClient):
    """OpenAI chat models (GPT)."""

    DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
    DEFAULT_MODEL = "gpt-5-nano"

    def __init__(self, model: str, client: OpenAI) -> None:
        super().__init__(model)
        self.client = client

    @classmethod
    def from_api_key(cls, api_key: str, *, model: str | None = None) -> OpenAIClient:
        return cls(model=model or cls.DEFAULT_MODEL, client=OpenAI(api_key=api_key))

    def complete(self, messages: list[dict[str, str]]) -> CompletionResult:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return CompletionResult(
            text=response.choices[0].message.content or "",
            usage=_openai_style_usage(response),
        )


class MistralAIClient(LLMClient):
    """Mistral models via the official ``mistralai`` SDK."""

    DEFAULT_MODEL = "mistral-large-latest"
    DEFAULT_API_KEY_ENV = "MISTRAL_API_KEY"

    def __init__(self, model: str, client: Mistral) -> None:
        super().__init__(model)
        self.client = client

    @classmethod
    def from_api_key(cls, api_key: str, *, model: str | None = None) -> MistralAIClient:
        return cls(
            model=model or cls.DEFAULT_MODEL,
            client=Mistral(api_key=api_key),
        )

    def complete(self, messages: list[dict[str, str]]) -> CompletionResult:
        response = self.client.chat.complete(
            model=self.model,
            messages=messages,
        )
        return CompletionResult(
            text=response.choices[0].message.content or "",
            usage=_openai_style_usage(response),
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

    def complete(self, messages: list[dict[str, str]]) -> CompletionResult:
        system_parts: list[str] = []
        api_messages: list[dict[str, str]] = []
        for message in messages:
            if message["role"] == "system":
                system_parts.append(message["content"])
            else:
                api_messages.append(message)

        kwargs: dict = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": 4096,
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)

        response = self.client.messages.create(**kwargs)

        parts: list[str] = []
        for block in response.content:
            if block.type == "text":
                parts.append(block.text)

        usage = getattr(response, "usage", None)
        token_usage = TokenUsage()
        if usage is not None:
            token_usage = TokenUsage(
                input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            )
        return CompletionResult(text="".join(parts), usage=token_usage)
