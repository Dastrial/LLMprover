"""Registry of available LLM models for strategy selection."""

from __future__ import annotations

from dataclasses import dataclass

from llmprover.llm_client import LLMClient, OpenAIClient


@dataclass(frozen=True)
class ModelEntry:
    """One selectable model: provider class, API model name, and prompt text."""

    client_cls: type[LLMClient]
    model: str
    description: str


# Descriptions are meant for the orchestration strategy prompt.
DEFAULT_OPENAI: tuple[tuple[str, str], ...] = (
    (
        "gpt-5-nano",
        "Cheapest/fastest OpenAI model; prefer for simple direct proofs.",
    ),
    (
        "gpt-4o-mini",
        "Inexpensive OpenAI model; good default for repair and light decomposition.",
    ),
    (
        "gpt-4o",
        "Stronger OpenAI model; use for hard goals or repeated failures.",
    ),
)


class ModelRegistry:
    """Maps model ids (API model names) to ``LLMClient`` factories.

    Specs are read from registered descriptions (no API call).
    Clients are created lazily via ``client_cls.from_env`` and cached.
    """

    def __init__(self) -> None:
        self._entries: dict[str, ModelEntry] = {}
        self._instances: dict[str, LLMClient] = {}

    @classmethod
    def with_default_openai_models(cls) -> ModelRegistry:
        """Return a registry preloaded with a small OpenAI catalogue."""
        registry = cls()
        for model, description in DEFAULT_OPENAI:
            registry.register(OpenAIClient, model, description)
        return registry

    def register(
        self,
        client_cls: type[LLMClient],
        model: str,
        description: str,
    ) -> str:
        """Register a model and return its id (the API model name)."""
        if model in self._entries:
            raise ValueError(f"Model already registered: {model!r}")
        self._entries[model] = ModelEntry(
            client_cls=client_cls,
            model=model,
            description=description,
        )
        return model

    def specs(self) -> dict[str, str]:
        """Return ``{model_id: description}`` for every registered model."""
        return {
            model_id: entry.description for model_id, entry in self._entries.items()
        }

    def get(self, model_id: str) -> LLMClient:
        """Return the (lazily created, cached) client for *model_id*."""
        if model_id not in self._entries:
            raise KeyError(f"Unknown model id: {model_id!r}")
        if model_id not in self._instances:
            entry = self._entries[model_id]
            self._instances[model_id] = entry.client_cls.from_env(model=entry.model)
        return self._instances[model_id]
