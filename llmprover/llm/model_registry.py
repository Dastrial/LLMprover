"""Registry of available LLM models for strategy selection."""

from __future__ import annotations

from dataclasses import dataclass

from llmprover.llm.client import LLMClient, OpenAIClient


@dataclass(frozen=True)
class ModelEntry:
    """One selectable model: provider class, API model name, and prompt text."""

    client_cls: type[LLMClient]
    model: str
    description: str
    reasoning_effort: str | None = None


# Descriptions are meant for the orchestration strategy prompt.
# Third field is Chat Completions ``reasoning_effort`` (None = omit the parameter).
DEFAULT_OPENAI: tuple[tuple[str, str, str | None], ...] = (
    (
        "gpt-4o-mini",
        "Cheapest OpenAI model; no hidden reasoning. Prefer as a baseline for simple direct proofs.",
        None,
    ),
    (
        "gpt-5.6-luna",
        "Small recent model with reasoning=none; prefer for simple direct proofs.",
        "none",
    ),
    (
        "gpt-5.4-mini",
        "Stronger mini with reasoning=none; use for repairs or harder goals.",
        "none",
    ),
)


class ModelRegistry:
    """Maps model ids to ``LLMClient`` factories.

    Specs are read from registered descriptions (no API call).
    Clients are created lazily via ``client_cls.from_env`` and cached.

    The id is the API model name, or ``{model}-{reasoning_effort}`` when an
    effort is set, so the same model can be registered at several efforts.
    """

    def __init__(self) -> None:
        self._entries: dict[str, ModelEntry] = {}
        self._instances: dict[str, LLMClient] = {}

    @classmethod
    def with_default_openai_models(cls) -> ModelRegistry:
        """Return a registry preloaded with a small OpenAI catalogue."""
        registry = cls()
        for model, description, reasoning_effort in DEFAULT_OPENAI:
            registry.register(
                OpenAIClient,
                model,
                description,
                reasoning_effort=reasoning_effort,
            )
        return registry

    def register(
        self,
        client_cls: type[LLMClient],
        model: str,
        description: str,
        *,
        reasoning_effort: str | None = None,
    ) -> str:
        """Register a model and return its id.

        The id is *model*, or ``{model}-{reasoning_effort}`` when an effort is
        set. *reasoning_effort* is forwarded to the client when not None.
        """
        model_id = model if reasoning_effort is None else f"{model}-{reasoning_effort}"
        if model_id in self._entries:
            raise ValueError(f"Model already registered: {model_id!r}")
        self._entries[model_id] = ModelEntry(
            client_cls=client_cls,
            model=model,
            description=description,
            reasoning_effort=reasoning_effort,
        )
        return model_id

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
            kwargs: dict = {"model": entry.model}
            if entry.reasoning_effort is not None:
                kwargs["reasoning_effort"] = entry.reasoning_effort
            self._instances[model_id] = entry.client_cls.from_env(**kwargs)
        return self._instances[model_id]
