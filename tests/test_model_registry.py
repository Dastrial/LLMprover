"""Tests for llmprover.llm.model_registry."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llmprover.llm.client import OpenAIClient
from llmprover.llm.model_registry import ModelRegistry


def test_register_returns_model_name_as_id() -> None:
    registry = ModelRegistry()
    model_id = registry.register(
        OpenAIClient,
        "gpt-5-nano",
        "cheap OpenAI model",
    )
    assert model_id == "gpt-5-nano"


def test_register_rejects_duplicate_model_name() -> None:
    registry = ModelRegistry()
    registry.register(OpenAIClient, "gpt-5-nano", "first")
    with pytest.raises(ValueError, match="already registered"):
        registry.register(OpenAIClient, "gpt-5-nano", "second")


def test_register_rejects_duplicate_reasoning_config() -> None:
    registry = ModelRegistry()
    registry.register(OpenAIClient, "gpt-5.6-luna", "first", reasoning_effort="none")
    with pytest.raises(ValueError, match="already registered"):
        registry.register(
            OpenAIClient, "gpt-5.6-luna", "second", reasoning_effort="none"
        )


def test_specs_returns_descriptions_without_creating_clients() -> None:
    registry = ModelRegistry()
    registry.register(OpenAIClient, "gpt-5-nano", "cheap")
    registry.register(OpenAIClient, "gpt-4o", "strong")

    with patch.object(OpenAIClient, "from_env") as from_env:
        assert registry.specs() == {
            "gpt-5-nano": "cheap",
            "gpt-4o": "strong",
        }
        from_env.assert_not_called()


def test_get_creates_client_via_from_env_with_model_name() -> None:
    registry = ModelRegistry()
    registry.register(OpenAIClient, "gpt-4o-mini", "mid")
    fake_client = MagicMock()

    with patch.object(OpenAIClient, "from_env", return_value=fake_client) as from_env:
        client = registry.get("gpt-4o-mini")

    assert client is fake_client
    from_env.assert_called_once_with(model="gpt-4o-mini")


def test_register_with_reasoning_effort_uses_composite_id() -> None:
    registry = ModelRegistry()
    model_id = registry.register(
        OpenAIClient,
        "gpt-5.6-luna",
        "luna none",
        reasoning_effort="none",
    )
    assert model_id == "gpt-5.6-luna-none"
    assert registry.specs() == {"gpt-5.6-luna-none": "luna none"}


def test_register_allows_same_model_at_different_reasoning_efforts() -> None:
    registry = ModelRegistry()
    none_id = registry.register(
        OpenAIClient, "gpt-5.6-luna", "none", reasoning_effort="none"
    )
    low_id = registry.register(
        OpenAIClient, "gpt-5.6-luna", "low", reasoning_effort="low"
    )
    assert none_id == "gpt-5.6-luna-none"
    assert low_id == "gpt-5.6-luna-low"
    assert set(registry.specs()) == {"gpt-5.6-luna-none", "gpt-5.6-luna-low"}


def test_get_forwards_reasoning_effort_to_from_env() -> None:
    registry = ModelRegistry()
    registry.register(OpenAIClient, "gpt-5.6-luna", "none", reasoning_effort="none")
    fake_client = MagicMock()

    with patch.object(OpenAIClient, "from_env", return_value=fake_client) as from_env:
        client = registry.get("gpt-5.6-luna-none")

    assert client is fake_client
    from_env.assert_called_once_with(model="gpt-5.6-luna", reasoning_effort="none")


def test_get_caches_client() -> None:
    registry = ModelRegistry()
    registry.register(OpenAIClient, "gpt-4o", "strong")
    fake_client = MagicMock()

    with patch.object(OpenAIClient, "from_env", return_value=fake_client) as from_env:
        first = registry.get("gpt-4o")
        second = registry.get("gpt-4o")

    assert first is second
    from_env.assert_called_once_with(model="gpt-4o")


def test_get_unknown_id_raises_key_error() -> None:
    registry = ModelRegistry()
    with pytest.raises(KeyError, match="Unknown model id"):
        registry.get("missing-model")


def test_get_is_lazy_until_called() -> None:
    registry = ModelRegistry()
    registry.register(OpenAIClient, "gpt-5-nano", "cheap")

    with patch.object(OpenAIClient, "from_env") as from_env:
        from_env.assert_not_called()
        registry.get("gpt-5-nano")
        from_env.assert_called_once()


def test_with_default_openai_models_registers_openai_catalogue() -> None:
    registry = ModelRegistry.with_default_openai_models()
    specs = registry.specs()

    assert set(specs) == {
        "gpt-4o-mini",
        "gpt-5.6-luna-none",
        "gpt-5.4-mini-none",
    }
    assert all(isinstance(text, str) and text for text in specs.values())

    fake_client = MagicMock()
    with patch.object(OpenAIClient, "from_env", return_value=fake_client) as from_env:
        assert registry.get("gpt-5.6-luna-none") is fake_client
        from_env.assert_called_once_with(model="gpt-5.6-luna", reasoning_effort="none")
