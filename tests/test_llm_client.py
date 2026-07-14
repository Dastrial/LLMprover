"""Tests for llmprover.llm_client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llmprover.llm_client import AnthropicClient, MistralAIClient, OpenAIClient

USER_MESSAGE = [{"role": "user", "content": "Hello"}]
SYSTEM_AND_USER = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello"},
]


# --- OpenAIClient ---


def test_openai_complete_returns_assistant_text() -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Hi there"
    mock_client.chat.completions.create.return_value = mock_response

    client = OpenAIClient(model="gpt-4o", client=mock_client)
    assert client.complete(USER_MESSAGE) == "Hi there"
    mock_client.chat.completions.create.assert_called_once_with(
        model="gpt-4o",
        messages=USER_MESSAGE,
    )


def test_openai_complete_returns_empty_string_when_content_is_none() -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = None
    mock_client.chat.completions.create.return_value = mock_response

    client = OpenAIClient(model="gpt-4o", client=mock_client)
    assert client.complete(USER_MESSAGE) == ""


@patch("llmprover.llm_client.OpenAI")
def test_openai_from_api_key_uses_default_model(mock_openai_cls: MagicMock) -> None:
    sdk_client = MagicMock()
    mock_openai_cls.return_value = sdk_client

    openai_client = OpenAIClient.from_api_key("sk-test")

    assert openai_client.model == OpenAIClient.DEFAULT_MODEL
    assert openai_client.client is sdk_client
    mock_openai_cls.assert_called_once_with(api_key="sk-test")


@patch("llmprover.llm_client.OpenAI")
def test_openai_from_api_key_accepts_custom_model(mock_openai_cls: MagicMock) -> None:
    client = OpenAIClient.from_api_key("sk-test", model="gpt-4o-mini")
    assert client.model == "gpt-4o-mini"


def test_openai_from_env_reads_default_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    with patch("llmprover.llm_client.OpenAI") as mock_openai_cls:
        client = OpenAIClient.from_env()
    assert client.model == OpenAIClient.DEFAULT_MODEL
    mock_openai_cls.assert_called_once_with(api_key="sk-from-env")


def test_openai_from_env_raises_when_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(
        ValueError, match="API key not found in environment variable 'OPENAI_API_KEY'"
    ):
        OpenAIClient.from_env()


def test_openai_from_env_reads_custom_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CUSTOM_OPENAI_KEY", "sk-custom-env")
    with patch("llmprover.llm_client.OpenAI") as mock_openai_cls:
        client = OpenAIClient.from_env("CUSTOM_OPENAI_KEY")
    assert client.model == OpenAIClient.DEFAULT_MODEL
    mock_openai_cls.assert_called_once_with(api_key="sk-custom-env")


def test_openai_from_env_accepts_custom_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    with patch("llmprover.llm_client.OpenAI"):
        client = OpenAIClient.from_env(model="gpt-4o")
    assert client.model == "gpt-4o"


# --- MistralAIClient ---


def test_mistral_complete_returns_assistant_text() -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Bonjour"
    mock_client.chat.complete.return_value = mock_response

    client = MistralAIClient(model="mistral-large-latest", client=mock_client)
    assert client.complete(USER_MESSAGE) == "Bonjour"
    mock_client.chat.complete.assert_called_once_with(
        model="mistral-large-latest",
        messages=USER_MESSAGE,
    )


@patch("llmprover.llm_client.Mistral")
def test_mistral_from_api_key_uses_default_model(mock_mistral_cls: MagicMock) -> None:
    sdk_client = MagicMock()
    mock_mistral_cls.return_value = sdk_client

    mistral_client = MistralAIClient.from_api_key("mistral-key")

    assert mistral_client.model == "mistral-large-latest"
    assert mistral_client.client is sdk_client
    mock_mistral_cls.assert_called_once_with(api_key="mistral-key")


def test_mistral_from_env_reads_default_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "mistral-from-env")
    with patch("llmprover.llm_client.Mistral") as mock_mistral_cls:
        client = MistralAIClient.from_env()
    assert client.model == "mistral-large-latest"
    mock_mistral_cls.assert_called_once_with(api_key="mistral-from-env")


def test_mistral_from_env_raises_when_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(
        ValueError, match="API key not found in environment variable 'MISTRAL_API_KEY'"
    ):
        MistralAIClient.from_env()


@patch("llmprover.llm_client.Mistral")
def test_mistral_from_api_key_accepts_custom_model(mock_mistral_cls: MagicMock) -> None:
    client = MistralAIClient.from_api_key("mistral-key", model="mistral-small-latest")
    assert client.model == "mistral-small-latest"


def test_mistral_from_env_reads_custom_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.setenv("CUSTOM_MISTRAL_KEY", "mistral-custom-env")
    with patch("llmprover.llm_client.Mistral") as mock_mistral_cls:
        client = MistralAIClient.from_env("CUSTOM_MISTRAL_KEY")
    assert client.model == MistralAIClient.DEFAULT_MODEL
    mock_mistral_cls.assert_called_once_with(api_key="mistral-custom-env")


def test_mistral_from_env_accepts_custom_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "mistral-from-env")
    with patch("llmprover.llm_client.Mistral"):
        client = MistralAIClient.from_env(model="mistral-small-latest")
    assert client.model == "mistral-small-latest"


# --- AnthropicClient ---


def test_anthropic_complete_splits_system_messages_and_keeps_user_messages() -> None:
    mock_client = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Response text"
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    mock_response = MagicMock()
    mock_response.content = [text_block, tool_block]
    mock_client.messages.create.return_value = mock_response

    model = "claude-sonnet-4-20250514"
    client = AnthropicClient(model=model, client=mock_client)
    assert client.complete(SYSTEM_AND_USER) == "Response text"
    mock_client.messages.create.assert_called_once_with(
        model=model,
        messages=USER_MESSAGE,
        max_tokens=4096,
        system="You are helpful.",
    )


def test_anthropic_complete_joins_multiple_system_messages() -> None:
    mock_client = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "OK"
    mock_response = MagicMock()
    mock_response.content = [text_block]
    mock_client.messages.create.return_value = mock_response

    messages = [
        {"role": "system", "content": "Part 1"},
        {"role": "system", "content": "Part 2"},
        {"role": "user", "content": "Hi"},
    ]
    client = AnthropicClient(model=AnthropicClient.DEFAULT_MODEL, client=mock_client)
    client.complete(messages)

    kwargs = mock_client.messages.create.call_args.kwargs
    assert kwargs["system"] == "Part 1\n\nPart 2"
    assert kwargs["messages"] == [{"role": "user", "content": "Hi"}]


def test_anthropic_complete_omits_system_when_none() -> None:
    mock_client = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "OK"
    mock_response = MagicMock()
    mock_response.content = [text_block]
    mock_client.messages.create.return_value = mock_response

    client = AnthropicClient(model=AnthropicClient.DEFAULT_MODEL, client=mock_client)
    client.complete(USER_MESSAGE)

    kwargs = mock_client.messages.create.call_args.kwargs
    assert "system" not in kwargs


@patch("llmprover.llm_client.Anthropic")
def test_anthropic_from_api_key_uses_default_model(
    mock_anthropic_cls: MagicMock,
) -> None:
    sdk_client = MagicMock()
    mock_anthropic_cls.return_value = sdk_client

    anthropic_client = AnthropicClient.from_api_key("anthropic-key")

    assert anthropic_client.model == AnthropicClient.DEFAULT_MODEL
    assert anthropic_client.client is sdk_client
    mock_anthropic_cls.assert_called_once_with(api_key="anthropic-key")


def test_anthropic_from_env_raises_when_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(
        ValueError,
        match="API key not found in environment variable 'ANTHROPIC_API_KEY'",
    ):
        AnthropicClient.from_env()


@patch("llmprover.llm_client.Anthropic")
def test_anthropic_from_api_key_accepts_custom_model(
    mock_anthropic_cls: MagicMock,
) -> None:
    client = AnthropicClient.from_api_key(
        "anthropic-key", model="claude-sonnet-4-20250514"
    )
    assert client.model == "claude-sonnet-4-20250514"


def test_anthropic_from_env_reads_custom_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CUSTOM_ANTHROPIC_KEY", "anthropic-custom-env")
    with patch("llmprover.llm_client.Anthropic") as mock_anthropic_cls:
        client = AnthropicClient.from_env("CUSTOM_ANTHROPIC_KEY")
    assert client.model == AnthropicClient.DEFAULT_MODEL
    mock_anthropic_cls.assert_called_once_with(api_key="anthropic-custom-env")


def test_anthropic_from_env_accepts_custom_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-from-env")
    with patch("llmprover.llm_client.Anthropic"):
        client = AnthropicClient.from_env(model="claude-sonnet-4-20250514")
    assert client.model == "claude-sonnet-4-20250514"
