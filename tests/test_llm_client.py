"""Tests for llmprover.llm_client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llmprover.llm_client import (
    AnthropicClient,
    CompletionResult,
    MistralAIClient,
    OpenAIClient,
    TokenUsage,
    _openai_style_usage,
    _to_anthropic_payload,
    _to_mistral_messages,
    _to_openai_messages,
    openai_supports_explicit_cache,
)
from llmprover.prompts import PromptMessage, PromptPart

USER_MESSAGE = [PromptMessage.text("user", "Hello")]
SYSTEM_AND_USER = [
    PromptMessage.text("system", "You are helpful."),
    PromptMessage.text("user", "Hello"),
]


def _openai_usage(
    *,
    prompt: int = 10,
    completion: int = 4,
    cached: int = 0,
    cache_write: int = 0,
    reasoning: int = 0,
) -> MagicMock:
    usage = MagicMock()
    usage.prompt_tokens = prompt
    usage.completion_tokens = completion
    details = MagicMock()
    details.cached_tokens = cached
    details.cache_write_tokens = cache_write
    usage.prompt_tokens_details = details
    completion_details = MagicMock()
    completion_details.reasoning_tokens = reasoning
    usage.completion_tokens_details = completion_details
    return usage


# --- TokenUsage ---


def test_token_usage_aggregation_preserves_cache_categories() -> None:
    a = TokenUsage(
        100,
        10,
        3,
        model="gpt-5.6-luna",
        cache_write_tokens=40,
        cache_read_tokens=50,
    )
    b = TokenUsage(
        200,
        20,
        1,
        model="gpt-5.6-luna",
        cache_write_tokens=10,
        cache_read_tokens=80,
    )
    c = TokenUsage(
        50,
        5,
        model="claude-haiku-4-5",
        cache_write_tokens=20,
        cache_read_tokens=10,
    )
    total = a + b + c
    assert total.input_tokens == 350
    assert total.cache_write_tokens == 70
    assert total.cache_read_tokens == 140
    assert total.output_tokens == 35
    assert total.reasoning_tokens == 4
    assert total.cost_usd() == pytest.approx(a.cost_usd() + b.cost_usd() + c.cost_usd())


def test_token_usage_cost_with_cache_categories() -> None:
    # gpt-5.6-luna: in=0.20, write=0.25, read=0.02, out=1.20
    usage = TokenUsage(
        input_tokens=1000,
        output_tokens=10,
        model="gpt-5.6-luna",
        cache_write_tokens=400,
        cache_read_tokens=500,
    )
    # regular = 1000 - 400 - 500 = 100
    expected = (
        100 * 0.20 / 1_000_000
        + 400 * 0.25 / 1_000_000
        + 500 * 0.02 / 1_000_000
        + 10 * 1.20 / 1_000_000
    )
    assert usage.cost_usd() == pytest.approx(expected)


def test_token_usage_cost_usd_never_negative_with_inconsistent_counters() -> None:
    usage = TokenUsage(
        input_tokens=10,
        output_tokens=0,
        model="gpt-5.6-luna",
        cache_write_tokens=100,
        cache_read_tokens=100,
    )
    assert usage.cost_usd() >= 0


# --- openai_supports_explicit_cache ---


def test_openai_supports_explicit_cache_version_gate() -> None:
    assert openai_supports_explicit_cache("gpt-5.6-luna")
    assert openai_supports_explicit_cache("gpt-5.6")
    assert openai_supports_explicit_cache("gpt-6-preview")
    assert not openai_supports_explicit_cache("gpt-5.4-mini")
    assert not openai_supports_explicit_cache("gpt-5-nano")
    assert not openai_supports_explicit_cache("gpt-4o-mini")


# --- _to_openai_messages ---


def test_to_openai_messages_without_explicit_cache_flattens_parts() -> None:
    messages = [
        PromptMessage.text("system", "pre", cache_breakpoint=True),
        PromptMessage(
            role="user",
            parts=(
                PromptPart("hist\n", cache_breakpoint=True),
                PromptPart("dyn"),
            ),
        ),
    ]
    assert _to_openai_messages(messages, explicit_cache=False) == [
        {"role": "system", "content": "pre"},
        {"role": "user", "content": "hist\ndyn"},
    ]


def test_to_openai_messages_with_explicit_cache_emits_breakpoints() -> None:
    messages = [
        PromptMessage.text("system", "stable pre-prompt", cache_breakpoint=True),
        PromptMessage(
            role="user",
            parts=(
                PromptPart("header\nstatement\n"),
                PromptPart("history\n", cache_breakpoint=True),
                PromptPart("ABOUT\ndynamic\n"),
            ),
        ),
    ]
    assert _to_openai_messages(messages, explicit_cache=True) == [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "stable pre-prompt",
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "header\nstatement\n"},
                {
                    "type": "text",
                    "text": "history\n",
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                },
                {"type": "text", "text": "ABOUT\ndynamic\n"},
            ],
        },
    ]


# --- _to_anthropic_payload ---


def test_to_anthropic_payload_without_system_returns_none_system() -> None:
    messages = [
        PromptMessage(
            role="user",
            parts=(
                PromptPart("hist\n", cache_breakpoint=True),
                PromptPart("dyn"),
            ),
        ),
    ]
    assert _to_anthropic_payload(messages) == (
        None,
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "hist\n",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": "dyn"},
                ],
            }
        ],
    )


def test_to_anthropic_payload_splits_system_and_emits_cache_control() -> None:
    messages = [
        PromptMessage.text("system", "stable pre-prompt", cache_breakpoint=True),
        PromptMessage.text("system", "extra system"),
        PromptMessage(
            role="user",
            parts=(
                PromptPart("header\nstatement\n"),
                PromptPart("history\n", cache_breakpoint=True),
                PromptPart("ABOUT\ndynamic\n"),
            ),
        ),
    ]
    assert _to_anthropic_payload(messages) == (
        [
            {
                "type": "text",
                "text": "stable pre-prompt",
                "cache_control": {"type": "ephemeral"},
            },
            {"type": "text", "text": "extra system"},
        ],
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "header\nstatement\n"},
                    {
                        "type": "text",
                        "text": "history\n",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": "ABOUT\ndynamic\n"},
                ],
            }
        ],
    )


# --- _to_mistral_messages ---


def test_to_mistral_messages_flattens_parts_and_ignores_breakpoints() -> None:
    messages = [
        PromptMessage.text("system", "pre", cache_breakpoint=True),
        PromptMessage(
            role="user",
            parts=(
                PromptPart("hist\n", cache_breakpoint=True),
                PromptPart("dyn"),
            ),
        ),
    ]
    assert _to_mistral_messages(messages) == [
        {"role": "system", "content": "pre"},
        {"role": "user", "content": "hist\ndyn"},
    ]


# --- _openai_style_usage ---


def test_openai_style_usage_returns_empty_when_usage_missing() -> None:
    assert _openai_style_usage(MagicMock(usage=None), model="gpt-4o") == TokenUsage()


def test_openai_style_usage_parses_cache_and_reasoning_details() -> None:
    response = MagicMock()
    response.usage = _openai_usage(
        prompt=2600,
        completion=5,
        cached=2000,
        cache_write=400,
        reasoning=2,
    )
    assert _openai_style_usage(response, model="gpt-5.6-luna") == TokenUsage(
        input_tokens=2600,
        output_tokens=5,
        reasoning_tokens=2,
        model="gpt-5.6-luna",
        cache_write_tokens=400,
        cache_read_tokens=2000,
    )


# --- OpenAIClient ---


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
    openai_client = OpenAIClient.from_api_key("sk-test", model="gpt-4o-mini")
    assert openai_client.model == "gpt-4o-mini"


def test_openai_from_env_reads_default_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(OpenAIClient.DEFAULT_API_KEY_ENV, "sk-from-env")
    with patch("llmprover.llm_client.OpenAI") as mock_openai_cls:
        openai_client = OpenAIClient.from_env()
    assert openai_client.model == OpenAIClient.DEFAULT_MODEL
    mock_openai_cls.assert_called_once_with(api_key="sk-from-env")


def test_openai_from_env_raises_when_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(OpenAIClient.DEFAULT_API_KEY_ENV, raising=False)
    with pytest.raises(
        ValueError, match="API key not found in environment variable 'OPENAI_API_KEY'"
    ):
        OpenAIClient.from_env()


def test_openai_from_env_reads_custom_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(OpenAIClient.DEFAULT_API_KEY_ENV, raising=False)
    monkeypatch.setenv("CUSTOM_OPENAI_KEY", "sk-custom-env")
    with patch("llmprover.llm_client.OpenAI") as mock_openai_cls:
        openai_client = OpenAIClient.from_env("CUSTOM_OPENAI_KEY")
    assert openai_client.model == OpenAIClient.DEFAULT_MODEL
    mock_openai_cls.assert_called_once_with(api_key="sk-custom-env")


def test_openai_from_env_accepts_custom_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(OpenAIClient.DEFAULT_API_KEY_ENV, "sk-from-env")
    with patch("llmprover.llm_client.OpenAI"):
        openai_client = OpenAIClient.from_env(model="gpt-4o")
    assert openai_client.model == "gpt-4o"


def test_openai_complete_returns_assistant_text() -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Hi there"
    mock_response.usage = _openai_usage(prompt=10, completion=4)
    mock_client.chat.completions.create.return_value = mock_response

    client = OpenAIClient(model="gpt-4o", client=mock_client)
    result = client.complete(USER_MESSAGE)
    assert result == CompletionResult(
        text="Hi there",
        usage=TokenUsage(input_tokens=10, output_tokens=4, model="gpt-4o"),
    )
    mock_client.chat.completions.create.assert_called_once_with(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
    )


def test_openai_gpt56_sends_explicit_breakpoints_and_mode() -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "ok"
    mock_response.usage = _openai_usage(
        prompt=2600, completion=5, cached=2000, cache_write=400, reasoning=2
    )
    mock_client.chat.completions.create.return_value = mock_response

    client = OpenAIClient(model="gpt-5.6-luna", client=mock_client)
    messages = [
        PromptMessage.text("system", "stable pre-prompt", cache_breakpoint=True),
        PromptMessage(
            role="user",
            parts=(
                PromptPart("header\nstatement\n"),
                PromptPart("history\n", cache_breakpoint=True),
                PromptPart("ABOUT\ndynamic\n"),
            ),
        ),
    ]
    result = client.complete(messages)

    assert result.usage == TokenUsage(
        input_tokens=2600,
        output_tokens=5,
        reasoning_tokens=2,
        model="gpt-5.6-luna",
        cache_write_tokens=400,
        cache_read_tokens=2000,
    )
    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs["prompt_cache_options"] == {"mode": "explicit"}
    assert kwargs["prompt_cache_key"] == "llmprover:gpt-5.6-luna"
    assert kwargs["messages"][0]["content"] == [
        {
            "type": "text",
            "text": "stable pre-prompt",
            "prompt_cache_breakpoint": {"mode": "explicit"},
        }
    ]
    user_blocks = kwargs["messages"][1]["content"]
    assert user_blocks[0] == {"type": "text", "text": "header\nstatement\n"}
    assert user_blocks[1] == {
        "type": "text",
        "text": "history\n",
        "prompt_cache_breakpoint": {"mode": "explicit"},
    }
    assert user_blocks[2] == {"type": "text", "text": "ABOUT\ndynamic\n"}
    assert "prompt_cache_breakpoint" not in user_blocks[2]


def test_openai_older_model_does_not_send_gpt56_cache_fields() -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "ok"
    mock_response.usage = _openai_usage(prompt=100, completion=1, cached=80)
    mock_client.chat.completions.create.return_value = mock_response

    client = OpenAIClient(model="gpt-5.4-mini", client=mock_client)
    client.complete(
        [
            PromptMessage.text("system", "pre", cache_breakpoint=True),
            PromptMessage(
                role="user",
                parts=(PromptPart("hist\n", cache_breakpoint=True), PromptPart("dyn")),
            ),
        ]
    )
    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert "prompt_cache_options" not in kwargs
    assert "prompt_cache_key" not in kwargs
    assert kwargs["messages"] == [
        {"role": "system", "content": "pre"},
        {"role": "user", "content": "hist\ndyn"},
    ]
    assert kwargs["messages"][0]["content"] == "pre"


def test_openai_complete_returns_empty_string_when_content_is_none() -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = None
    mock_response.usage = None
    mock_client.chat.completions.create.return_value = mock_response

    openai_client = OpenAIClient(model="gpt-4o", client=mock_client)
    assert openai_client.complete(USER_MESSAGE) == CompletionResult(text="")


# --- MistralAIClient ---


@patch("llmprover.llm_client.Mistral")
def test_mistral_from_api_key_uses_default_model(mock_mistral_cls: MagicMock) -> None:
    sdk_client = MagicMock()
    mock_mistral_cls.return_value = sdk_client

    mistral_client = MistralAIClient.from_api_key("mistral-key")

    assert mistral_client.model == MistralAIClient.DEFAULT_MODEL
    assert mistral_client.client is sdk_client
    mock_mistral_cls.assert_called_once_with(api_key="mistral-key")


@patch("llmprover.llm_client.Mistral")
def test_mistral_from_api_key_accepts_custom_model(mock_mistral_cls: MagicMock) -> None:
    mistral_client = MistralAIClient.from_api_key(
        "mistral-key", model="mistral-small-latest"
    )
    assert mistral_client.model == "mistral-small-latest"


def test_mistral_from_env_reads_default_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MistralAIClient.DEFAULT_API_KEY_ENV, "mistral-from-env")
    with patch("llmprover.llm_client.Mistral") as mock_mistral_cls:
        mistral_client = MistralAIClient.from_env()
    assert mistral_client.model == MistralAIClient.DEFAULT_MODEL
    mock_mistral_cls.assert_called_once_with(api_key="mistral-from-env")


def test_mistral_from_env_raises_when_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(MistralAIClient.DEFAULT_API_KEY_ENV, raising=False)
    with pytest.raises(
        ValueError, match="API key not found in environment variable 'MISTRAL_API_KEY'"
    ):
        MistralAIClient.from_env()


def test_mistral_from_env_reads_custom_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(MistralAIClient.DEFAULT_API_KEY_ENV, raising=False)
    monkeypatch.setenv("CUSTOM_MISTRAL_KEY", "mistral-custom-env")
    with patch("llmprover.llm_client.Mistral") as mock_mistral_cls:
        mistral_client = MistralAIClient.from_env("CUSTOM_MISTRAL_KEY")
    assert mistral_client.model == MistralAIClient.DEFAULT_MODEL
    mock_mistral_cls.assert_called_once_with(api_key="mistral-custom-env")


def test_mistral_from_env_accepts_custom_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MistralAIClient.DEFAULT_API_KEY_ENV, "mistral-from-env")
    with patch("llmprover.llm_client.Mistral"):
        mistral_client = MistralAIClient.from_env(model="mistral-small-latest")
    assert mistral_client.model == "mistral-small-latest"


def test_mistral_complete_returns_assistant_text() -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Bonjour"
    mock_response.usage = _openai_usage(prompt=7, completion=2)
    mock_client.chat.complete.return_value = mock_response

    mistral_client = MistralAIClient(model="mistral-large-latest", client=mock_client)
    assert mistral_client.complete(USER_MESSAGE) == CompletionResult(
        text="Bonjour",
        usage=TokenUsage(input_tokens=7, output_tokens=2, model="mistral-large-latest"),
    )
    mock_client.chat.complete.assert_called_once_with(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": "Hello"}],
        prompt_cache_key="llmprover:mistral-large-latest",
    )


def test_mistral_flattens_breakpoints_and_bills_cached_reads() -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "ok"
    mock_response.usage = _openai_usage(prompt=1008 + 5, completion=30, cached=1008)
    mock_client.chat.complete.return_value = mock_response

    client = MistralAIClient(model="mistral-large-latest", client=mock_client)
    result = client.complete(
        [
            PromptMessage.text("system", "pre", cache_breakpoint=True),
            PromptMessage(
                role="user",
                parts=(
                    PromptPart("hist\n", cache_breakpoint=True),
                    PromptPart("dyn"),
                ),
            ),
        ]
    )
    kwargs = mock_client.chat.complete.call_args.kwargs
    assert kwargs["messages"] == [
        {"role": "system", "content": "pre"},
        {"role": "user", "content": "hist\ndyn"},
    ]
    assert "prompt_cache_breakpoint" not in str(kwargs)
    # large: in=0.50, cached=0.05, out=1.50; regular=5
    expected = 5 * 0.50 / 1_000_000 + 1008 * 0.05 / 1_000_000 + 30 * 1.50 / 1_000_000
    assert result.usage.cost_usd() == pytest.approx(expected)
    assert result.usage.cache_write_tokens == 0
    assert result.usage.cache_read_tokens == 1008


# --- AnthropicClient ---


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


@patch("llmprover.llm_client.Anthropic")
def test_anthropic_from_api_key_accepts_custom_model(
    mock_anthropic_cls: MagicMock,
) -> None:
    anthropic_client = AnthropicClient.from_api_key(
        "anthropic-key", model="claude-sonnet-4-20250514"
    )
    assert anthropic_client.model == "claude-sonnet-4-20250514"


def test_anthropic_from_env_raises_when_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(AnthropicClient.DEFAULT_API_KEY_ENV, raising=False)
    with pytest.raises(
        ValueError,
        match="API key not found in environment variable 'ANTHROPIC_API_KEY'",
    ):
        AnthropicClient.from_env()


def test_anthropic_from_env_reads_custom_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(AnthropicClient.DEFAULT_API_KEY_ENV, raising=False)
    monkeypatch.setenv("CUSTOM_ANTHROPIC_KEY", "anthropic-custom-env")
    with patch("llmprover.llm_client.Anthropic") as mock_anthropic_cls:
        anthropic_client = AnthropicClient.from_env("CUSTOM_ANTHROPIC_KEY")
    assert anthropic_client.model == AnthropicClient.DEFAULT_MODEL
    mock_anthropic_cls.assert_called_once_with(api_key="anthropic-custom-env")


def test_anthropic_from_env_accepts_custom_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(AnthropicClient.DEFAULT_API_KEY_ENV, "anthropic-from-env")
    with patch("llmprover.llm_client.Anthropic"):
        anthropic_client = AnthropicClient.from_env(model="claude-sonnet-4-20250514")
    assert anthropic_client.model == "claude-sonnet-4-20250514"


def test_anthropic_complete_splits_system_messages_and_keeps_user_messages() -> None:
    mock_client = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Response text"
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    mock_response = MagicMock()
    mock_response.content = [text_block, tool_block]
    mock_response.usage.input_tokens = 12
    mock_response.usage.output_tokens = 3
    mock_response.usage.cache_creation_input_tokens = 0
    mock_response.usage.cache_read_input_tokens = 0
    mock_client.messages.create.return_value = mock_response

    model = "claude-sonnet-4-20250514"
    anthropic_client = AnthropicClient(model=model, client=mock_client)
    assert anthropic_client.complete(SYSTEM_AND_USER) == CompletionResult(
        text="Response text",
        usage=TokenUsage(input_tokens=12, output_tokens=3, model=model),
    )
    mock_client.messages.create.assert_called_once_with(
        model=model,
        messages=[{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
        max_tokens=4096,
        system=[{"type": "text", "text": "You are helpful."}],
    )


def test_anthropic_translates_breakpoints_to_cache_control() -> None:
    mock_client = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "OK"
    mock_response = MagicMock()
    mock_response.content = [text_block]
    mock_response.usage.input_tokens = 50
    mock_response.usage.output_tokens = 2
    mock_response.usage.cache_creation_input_tokens = 1000
    mock_response.usage.cache_read_input_tokens = 2000
    mock_client.messages.create.return_value = mock_response

    client = AnthropicClient(model="claude-haiku-4-5", client=mock_client)
    result = client.complete(
        [
            PromptMessage.text("system", "pre", cache_breakpoint=True),
            PromptMessage(
                role="user",
                parts=(
                    PromptPart("header\n"),
                    PromptPart("history\n", cache_breakpoint=True),
                    PromptPart("about\n"),
                ),
            ),
        ]
    )
    kwargs = mock_client.messages.create.call_args.kwargs
    assert kwargs["system"] == [
        {
            "type": "text",
            "text": "pre",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    user_content = kwargs["messages"][0]["content"]
    assert user_content[1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in user_content[2]
    # total input = 50 + 1000 + 2000
    assert result.usage.input_tokens == 3050
    assert result.usage.cache_write_tokens == 1000
    assert result.usage.cache_read_tokens == 2000
    # haiku: in=1, write=1.25, read=0.10, out=5
    expected = (
        50 * 1.00 / 1_000_000
        + 1000 * 1.25 / 1_000_000
        + 2000 * 0.10 / 1_000_000
        + 2 * 5.00 / 1_000_000
    )
    assert result.usage.cost_usd() == pytest.approx(expected)


def test_anthropic_complete_joins_multiple_system_messages() -> None:
    mock_client = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "OK"
    mock_response = MagicMock()
    mock_response.content = [text_block]
    mock_response.usage.input_tokens = 1
    mock_response.usage.output_tokens = 1
    mock_response.usage.cache_creation_input_tokens = 0
    mock_response.usage.cache_read_input_tokens = 0
    mock_client.messages.create.return_value = mock_response

    messages = [
        PromptMessage.text("system", "Part 1"),
        PromptMessage.text("system", "Part 2"),
        PromptMessage.text("user", "Hi"),
    ]
    anthropic_client = AnthropicClient(
        model=AnthropicClient.DEFAULT_MODEL, client=mock_client
    )
    anthropic_client.complete(messages)

    kwargs = mock_client.messages.create.call_args.kwargs
    assert kwargs["system"] == [
        {"type": "text", "text": "Part 1"},
        {"type": "text", "text": "Part 2"},
    ]
    assert kwargs["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Hi"}]}
    ]


def test_anthropic_complete_omits_system_when_none() -> None:
    mock_client = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "OK"
    mock_response = MagicMock()
    mock_response.content = [text_block]
    mock_response.usage.input_tokens = 1
    mock_response.usage.output_tokens = 1
    mock_response.usage.cache_creation_input_tokens = 0
    mock_response.usage.cache_read_input_tokens = 0
    mock_client.messages.create.return_value = mock_response

    anthropic_client = AnthropicClient(
        model=AnthropicClient.DEFAULT_MODEL, client=mock_client
    )
    anthropic_client.complete(USER_MESSAGE)

    kwargs = mock_client.messages.create.call_args.kwargs
    assert "system" not in kwargs
