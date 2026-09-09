"""Tests for LLM history presenters."""

from __future__ import annotations

from unittest.mock import MagicMock

from llmprover.detailed_llm_history_presenter import DetailedLLMHistoryPresenter
from llmprover.domain import (
    AttemptRecord,
    CoqcResult,
    Goal,
    LemmaNode,
    Polarity,
    ProofAttempt,
)
from llmprover.history_presenter import HistoryPresenter
from llmprover.llm_client import CompletionResult, TokenUsage
from llmprover.strategy_llm_history_presenter import StrategyLLMHistoryPresenter
from llmprover.utils import EMPTY_ATTEMPTS

GOAL = Goal(name="plus_n0", statement="forall n : nat, n + 0 = n.")


def attempt_record(
    script: str,
    error: str,
    *,
    polarity: Polarity = Polarity.Positive,
    agent: str = "",
) -> AttemptRecord:
    return AttemptRecord(
        attempt=ProofAttempt(
            goal=GOAL,
            polarity=polarity,
            script=script,
            new_lemmas=[],
            agent=agent,
        ),
        rocq_error=CoqcResult(success=False, stderr=error),
    )


def test_detailed_is_history_presenter() -> None:
    assert isinstance(DetailedLLMHistoryPresenter(MagicMock()), HistoryPresenter)


def test_strategy_is_history_presenter() -> None:
    assert isinstance(StrategyLLMHistoryPresenter(MagicMock()), HistoryPresenter)


def test_detailed_render_calls_llm_with_raw_attempt() -> None:
    model = MagicMock()
    model.complete.return_value = CompletionResult(
        text="Rocq failed on induction.\n", usage=TokenUsage(4, 6)
    )
    record = attempt_record("induction n.", "Error on line 1.")

    text, usage = DetailedLLMHistoryPresenter(model).render(record, index=1)

    assert text == "Attempt 1:\nRocq failed on induction.\n\n"
    assert usage == TokenUsage(4, 6)
    user = model.complete.call_args.args[0][1].joined_text()
    assert "induction n." in user
    assert "Attempt 1" in user
    assert EMPTY_ATTEMPTS not in user


def test_present_prompt_block_skips_llm_when_empty() -> None:
    model = MagicMock()
    block, usage = DetailedLLMHistoryPresenter(model).present_prompt_block(
        LemmaNode(goal=GOAL)
    )
    assert block == EMPTY_ATTEMPTS
    assert usage == TokenUsage()
    model.complete.assert_not_called()


def test_present_prompt_block_reuses_cache_and_only_renders_new_record() -> None:
    model = MagicMock()
    model.complete.side_effect = [
        CompletionResult(text="first\n", usage=TokenUsage(1, 1)),
        CompletionResult(text="second\n", usage=TokenUsage(2, 2)),
    ]
    node = LemmaNode(
        goal=GOAL,
        positive=[attempt_record("auto.", "fail")],
    )
    presenter = DetailedLLMHistoryPresenter(model)
    legend = presenter.format_legend() + "\n\n"

    first, usage1 = presenter.present_prompt_block(node)
    second, usage2 = presenter.present_prompt_block(node)

    assert first == legend + "Attempt 1:\nfirst\n\n"
    assert second == first
    assert usage1 == TokenUsage(1, 1)
    assert usage2 == TokenUsage()
    model.complete.assert_called_once()

    node.positive.append(attempt_record("induction n.", "fail2"))
    third, usage3 = presenter.present_prompt_block(node)

    assert third == legend + "Attempt 1:\nfirst\n\n" + "Attempt 2:\nsecond\n\n"
    assert usage3 == TokenUsage(2, 2)
    assert model.complete.call_count == 2
    second_user = model.complete.call_args_list[1].args[0][1].joined_text()
    assert "first" not in second_user
    assert "Attempt 2" in second_user
    assert "induction n." in second_user


def test_present_prompt_block_includes_legend() -> None:
    model = MagicMock()
    model.complete.return_value = CompletionResult(
        text="summary\n", usage=TokenUsage(1, 1)
    )
    node = LemmaNode(
        goal=GOAL,
        positive=[attempt_record("auto.", "fail")],
    )
    presenter = DetailedLLMHistoryPresenter(model)

    block, _ = presenter.present_prompt_block(node)

    assert block.startswith(presenter.format_legend() + "\n\n")
    assert "Attempt 1:\nsummary" in block


def test_strategy_render_includes_agent_in_raw_attempt() -> None:
    model = MagicMock()
    model.complete.return_value = CompletionResult(
        text="syntax error on auto\n", usage=TokenUsage(1, 1)
    )
    record = attempt_record(
        "auto.",
        "Syntax error.",
        agent="DirectAgent",
    )

    text, usage = StrategyLLMHistoryPresenter(model).render(record, index=1)

    assert text == ("Attempt 1:\nAgent: DirectAgent\nsyntax error on auto\n\n")
    assert usage == TokenUsage(1, 1)
    user = model.complete.call_args.args[0][1].joined_text()
    assert "Agent: DirectAgent" in user
    assert "Explain the problem of Attempt 1" in user
    system = model.complete.call_args.args[0][0].joined_text()
    assert "identify the single concrete blocker" in system
    assert "name the prover agent" in system


def test_strategy_render_omits_agent_line_when_missing() -> None:
    model = MagicMock()
    model.complete.return_value = CompletionResult(
        text="empty script\n", usage=TokenUsage(1, 1)
    )
    record = attempt_record("auto.", "Syntax error.")

    text, _ = StrategyLLMHistoryPresenter(model).render(record, index=2)

    assert text == "Attempt 2:\nempty script\n\n"
    assert "Agent:" not in text


def test_render_includes_empty_script_placeholder() -> None:
    model = MagicMock()
    model.complete.return_value = CompletionResult(
        text="No tactic script was produced.\n", usage=TokenUsage(1, 1)
    )
    record = attempt_record("", "Error: incomplete proof.")

    DetailedLLMHistoryPresenter(model).render(record, index=1)

    user = model.complete.call_args.args[0][1].joined_text()
    assert "Script: (empty)" in user
    assert "Error: incomplete proof." in user


def test_strategy_present_prompt_block_appends_chunks() -> None:
    model = MagicMock()
    model.complete.side_effect = [
        CompletionResult(text="first\n", usage=TokenUsage(1, 1)),
        CompletionResult(text="second\n", usage=TokenUsage(1, 1)),
    ]
    node = LemmaNode(
        goal=GOAL,
        positive=[
            attempt_record("auto.", "fail", agent="DirectAgent1"),
            attempt_record("induction n.", "fail2", agent="DirectAgent2"),
        ],
    )
    presenter = StrategyLLMHistoryPresenter(model)

    text, _ = presenter.present_prompt_block(node)

    assert text == (
        presenter.format_legend()
        + "\n\n"
        + "Attempt 1:\nAgent: DirectAgent1\nfirst\n\n"
        + "Attempt 2:\nAgent: DirectAgent2\nsecond\n\n"
    )
