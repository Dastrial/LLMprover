"""Tests for llmprover.prover_agents.direct_agent."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from llmprover.domain import (
    AttemptRecord,
    CoqcResult,
    Goal,
    LemmaNode,
    Polarity,
    ProofAttempt,
    statement_for_polarity,
)
from llmprover.llm.client import CompletionResult, TokenUsage
from llmprover.llm.prompting import (
    PromptMessage,
    PromptPart,
    fill_prompt,
)
from llmprover.prover_agents.direct_agent import DirectAgent
from llmprover.prover_agents.prompt_assembly import DIRECT_SPEC, cached_system_prompt

PROMPTS_DIR = (
    Path(__file__).resolve().parent.parent / "llmprover" / "prover_agents" / "prompts"
)
GOAL = Goal(name="plus_n0", statement="forall n : nat, n + 0 = n.")
NODE_WITH_HISTORY = LemmaNode(
    goal=GOAL,
    positive=[
        AttemptRecord(
            attempt=ProofAttempt(
                goal=GOAL,
                script="induction n.",
                new_lemmas=[],
            ),
            rocq_error=CoqcResult(success=False, stderr="Error."),
        )
    ],
)
EXPECTED_SYSTEM = cached_system_prompt(DIRECT_SPEC)


def expected_messages(polarity: Polarity) -> list[PromptMessage]:
    before = fill_prompt(
        (PROMPTS_DIR / "direct_proof_user_before.txt").read_text(encoding="utf-8").strip(),
        header=GOAL.environment.header,
    )
    if not before.endswith("\n"):
        before = f"{before}\n"
    after = fill_prompt(
        (PROMPTS_DIR / "direct_proof_user_after.txt").read_text(encoding="utf-8").strip(),
        statement=statement_for_polarity(GOAL.statement, polarity),
    )
    if after and not after.endswith("\n"):
        after = f"{after}\n"
    return [
        PromptMessage.text("system", EXPECTED_SYSTEM, cache_breakpoint=True),
        PromptMessage(
            role="user",
            parts=(PromptPart(before, cache_breakpoint=True), PromptPart(after)),
        ),
    ]


def test_prove_returns_parsed_script_from_clean_llm_output() -> None:
    mock_model = MagicMock()
    mock_model.complete.return_value = CompletionResult(
        text="induction n.\n- reflexivity.", usage=TokenUsage(3, 5)
    )

    attempt, usage = DirectAgent(mock_model).prove(NODE_WITH_HISTORY, Polarity.Positive)

    assert attempt == ProofAttempt(
        goal=GOAL,
        polarity=Polarity.Positive,
        script="induction n.\n- reflexivity.",
        new_lemmas=[],
    )
    assert usage == TokenUsage(3, 5)
    mock_model.complete.assert_called_once_with(expected_messages(Polarity.Positive))
    assert mock_model.complete.call_args.args[0][0].parts[-1].cache_breakpoint is True


def test_prove_negative_polarity_uses_negated_statement_in_prompt() -> None:
    mock_model = MagicMock()
    mock_model.complete.return_value = CompletionResult(
        text="intro H. contradiction.", usage=TokenUsage(3, 5)
    )

    attempt, usage = DirectAgent(mock_model).prove(
        LemmaNode(goal=GOAL), Polarity.Negative
    )

    assert attempt == ProofAttempt(
        goal=GOAL,
        polarity=Polarity.Negative,
        script="intro H. contradiction.",
        new_lemmas=[],
    )
    assert usage == TokenUsage(3, 5)
    mock_model.complete.assert_called_once_with(expected_messages(Polarity.Negative))


def test_prove_strips_markdown_fences_from_llm_output() -> None:
    mock_model = MagicMock()
    mock_model.complete.return_value = CompletionResult(
        text="```rocq\napply H.\n```", usage=TokenUsage(3, 5)
    )

    attempt, usage = DirectAgent(mock_model).prove(
        LemmaNode(goal=GOAL), Polarity.Positive
    )

    assert attempt == ProofAttempt(
        goal=GOAL,
        polarity=Polarity.Positive,
        script="apply H.",
        new_lemmas=[],
    )
    assert usage == TokenUsage(3, 5)


def test_prove_strips_proof_and_qed_from_llm_output() -> None:
    mock_model = MagicMock()
    mock_model.complete.return_value = CompletionResult(
        text="Proof.\napply H.\nQed.", usage=TokenUsage(3, 5)
    )

    attempt, usage = DirectAgent(mock_model).prove(
        LemmaNode(goal=GOAL), Polarity.Positive
    )

    assert attempt == ProofAttempt(
        goal=GOAL,
        polarity=Polarity.Positive,
        script="apply H.",
        new_lemmas=[],
    )
    assert usage == TokenUsage(3, 5)


def test_prove_handles_empty_llm_response() -> None:
    mock_model = MagicMock()
    mock_model.complete.return_value = CompletionResult(text="", usage=TokenUsage(3, 5))

    attempt, usage = DirectAgent(mock_model).prove(
        LemmaNode(goal=GOAL), Polarity.Positive
    )

    assert attempt == ProofAttempt(
        goal=GOAL,
        polarity=Polarity.Positive,
        script="",
        new_lemmas=[],
    )
    assert usage == TokenUsage(3, 5)


def test_describe_includes_model_and_reasoning_tokens() -> None:
    mock_model = MagicMock()
    mock_model.model = "gpt-5.6-luna"
    mock_model.reasoning_effort = "none"

    description = DirectAgent(mock_model).describe(TokenUsage(3, 5, 4))

    assert description == ("DirectAgent, model=gpt-5.6-luna reasoning=none")


def test_describe_omits_reasoning_effort_when_unset() -> None:
    mock_model = MagicMock()
    mock_model.model = "gpt-4o-mini"
    mock_model.reasoning_effort = None

    assert DirectAgent(mock_model).describe(TokenUsage()) == (
        "DirectAgent, model=gpt-4o-mini"
    )
