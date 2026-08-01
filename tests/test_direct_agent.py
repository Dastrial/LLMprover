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
from llmprover.llm_client import CompletionResult, TokenUsage
from llmprover.prompts import fill_prompt, load_prompt
from llmprover.prover_agents.direct_agent import DirectAgent

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
EXPECTED_SYSTEM = load_prompt(PROMPTS_DIR / "direct_proof_system.txt")


def expected_user(polarity: Polarity) -> str:
    return fill_prompt(
        load_prompt(PROMPTS_DIR / "direct_proof_user.txt"),
        statement=statement_for_polarity(GOAL.statement, polarity),
    )


def test_prove_returns_parsed_script_from_clean_llm_output() -> None:
    mock_model = MagicMock()
    mock_model.complete.return_value = CompletionResult(text="induction n.\n- reflexivity.", usage=TokenUsage(3, 5))

    attempt, usage = DirectAgent(mock_model).prove(NODE_WITH_HISTORY, Polarity.Positive)

    assert attempt == ProofAttempt(
        goal=GOAL,
        polarity=Polarity.Positive,
        script="induction n.\n- reflexivity.",
        new_lemmas=[],
    )
    assert usage == TokenUsage(3, 5)
    mock_model.complete.assert_called_once_with(
        [
            {"role": "system", "content": EXPECTED_SYSTEM},
            {"role": "user", "content": expected_user(Polarity.Positive)},
        ]
    )


def test_prove_negative_polarity_uses_negated_statement_in_prompt() -> None:
    mock_model = MagicMock()
    mock_model.complete.return_value = CompletionResult(text="intro H. contradiction.", usage=TokenUsage(3, 5))

    attempt, usage = DirectAgent(mock_model).prove(LemmaNode(goal=GOAL), Polarity.Negative)

    assert attempt == ProofAttempt(
        goal=GOAL,
        polarity=Polarity.Negative,
        script="intro H. contradiction.",
        new_lemmas=[],
    )
    assert usage == TokenUsage(3, 5)
    mock_model.complete.assert_called_once_with(
        [
            {"role": "system", "content": EXPECTED_SYSTEM},
            {"role": "user", "content": expected_user(Polarity.Negative)},
        ]
    )


def test_prove_strips_markdown_fences_from_llm_output() -> None:
    mock_model = MagicMock()
    mock_model.complete.return_value = CompletionResult(text="```rocq\napply H.\n```", usage=TokenUsage(3, 5))

    attempt, usage = DirectAgent(mock_model).prove(LemmaNode(goal=GOAL), Polarity.Positive)

    assert attempt == ProofAttempt(
        goal=GOAL,
        polarity=Polarity.Positive,
        script="apply H.",
        new_lemmas=[],
    )
    assert usage == TokenUsage(3, 5)


def test_prove_strips_proof_and_qed_from_llm_output() -> None:
    mock_model = MagicMock()
    mock_model.complete.return_value = CompletionResult(text="Proof.\napply H.\nQed.", usage=TokenUsage(3, 5))

    attempt, usage = DirectAgent(mock_model).prove(LemmaNode(goal=GOAL), Polarity.Positive)

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

    attempt, usage = DirectAgent(mock_model).prove(LemmaNode(goal=GOAL), Polarity.Positive)

    assert attempt == ProofAttempt(
        goal=GOAL,
        polarity=Polarity.Positive,
        script="",
        new_lemmas=[],
    )
    assert usage == TokenUsage(3, 5)
