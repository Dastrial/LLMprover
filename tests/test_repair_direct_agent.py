"""Tests for llmprover.prover_agents.repair_direct_agent."""

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
from llmprover.prompts import fill_prompt, load_prompt
from llmprover.prover_agents.repair_direct_agent import RepairDirectAgent

PROMPTS_DIR = (
    Path(__file__).resolve().parent.parent / "llmprover" / "prover_agents" / "prompts"
)
GOAL = Goal(name="plus_n0", statement="forall n : nat, n + 0 = n.")
EXPECTED_SYSTEM = load_prompt(PROMPTS_DIR / "direct_proof_system.txt")
EMPTY = "No previous attempts.\n"


def attempt_record(
    script: str,
    error: str,
    *,
    polarity: Polarity = Polarity.Positive,
    new_lemmas: list[Goal] | None = None,
) -> AttemptRecord:
    return AttemptRecord(
        attempt=ProofAttempt(
            goal=GOAL,
            polarity=polarity,
            script=script,
            new_lemmas=new_lemmas or [],
        ),
        rocq_error=CoqcResult(success=False, stderr=error),
    )


def expected_user(
    polarity: Polarity,
    this_formula_attempts: str,
    opposite_attempts: str,
) -> str:
    return fill_prompt(
        load_prompt(PROMPTS_DIR / "repair_direct_proof_user.txt"),
        statement=statement_for_polarity(GOAL.statement, polarity),
        this_formula_attempts=this_formula_attempts,
        opposite_attempts=opposite_attempts,
    )


def test_format_histories_formats_both_polarities() -> None:
    node = LemmaNode(
        goal=GOAL,
        positive=[
            attempt_record("induction n.", "Error on line 1."),
            attempt_record("auto.", "Unable to unify."),
        ],
        negative=[attempt_record("intro H.", "Error B.", polarity=Polarity.Negative)],
    )

    positive_text, negative_text = RepairDirectAgent.format_histories(node)

    assert positive_text == (
        "Attempt 1:\n"
        "Statement: forall n : nat, n + 0 = n.\n"
        "Script: induction n.\n"
        "Rocq errors: Error on line 1.\n"
        "\n"
        "Attempt 2:\n"
        "Statement: forall n : nat, n + 0 = n.\n"
        "Script: auto.\n"
        "Rocq errors: Unable to unify.\n"
        "\n"
    )
    assert negative_text == (
        "Attempt 1:\n"
        "Statement: ~ (forall n : nat, n + 0 = n.)\n"
        "Script: intro H.\n"
        "Rocq errors: Error B.\n"
        "\n"
    )


def test_format_histories_skips_decomposition_attempts() -> None:
    node = LemmaNode(
        goal=GOAL,
        positive=[
            attempt_record(
                "apply helper.",
                "Error.",
                new_lemmas=[Goal(name="helper", statement="True.")],
            ),
            attempt_record("reflexivity.", "Still failing."),
        ],
    )

    positive_text, negative_text = RepairDirectAgent.format_histories(node)

    assert positive_text == (
        "Attempt 1:\n"
        "Statement: forall n : nat, n + 0 = n.\n"
        "Script: reflexivity.\n"
        "Rocq errors: Still failing.\n"
        "\n"
    )
    assert negative_text == EMPTY


def test_format_histories_returns_placeholders_when_empty() -> None:
    positive_text, negative_text = RepairDirectAgent.format_histories(
        LemmaNode(goal=GOAL)
    )
    assert positive_text == EMPTY
    assert negative_text == EMPTY


def test_prove_positive_keeps_histories_in_order() -> None:
    mock_model = MagicMock()
    mock_model.complete.return_value = "reflexivity."
    node = LemmaNode(
        goal=GOAL,
        positive=[attempt_record("induction n.", "Error on line 1.")],
        negative=[attempt_record("intro H.", "Error B.", polarity=Polarity.Negative)],
    )
    positive_attempts = (
        "Attempt 1:\n"
        "Statement: forall n : nat, n + 0 = n.\n"
        "Script: induction n.\n"
        "Rocq errors: Error on line 1.\n"
        "\n"
    )
    negative_attempts = (
        "Attempt 1:\n"
        "Statement: ~ (forall n : nat, n + 0 = n.)\n"
        "Script: intro H.\n"
        "Rocq errors: Error B.\n"
        "\n"
    )

    attempt = RepairDirectAgent(mock_model).prove(node, Polarity.Positive)

    assert attempt == ProofAttempt(
        goal=GOAL,
        polarity=Polarity.Positive,
        script="reflexivity.",
        new_lemmas=[],
    )
    mock_model.complete.assert_called_once_with(
        [
            {"role": "system", "content": EXPECTED_SYSTEM},
            {
                "role": "user",
                "content": expected_user(
                    Polarity.Positive, positive_attempts, negative_attempts
                ),
            },
        ]
    )


def test_prove_negative_swaps_histories_in_prompt() -> None:
    mock_model = MagicMock()
    mock_model.complete.return_value = "intro H. contradiction."
    node = LemmaNode(
        goal=GOAL,
        positive=[attempt_record("induction n.", "Error on line 1.")],
        negative=[attempt_record("intro H.", "Error B.", polarity=Polarity.Negative)],
    )
    positive_attempts = (
        "Attempt 1:\n"
        "Statement: forall n : nat, n + 0 = n.\n"
        "Script: induction n.\n"
        "Rocq errors: Error on line 1.\n"
        "\n"
    )
    negative_attempts = (
        "Attempt 1:\n"
        "Statement: ~ (forall n : nat, n + 0 = n.)\n"
        "Script: intro H.\n"
        "Rocq errors: Error B.\n"
        "\n"
    )

    attempt = RepairDirectAgent(mock_model).prove(node, Polarity.Negative)

    assert attempt == ProofAttempt(
        goal=GOAL,
        polarity=Polarity.Negative,
        script="intro H. contradiction.",
        new_lemmas=[],
    )
    mock_model.complete.assert_called_once_with(
        [
            {"role": "system", "content": EXPECTED_SYSTEM},
            {
                "role": "user",
                "content": expected_user(
                    Polarity.Negative,
                    this_formula_attempts=negative_attempts,
                    opposite_attempts=positive_attempts,
                ),
            },
        ]
    )


def test_prove_strips_markdown_fences_from_llm_output() -> None:
    mock_model = MagicMock()
    mock_model.complete.return_value = "```\napply H.\n```"

    attempt = RepairDirectAgent(mock_model).prove(
        LemmaNode(goal=GOAL), Polarity.Positive
    )

    assert attempt == ProofAttempt(
        goal=GOAL,
        polarity=Polarity.Positive,
        script="apply H.",
        new_lemmas=[],
    )


def test_prove_handles_empty_llm_response() -> None:
    mock_model = MagicMock()
    mock_model.complete.return_value = ""

    attempt = RepairDirectAgent(mock_model).prove(
        LemmaNode(goal=GOAL), Polarity.Positive
    )

    assert attempt == ProofAttempt(
        goal=GOAL,
        polarity=Polarity.Positive,
        script="",
        new_lemmas=[],
    )
