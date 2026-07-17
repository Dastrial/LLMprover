"""Tests for llmprover.prover_agents.decomposition_agent."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llmprover.domain import (
    Goal,
    LemmaNode,
    Polarity,
    ProofAttempt,
    statement_for_polarity,
)
from llmprover.prompts import fill_prompt, load_prompt
from llmprover.prover_agents.decomposition_agent import (
    DecompositionAgent,
    parse_decomposition_answer,
)

PROMPTS_DIR = (
    Path(__file__).resolve().parent.parent / "llmprover" / "prover_agents" / "prompts"
)
GOAL = Goal(name="plus_n0", statement="forall n : nat, n + 0 = n.")
NODE = LemmaNode(goal=GOAL)
EXPECTED_SYSTEM = load_prompt(PROMPTS_DIR / "decomposition_system.txt")


def expected_user(polarity: Polarity) -> str:
    return fill_prompt(
        load_prompt(PROMPTS_DIR / "decomposition_user.txt"),
        statement=statement_for_polarity(GOAL.statement, polarity),
    )


def test_parse_decomposition_answer_splits_lemmas_and_script() -> None:
    answer = (
        "plus_n0_base: forall n : nat, 0 + 0 = 0.\n"
        "plus_n0_step: forall n : nat, S n + 0 = S n.\n"
        "\n"
        "induction n.\n- apply plus_n0_base.\n- apply plus_n0_step."
    )

    new_lemmas, script = parse_decomposition_answer(answer)

    assert new_lemmas == [
        Goal(name="plus_n0_base", statement="forall n : nat, 0 + 0 = 0."),
        Goal(name="plus_n0_step", statement="forall n : nat, S n + 0 = S n."),
    ]
    assert script == "induction n.\n- apply plus_n0_base.\n- apply plus_n0_step."


def test_parse_decomposition_answer_strips_fences_and_proof_wrappers() -> None:
    answer = "```\nhelper: True.\n\nProof.\napply I.\nQed.\n```"

    new_lemmas, script = parse_decomposition_answer(answer)

    assert new_lemmas == [Goal(name="helper", statement="True.")]
    assert script == "apply I."


def test_parse_decomposition_answer_raises_when_no_script_section() -> None:
    with pytest.raises(ValueError) as exc_info:
        parse_decomposition_answer("helper: True.")
    assert str(exc_info.value) == "No proof script found"


def test_parse_decomposition_answer_raises_when_lemma_line_has_no_colon() -> None:
    with pytest.raises(ValueError) as exc_info:
        parse_decomposition_answer("helper True.\n\napply I.")
    assert str(exc_info.value) == "No column found in lemma script"


def test_parse_decomposition_answer_raises_when_lemma_is_incomplete() -> None:
    with pytest.raises(ValueError) as exc_info:
        parse_decomposition_answer(": True.\n\napply I.")
    assert str(exc_info.value) == "No statement or name found in lemma script"


def test_prove_returns_lemmas_and_script_from_llm_output() -> None:
    mock_model = MagicMock()
    mock_model.complete.return_value = "helper: True.\n\napply helper."

    attempt = DecompositionAgent(mock_model).prove(NODE, Polarity.Positive)

    assert attempt == ProofAttempt(
        goal=GOAL,
        polarity=Polarity.Positive,
        script="apply helper.",
        new_lemmas=[Goal(name="helper", statement="True.")],
    )
    mock_model.complete.assert_called_once_with(
        [
            {"role": "system", "content": EXPECTED_SYSTEM},
            {"role": "user", "content": expected_user(Polarity.Positive)},
        ]
    )


def test_prove_negative_polarity_uses_negated_statement_in_prompt() -> None:
    mock_model = MagicMock()
    mock_model.complete.return_value = "helper: False.\n\napply helper."

    attempt = DecompositionAgent(mock_model).prove(NODE, Polarity.Negative)

    assert attempt == ProofAttempt(
        goal=GOAL,
        polarity=Polarity.Negative,
        script="apply helper.",
        new_lemmas=[Goal(name="helper", statement="False.")],
    )
    mock_model.complete.assert_called_once_with(
        [
            {"role": "system", "content": EXPECTED_SYSTEM},
            {"role": "user", "content": expected_user(Polarity.Negative)},
        ]
    )


def test_prove_normalizes_fenced_llm_output() -> None:
    mock_model = MagicMock()
    mock_model.complete.return_value = (
        "```\nhelper: True.\n\nProof.\napply helper.\nQed.\n```"
    )

    attempt = DecompositionAgent(mock_model).prove(NODE, Polarity.Positive)

    assert attempt == ProofAttempt(
        goal=GOAL,
        polarity=Polarity.Positive,
        script="apply helper.",
        new_lemmas=[Goal(name="helper", statement="True.")],
    )


def test_prove_propagates_invalid_llm_output() -> None:
    mock_model = MagicMock()
    mock_model.complete.return_value = "helper: True."

    with pytest.raises(ValueError) as exc_info:
        DecompositionAgent(mock_model).prove(NODE, Polarity.Positive)
    assert str(exc_info.value) == "No proof script found"
