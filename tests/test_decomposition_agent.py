"""Tests for llmprover.prover_agents.decomposition_agent."""

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
    RocqEnvironment,
    statement_for_polarity,
)
from llmprover.history_presenter import DeterministicHistoryPresenter
from llmprover.llm_client import CompletionResult, TokenUsage
from llmprover.prompts import (
    PromptMessage,
    PromptPart,
    fill_prompt,
)
from llmprover.prover_agents.decomposition_agent import (
    DecompositionAgent,
    parse_decomposition_answer,
)
from llmprover.prover_agents.prompt_assembly import (
    DECOMPOSITION_SPEC,
    cached_system_prompt,
)

PROMPTS_DIR = (
    Path(__file__).resolve().parent.parent / "llmprover" / "prover_agents" / "prompts"
)
GOAL = Goal(name="plus_n0", statement="forall n : nat, n + 0 = n.")
NODE = LemmaNode(goal=GOAL)
EXPECTED_SYSTEM = cached_system_prompt(DECOMPOSITION_SPEC)
EMPTY = "No previous attempts.\n"


def decomposition_output(*helpers: str, script: str = "") -> str:
    parts = ["SCRIPT:"]
    if script:
        parts.append(script)
    if helpers:
        parts.append("HELPERS:")
        parts.extend(helpers)
    return "\n".join(parts) + "\n"


def old_decomposition_output(*helpers: str, script: str = "") -> str:
    helpers_block = "\n".join(helpers)
    if helpers_block:
        helpers_block += "\n"
    script_block = f"{script}\n" if script else ""
    return f"HELPERS:\n{helpers_block}SCRIPT:\n{script_block}"


def attempt_record(
    script: str,
    error: str,
    *,
    polarity: Polarity = Polarity.Positive,
    new_lemmas: list[Goal] | None = None,
) -> AttemptRecord:
    goals = new_lemmas or []
    return AttemptRecord(
        attempt=ProofAttempt(
            goal=GOAL,
            polarity=polarity,
            script=script,
            new_lemmas=goals,
        ),
        rocq_error=CoqcResult(success=False, stderr=error),
        lemmas=[LemmaNode(goal=goal) for goal in goals],
    )


def expected_messages(
    polarity: Polarity,
    node: LemmaNode | None = None,
) -> list[PromptMessage]:
    node = node or NODE
    attempt_history, _ = DeterministicHistoryPresenter.full().present_prompt_block(node)
    before = fill_prompt(
        (PROMPTS_DIR / "decomposition_user_before.txt").read_text(encoding="utf-8").strip(),
        header=GOAL.environment.header,
    )
    if not before.endswith("\n"):
        before = f"{before}\n"
    after = fill_prompt(
        (PROMPTS_DIR / "decomposition_user_after.txt").read_text(encoding="utf-8").strip(),
        statement=statement_for_polarity(GOAL.statement, polarity),
    )
    if after and not after.endswith("\n"):
        after = f"{after}\n"
    history = (
        attempt_history if attempt_history.endswith("\n") else f"{attempt_history}\n"
    )
    return [
        PromptMessage.text("system", EXPECTED_SYSTEM, cache_breakpoint=True),
        PromptMessage(
            role="user",
            parts=(PromptPart(before + history, cache_breakpoint=True), PromptPart(after)),
        ),
    ]


def expected_user(polarity: Polarity, node: LemmaNode | None = None) -> str:
    return expected_messages(polarity, node)[1].joined_text()


def test_parse_decomposition_answer_splits_lemmas_and_script() -> None:
    answer = decomposition_output(
        "plus_n0_base: forall n : nat, 0 + 0 = 0.",
        "plus_n0_step: forall n : nat, S n + 0 = S n.",
        script="induction n.\n- apply {plus_n0_base}.\n- apply {plus_n0_step}.",
    )

    new_lemmas, script = parse_decomposition_answer(answer)

    assert new_lemmas == [
        Goal(name="plus_n0_base", statement="forall n : nat, 0 + 0 = 0."),
        Goal(name="plus_n0_step", statement="forall n : nat, S n + 0 = S n."),
    ]
    assert script == (
        "induction n.\n- apply {plus_n0_base}.\n- apply {plus_n0_step}."
    )


def test_parse_decomposition_answer_ignores_blank_lines_between_helpers() -> None:
    answer = (
        "SCRIPT:\n"
        "apply I.\n"
        "HELPERS:\n"
        "plus_n0_base: forall n : nat, 0 + 0 = 0.\n"
        "  \n"
        "\t\n"
        "\n"
        "plus_n0_step: forall n : nat, S n + 0 = S n.\n"
    )

    new_lemmas, script = parse_decomposition_answer(answer)

    assert new_lemmas == [
        Goal(name="plus_n0_base", statement="forall n : nat, 0 + 0 = 0."),
        Goal(name="plus_n0_step", statement="forall n : nat, S n + 0 = S n."),
    ]
    assert script == "apply I."


def test_parse_decomposition_answer_joins_wrapped_helper_statements() -> None:
    answer = (
        "SCRIPT:\n"
        "apply I.\n"
        "HELPERS:\n"
        "plus_n0_base : forall n : nat,\n"
        "  0 + 0 = 0.\n"
        "\n"
        "plus_n0_step : forall n : nat,\n"
        "S n + 0 = S n.\n"
    )

    new_lemmas, script = parse_decomposition_answer(answer)

    assert new_lemmas == [
        Goal(name="plus_n0_base", statement="forall n : nat, 0 + 0 = 0."),
        Goal(name="plus_n0_step", statement="forall n : nat, S n + 0 = S n."),
    ]
    assert script == "apply I."


def test_parse_decomposition_answer_strips_fences_and_proof_wrappers() -> None:
    answer = "```\nSCRIPT:\nProof.\napply I.\nQed.\nHELPERS:\nhelper: True.\n```"

    new_lemmas, script = parse_decomposition_answer(answer)

    assert new_lemmas == [Goal(name="helper", statement="True.")]
    assert script == "apply I."


def test_parse_decomposition_answer_inherits_environment() -> None:
    env = RocqEnvironment(
        header="Require Import Reals.",
        allowed_axioms=frozenset({"Classical.classic"}),
    )

    new_lemmas, _ = parse_decomposition_answer(
        decomposition_output("helper: True.", script="exact I."),
        environment=env,
    )

    assert len(new_lemmas) == 1
    assert new_lemmas[0].environment is env


def test_parse_decomposition_answer_empty_helpers_when_no_helpers_section() -> None:
    new_lemmas, script = parse_decomposition_answer("SCRIPT:\napply I.")
    assert new_lemmas == []
    assert script == "apply I."


def test_parse_decomposition_answer_empty_script_when_no_script_section() -> None:
    new_lemmas, script = parse_decomposition_answer("HELPERS:\nhelper: True.")
    assert new_lemmas == [Goal(name="helper", statement="True.")]
    assert script == ""


def test_parse_decomposition_answer_accepts_legacy_helpers_then_script_order() -> None:
    new_lemmas, script = parse_decomposition_answer(
        old_decomposition_output("helper: True.", script="apply I.")
    )
    assert new_lemmas == [Goal(name="helper", statement="True.")]
    assert script == "apply I."


def test_parse_decomposition_answer_does_not_treat_tactic_colon_as_lemma() -> None:
    new_lemmas, script = parse_decomposition_answer(
        "HELPERS:\nSCRIPT:\nassert (H: True).\nexact I."
    )
    assert new_lemmas == []
    assert script == "assert (H: True).\nexact I."


def test_parse_decomposition_answer_skips_lemma_line_without_colon() -> None:
    new_lemmas, script = parse_decomposition_answer(
        decomposition_output("helper True.", script="apply I.")
    )
    assert new_lemmas == []
    assert script == "apply I."


def test_parse_decomposition_answer_skips_incomplete_lemma() -> None:
    new_lemmas, script = parse_decomposition_answer(
        decomposition_output(": True.", script="apply I.")
    )
    assert new_lemmas == []
    assert script == "apply I."


def test_parse_decomposition_answer_treats_script_only_output_as_script() -> None:
    new_lemmas, script = parse_decomposition_answer("induction n.\nreflexivity.")
    assert new_lemmas == []
    assert script == "induction n.\nreflexivity."


def test_parse_decomposition_answer_helpers_marker_without_blank_separator() -> None:
    new_lemmas, script = parse_decomposition_answer(
        "SCRIPT:\napply helper.\nHELPERS:\nhelper: True."
    )
    assert new_lemmas == [Goal(name="helper", statement="True.")]
    assert script == "apply helper."


def test_parse_decomposition_answer_treats_unmarked_tactic_colon_as_script() -> None:
    new_lemmas, script = parse_decomposition_answer(
        "SCRIPT:\nassert (H: True).\nexact I."
    )
    assert new_lemmas == []
    assert script == "assert (H: True).\nexact I."


def test_prove_returns_lemmas_and_script_from_llm_output() -> None:
    mock_model = MagicMock()
    mock_model.complete.return_value = CompletionResult(
        text=decomposition_output("helper: True.", script="apply {helper}."),
        usage=TokenUsage(3, 5),
    )

    attempt, usage = DecompositionAgent(mock_model).prove(NODE, Polarity.Positive)

    assert attempt == ProofAttempt(
        goal=GOAL,
        polarity=Polarity.Positive,
        script="apply {helper}.",
        new_lemmas=[Goal(name="helper", statement="True.")],
    )
    assert usage == TokenUsage(3, 5)
    mock_model.complete.assert_called_once_with(
        expected_messages(Polarity.Positive)
    )


def test_prove_negative_polarity_uses_negated_statement_in_prompt() -> None:
    mock_model = MagicMock()
    mock_model.complete.return_value = CompletionResult(
        text=decomposition_output("helper: False.", script="apply {helper}."),
        usage=TokenUsage(3, 5),
    )

    attempt, usage = DecompositionAgent(mock_model).prove(NODE, Polarity.Negative)

    assert attempt == ProofAttempt(
        goal=GOAL,
        polarity=Polarity.Negative,
        script="apply {helper}.",
        new_lemmas=[Goal(name="helper", statement="False.")],
    )
    assert usage == TokenUsage(3, 5)
    mock_model.complete.assert_called_once_with(
        expected_messages(Polarity.Negative)
    )


def test_prove_positive_includes_mixed_history() -> None:
    mock_model = MagicMock()
    mock_model.complete.return_value = CompletionResult(
        text=decomposition_output("helper: True.", script="apply {helper}."),
        usage=TokenUsage(3, 5),
    )
    node = LemmaNode(
        goal=GOAL,
        positive=[
            attempt_record("induction n.", "Error on line 1."),
            attempt_record(
                "apply old_helper.",
                "Still failing.",
                new_lemmas=[Goal(name="old_helper", statement="False.")],
            ),
        ],
        negative=[attempt_record("intro H.", "Error B.", polarity=Polarity.Negative)],
    )

    attempt, usage = DecompositionAgent(mock_model).prove(node, Polarity.Positive)

    assert attempt == ProofAttempt(
        goal=GOAL,
        polarity=Polarity.Positive,
        script="apply {helper}.",
        new_lemmas=[Goal(name="helper", statement="True.")],
    )
    assert usage == TokenUsage(3, 5)
    mock_model.complete.assert_called_once_with(
        expected_messages(Polarity.Positive, node)
    )
    user = mock_model.complete.call_args.args[0][1].joined_text()
    assert "old_helper" in user
    assert user.index("induction n.") < user.index("intro H.")


def test_prove_omits_agent_description_from_history() -> None:
    mock_model = MagicMock()
    mock_model.complete.return_value = CompletionResult(
        text=decomposition_output("helper: True.", script="apply {helper}."),
        usage=TokenUsage(3, 5),
    )
    node = LemmaNode(
        goal=GOAL,
        positive=[attempt_record("induction n.", "Error on line 1.")],
    )
    node.positive[0].attempt.agent = "DirectAgent, model=gpt-4o-mini"

    DecompositionAgent(mock_model).prove(node, Polarity.Positive)

    user_prompt = mock_model.complete.call_args.args[0][1].joined_text()
    assert "Agent:" not in user_prompt
    assert "DirectAgent" not in user_prompt


def test_prove_negative_keeps_same_history_order() -> None:
    mock_model = MagicMock()
    mock_model.complete.return_value = CompletionResult(
        text=decomposition_output("helper: False.", script="apply {helper}."),
        usage=TokenUsage(3, 5),
    )
    node = LemmaNode(
        goal=GOAL,
        positive=[attempt_record("induction n.", "Error on line 1.")],
        negative=[attempt_record("intro H.", "Error B.", polarity=Polarity.Negative)],
    )

    attempt, usage = DecompositionAgent(mock_model).prove(node, Polarity.Negative)

    assert attempt == ProofAttempt(
        goal=GOAL,
        polarity=Polarity.Negative,
        script="apply {helper}.",
        new_lemmas=[Goal(name="helper", statement="False.")],
    )
    assert usage == TokenUsage(3, 5)
    mock_model.complete.assert_called_once_with(
        expected_messages(Polarity.Negative, node)
    )
    user = mock_model.complete.call_args.args[0][1].joined_text()
    assert user.index("induction n.") < user.index("intro H.")


def test_prove_normalizes_fenced_llm_output() -> None:
    mock_model = MagicMock()
    mock_model.complete.return_value = CompletionResult(
        text=(
            "```\nSCRIPT:\nProof.\napply {helper}.\nQed.\nHELPERS:\nhelper: True.\n```"
        ),
        usage=TokenUsage(3, 5),
    )

    attempt, usage = DecompositionAgent(mock_model).prove(NODE, Polarity.Positive)

    assert attempt == ProofAttempt(
        goal=GOAL,
        polarity=Polarity.Positive,
        script="apply {helper}.",
        new_lemmas=[Goal(name="helper", statement="True.")],
    )
    assert usage == TokenUsage(3, 5)


def test_prove_returns_empty_helpers_when_llm_omits_helpers() -> None:
    mock_model = MagicMock()
    mock_model.complete.return_value = CompletionResult(
        text="SCRIPT:\napply I.", usage=TokenUsage(3, 5)
    )

    attempt, usage = DecompositionAgent(mock_model).prove(NODE, Polarity.Positive)

    assert attempt == ProofAttempt(
        goal=GOAL,
        polarity=Polarity.Positive,
        script="apply I.",
        new_lemmas=[],
    )
    assert usage == TokenUsage(3, 5)


def test_prove_returns_empty_script_when_llm_omits_script() -> None:
    mock_model = MagicMock()
    mock_model.complete.return_value = CompletionResult(
        text="HELPERS:\nhelper: True.", usage=TokenUsage(3, 5)
    )

    attempt, usage = DecompositionAgent(mock_model).prove(NODE, Polarity.Positive)

    assert attempt == ProofAttempt(
        goal=GOAL,
        polarity=Polarity.Positive,
        script="",
        new_lemmas=[Goal(name="helper", statement="True.")],
    )
    assert usage == TokenUsage(3, 5)


def test_describe_includes_model_and_reasoning_effort() -> None:
    mock_model = MagicMock()
    mock_model.model = "gpt-5.6-luna"
    mock_model.reasoning_effort = "none"

    description = DecompositionAgent(mock_model).describe(TokenUsage(3, 5, 4))

    assert description == (
        "DecompositionAgent, model=gpt-5.6-luna reasoning=none"
    )


def test_describe_omits_reasoning_effort_when_unset() -> None:
    mock_model = MagicMock()
    mock_model.model = "gpt-4o-mini"
    mock_model.reasoning_effort = None

    assert DecompositionAgent(mock_model).describe(TokenUsage()) == (
        "DecompositionAgent, model=gpt-4o-mini"
    )
