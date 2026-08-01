"""Tests for llmprover.utils."""

from __future__ import annotations

from llmprover.domain import (
    AttemptRecord,
    CoqcResult,
    Goal,
    LemmaNode,
    Polarity,
    ProofAttempt,
)
from llmprover.utils import (
    format_attempts,
    format_histories,
    parse_proof_script,
    strip_markdown_fences,
    strip_proof_wrappers,
)

GOAL = Goal(name="plus_n0", statement="forall n : nat, n + 0 = n.")
HELPER = Goal(name="helper", statement="True.")
STEP = Goal(name="step", statement="False.")
EMPTY = "No previous attempts.\n"


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


def child_record(
    goal: Goal,
    *,
    success: bool,
    polarity: Polarity = Polarity.Positive,
    error: str = "",
) -> AttemptRecord:
    return AttemptRecord(
        attempt=ProofAttempt(
            goal=goal,
            polarity=polarity,
            script="exact I." if success else "auto.",
            new_lemmas=[],
        ),
        rocq_error=CoqcResult(success=success, stderr=error),
    )


def test_strip_markdown_fences_removes_fences() -> None:
    text = "```rocq\ninduction n.\n- reflexivity.\n```"
    assert strip_markdown_fences(text) == "induction n.\n- reflexivity."


def test_strip_markdown_fences_leaves_plain_text_unchanged() -> None:
    text = "induction n.\n- reflexivity."
    assert strip_markdown_fences(text) == text


def test_strip_markdown_fences_handles_empty_string() -> None:
    assert strip_markdown_fences("") == ""


def test_strip_proof_wrappers_removes_proof_and_qed() -> None:
    script = "Proof.\ninduction n.\n- reflexivity.\nQed."
    assert strip_proof_wrappers(script) == "induction n.\n- reflexivity."


def test_strip_proof_wrappers_removes_only_qed() -> None:
    script = "induction n.\n- reflexivity.\nQed."
    assert strip_proof_wrappers(script) == "induction n.\n- reflexivity."


def test_strip_proof_wrappers_removes_only_proof() -> None:
    script = "Proof.\ninduction n.\n- reflexivity."
    assert strip_proof_wrappers(script) == "induction n.\n- reflexivity."


def test_strip_proof_wrappers_is_case_insensitive() -> None:
    script = "proof.\napply H.\nqed."
    assert strip_proof_wrappers(script) == "apply H."


def test_strip_proof_wrappers_without_dots() -> None:
    script = "Proof\napply H.\nQed"
    assert strip_proof_wrappers(script) == "apply H."


def test_strip_proof_wrappers_leaves_tactics_unchanged() -> None:
    script = "induction n.\n- reflexivity."
    assert strip_proof_wrappers(script) == script


def test_parse_proof_script_normalizes_fenced_wrapped_output() -> None:
    answer = "```\nProof.\ninduction n.\n- reflexivity.\nQed.\n```"
    assert parse_proof_script(answer) == "induction n.\n- reflexivity."


def test_parse_proof_script_normalizes_wrapped_output() -> None:
    answer = "```rocq\ninduction n.\n- reflexivity.\n```"
    assert parse_proof_script(answer) == "induction n.\n- reflexivity."


def test_parse_proof_script_normalizes_fenced_output() -> None:
    answer = "Proof\ninduction n.\n- reflexivity.\nQed."
    assert parse_proof_script(answer) == "induction n.\n- reflexivity."


def test_parse_proof_script_strips_outer_whitespace() -> None:
    answer = "  induction n.\n- reflexivity.  "
    assert parse_proof_script(answer) == "induction n.\n- reflexivity."


def test_parse_proof_script_returns_empty_string_for_empty_answer() -> None:
    assert parse_proof_script("") == ""


def test_format_attempts_returns_empty_string_by_default() -> None:
    assert format_attempts([]) == ""


def test_format_attempts_returns_custom_empty_message() -> None:
    assert format_attempts([], empty="No previous attempts.\n") == (
        "No previous attempts.\n"
    )


def test_format_attempts_formats_records_fully() -> None:
    records = [
        attempt_record("induction n.", "Error on line 1."),
        attempt_record("auto.", "Unable to unify."),
    ]

    text = format_attempts(records)

    assert text == (
        "Attempt 1:\n"
        "\n"
        "Statement: forall n : nat, n + 0 = n.\n"
        "\n"
        "Script: induction n.\n"
        "\n"
        "Rocq errors: Error on line 1.\n"
        "\n"
        "Attempt 2:\n"
        "\n"
        "Statement: forall n : nat, n + 0 = n.\n"
        "\n"
        "Script: auto.\n"
        "\n"
        "Rocq errors: Unable to unify.\n"
        "\n"
    )


def test_format_attempts_includes_indented_helper_lemmas_with_status() -> None:
    helper_node = LemmaNode(goal=HELPER)
    helper_node.append(child_record(HELPER, success=True))
    step_node = LemmaNode(goal=STEP)
    step_node.append(child_record(STEP, success=True, polarity=Polarity.Negative))

    records = [
        AttemptRecord(
            attempt=ProofAttempt(
                goal=GOAL,
                script="apply helper.",
                new_lemmas=[HELPER, STEP],
            ),
            rocq_error=CoqcResult(success=False, stderr="Error on helper."),
            lemmas=[helper_node, step_node],
        )
    ]

    text = format_attempts(records)

    assert text == (
        "Attempt 1:\n"
        "\n"
        "Statement: forall n : nat, n + 0 = n.\n"
        "\n"
        "Helper lemmas:\n"
        "  helper [proved]: True.\n"
        "  step [refuted]: False.\n"
        "\n"
        "Script: apply helper.\n"
        "\n"
        "Rocq errors: Error on helper.\n"
        "\n"
    )


def test_format_attempts_marks_open_child_lemma_nodes() -> None:
    records = [
        AttemptRecord(
            attempt=ProofAttempt(
                goal=GOAL,
                script="apply helper.",
                new_lemmas=[HELPER],
            ),
            rocq_error=CoqcResult(success=False, stderr="Error on helper."),
            lemmas=[LemmaNode(goal=HELPER)],
        )
    ]

    text = format_attempts(records)

    assert text == (
        "Attempt 1:\n"
        "\n"
        "Statement: forall n : nat, n + 0 = n.\n"
        "\n"
        "Helper lemmas:\n"
        "  helper [open]: True.\n"
        "\n"
        "Script: apply helper.\n"
        "\n"
        "Rocq errors: Error on helper.\n"
        "\n"
    )


def test_format_histories_includes_direct_and_decomposition_attempts() -> None:
    node = LemmaNode(
        goal=GOAL,
        positive=[
            attempt_record("induction n.", "Error on line 1."),
            attempt_record(
                "apply helper.",
                "Error on helper.",
                new_lemmas=[Goal(name="helper", statement="True.")],
            ),
        ],
        negative=[attempt_record("intro H.", "Error B.", polarity=Polarity.Negative)],
    )

    positive_text, negative_text = format_histories(node)

    assert positive_text == (
        "Attempt 1:\n"
        "\n"
        "Statement: forall n : nat, n + 0 = n.\n"
        "\n"
        "Script: induction n.\n"
        "\n"
        "Rocq errors: Error on line 1.\n"
        "\n"
        "Attempt 2:\n"
        "\n"
        "Statement: forall n : nat, n + 0 = n.\n"
        "\n"
        "Helper lemmas:\n"
        "  helper [open]: True.\n"
        "\n"
        "Script: apply helper.\n"
        "\n"
        "Rocq errors: Error on helper.\n"
        "\n"
    )
    assert negative_text == (
        "Attempt 1:\n"
        "\n"
        "Statement: ~ (forall n : nat, n + 0 = n.)\n"
        "\n"
        "Script: intro H.\n"
        "\n"
        "Rocq errors: Error B.\n"
        "\n"
    )


def test_format_histories_returns_placeholders_when_empty() -> None:
    positive_text, negative_text = format_histories(LemmaNode(goal=GOAL))
    assert positive_text == EMPTY
    assert negative_text == EMPTY
