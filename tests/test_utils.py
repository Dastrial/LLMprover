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
    format_attempt,
    format_attempts,
    format_histories,
    normalize_llm_summary,
    number_script_lines,
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
    agent: str = "",
) -> AttemptRecord:
    goals = new_lemmas or []
    return AttemptRecord(
        attempt=ProofAttempt(
            goal=GOAL,
            polarity=polarity,
            script=script,
            new_lemmas=goals,
            agent=agent,
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


# --- strip_markdown_fences ---


def test_strip_markdown_fences_removes_fences() -> None:
    text = "```rocq\ninduction n.\n- reflexivity.\n```"
    assert strip_markdown_fences(text) == "induction n.\n- reflexivity."


def test_strip_markdown_fences_leaves_plain_text_unchanged() -> None:
    text = "induction n.\n- reflexivity."
    assert strip_markdown_fences(text) == text


def test_strip_markdown_fences_handles_empty_string() -> None:
    assert strip_markdown_fences("") == ""


# --- strip_proof_wrappers ---


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


# --- parse_proof_script ---


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


# --- number_script_lines ---


def test_number_script_lines_prefixes_1_based_markers() -> None:
    assert number_script_lines("intros x.\napply foo.") == (
        "1 | intros x.\n2 | apply foo."
    )


# --- format_attempt ---


def test_format_attempt_uses_remapped_loci_not_original_stderr() -> None:
    record = AttemptRecord(
        attempt=ProofAttempt(goal=GOAL, script="oops.", new_lemmas=[]),
        rocq_error=CoqcResult(
            success=False,
            stderr=(
                'File "/tmp/check.v", line 4, characters 2-6:\n'
                "Error: The reference oops was not found in the current environment.\n"
            ),
            script_stderr=(
                "Script line 2, characters 2-6:\n"
                "Error: The reference oops was not found in the current environment.\n"
            ),
        ),
    )

    assert format_attempt(record, 1) == (
        "Attempt 1:\n"
        "\n"
        "Statement: forall n : nat, n + 0 = n.\n"
        "\n"
        "Script:\n"
        "1 | oops.\n"
        "\n"
        "Rocq errors: Script line 2, characters 2-6:\n"
        "Error: The reference oops was not found in the current environment.\n"
        "\n"
    )


# --- format_attempts ---


def test_format_attempts_shows_placeholder_for_empty_script() -> None:
    text = format_attempts([attempt_record("", "Error: incomplete proof.")])
    assert text == (
        "Attempt 1:\n"
        "\n"
        "Statement: forall n : nat, n + 0 = n.\n"
        "\n"
        "Script: (empty)\n"
        "\n"
        "Rocq errors: Error: incomplete proof.\n"
        "\n"
    )


def test_format_attempts_strips_coqc_warnings() -> None:
    error = (
        "Script line 1, characters 0-3:\n"
        "Warning: deprecated\n"
        "[deprecated,default]\n"
        "Script line 2, characters 0-3:\n"
        "Error: boom"
    )
    text = format_attempts([attempt_record("auto.", error)])
    assert text == (
        "Attempt 1:\n"
        "\n"
        "Statement: forall n : nat, n + 0 = n.\n"
        "\n"
        "Script:\n"
        "1 | auto.\n"
        "\n"
        "Rocq errors: Script line 2, characters 0-3:\n"
        "Error: boom\n"
        "\n"
    )


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
        "Script:\n"
        "1 | induction n.\n"
        "\n"
        "Rocq errors: Error on line 1.\n"
        "\n"
        "Attempt 2:\n"
        "\n"
        "Statement: forall n : nat, n + 0 = n.\n"
        "\n"
        "Script:\n"
        "1 | auto.\n"
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
        "Script:\n"
        "1 | apply helper.\n"
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
        "Script:\n"
        "1 | apply helper.\n"
        "\n"
        "Rocq errors: Error on helper.\n"
        "\n"
    )


def test_format_attempts_includes_agent_when_requested() -> None:
    records = [
        attempt_record(
            "induction n.",
            "Error on line 1.",
            agent="DirectAgent, model=gpt-4o-mini, reasoning_tokens=0",
        )
    ]

    assert format_attempts(records) == (
        "Attempt 1:\n"
        "\n"
        "Statement: forall n : nat, n + 0 = n.\n"
        "\n"
        "Script:\n"
        "1 | induction n.\n"
        "\n"
        "Rocq errors: Error on line 1.\n"
        "\n"
    )
    assert format_attempts(records, include_agent=True) == (
        "Attempt 1:\n"
        "\n"
        "Statement: forall n : nat, n + 0 = n.\n"
        "\n"
        "Agent: DirectAgent, model=gpt-4o-mini, reasoning_tokens=0\n"
        "\n"
        "Script:\n"
        "1 | induction n.\n"
        "\n"
        "Rocq errors: Error on line 1.\n"
        "\n"
    )


def test_format_attempts_numbers_multiline_script() -> None:
    text = format_attempts(
        [attempt_record("intros x.\napply foo.", "Error on line 2.")]
    )
    assert text == (
        "Attempt 1:\n"
        "\n"
        "Statement: forall n : nat, n + 0 = n.\n"
        "\n"
        "Script:\n"
        "1 | intros x.\n"
        "2 | apply foo.\n"
        "\n"
        "Rocq errors: Error on line 2.\n"
        "\n"
    )


def test_format_attempts_shows_resolved_child_names() -> None:
    shared = LemmaNode(goal=Goal(name="base", statement="True."))
    records = [
        AttemptRecord(
            attempt=ProofAttempt(
                goal=GOAL,
                script="exact {dup}.",
                new_lemmas=[Goal(name="dup", statement="True.")],
            ),
            rocq_error=CoqcResult(success=False, stderr="Error on helper."),
            lemmas=[shared],
        )
    ]

    assert format_attempts(records) == (
        "Attempt 1:\n"
        "\n"
        "Statement: forall n : nat, n + 0 = n.\n"
        "\n"
        "Helper lemmas:\n"
        "  base [open]: True.\n"
        "\n"
        "Script:\n"
        "1 | exact base.\n"
        "\n"
        "Rocq errors: Error on helper.\n"
        "\n"
    )
    assert records[0].attempt.script == "exact {dup}."


# --- format_histories ---


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
        "Script:\n"
        "1 | induction n.\n"
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
        "Script:\n"
        "1 | apply helper.\n"
        "\n"
        "Rocq errors: Error on helper.\n"
        "\n"
    )
    assert negative_text == (
        "Attempt 1:\n"
        "\n"
        "Statement: ~ (forall n : nat, n + 0 = n.)\n"
        "\n"
        "Script:\n"
        "1 | intro H.\n"
        "\n"
        "Rocq errors: Error B.\n"
        "\n"
    )


def test_format_histories_returns_placeholders_when_empty() -> None:
    positive_text, negative_text = format_histories(LemmaNode(goal=GOAL))
    assert positive_text == EMPTY
    assert negative_text == EMPTY


def test_format_histories_can_include_agent() -> None:
    node = LemmaNode(
        goal=GOAL,
        positive=[
            attempt_record(
                "induction n.",
                "Error on line 1.",
                agent="DirectAgent, model=gpt-5.6-luna reasoning=none, reasoning_tokens=4",
            )
        ],
    )

    positive_text, _ = format_histories(node, include_agent=True)
    assert positive_text == (
        "Attempt 1:\n"
        "\n"
        "Statement: forall n : nat, n + 0 = n.\n"
        "\n"
        "Agent: DirectAgent, model=gpt-5.6-luna reasoning=none, reasoning_tokens=4\n"
        "\n"
        "Script:\n"
        "1 | induction n.\n"
        "\n"
        "Rocq errors: Error on line 1.\n"
        "\n"
    )
    omitted, _ = format_histories(node)
    assert omitted == (
        "Attempt 1:\n"
        "\n"
        "Statement: forall n : nat, n + 0 = n.\n"
        "\n"
        "Script:\n"
        "1 | induction n.\n"
        "\n"
        "Rocq errors: Error on line 1.\n"
        "\n"
    )


# --- normalize_llm_summary ---


def test_normalize_llm_summary_returns_empty_when_blank() -> None:
    assert normalize_llm_summary("") == EMPTY


def test_normalize_llm_summary_returns_empty_for_empty_markdown_fence() -> None:
    assert normalize_llm_summary("```markdown\n```") == EMPTY


def test_normalize_llm_summary_removes_fences_and_returns_cleaned() -> None:
    assert normalize_llm_summary("```markdown\nfirst blocker\n```") == (
        "first blocker\n"
    )


def test_normalize_llm_summary_adds_trailing_newline_when_missing() -> None:
    assert normalize_llm_summary("first blocker") == "first blocker\n"
