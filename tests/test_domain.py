"""Tests for llmprover.domain."""

from __future__ import annotations

import pytest

from llmprover.domain import (
    AttemptRecord,
    CoqcResult,
    Goal,
    LemmaNode,
    LemmaStatus,
    Polarity,
    ProofAttempt,
    statement_for_polarity,
    status_from_frontiers,
)

GOAL = Goal(name="plus_n0", statement="forall n : nat, n + 0 = n.")


def record(
    script: str,
    *,
    success: bool,
    polarity: Polarity = Polarity.Positive,
    error: str = "Error.",
    goal: Goal = GOAL,
) -> AttemptRecord:
    return AttemptRecord(
        attempt=ProofAttempt(
            goal=goal,
            script=script,
            new_lemmas=[],
            polarity=polarity,
        ),
        rocq_error=CoqcResult(
            success=success,
            stderr="" if success else error,
        ),
    )


# --- CoqcResult ---


def test_coqc_result_output_combines_stdout_and_stderr() -> None:
    result = CoqcResult(success=False, stdout="line1", stderr="line2")
    assert result.output == "line1\nline2"


def test_coqc_result_output_empty_when_no_output() -> None:
    result = CoqcResult(success=True)
    assert result.output == ""


def test_coqc_result_output_no_stdout() -> None:
    result = CoqcResult(success=False, stderr="line1")
    assert result.output == "line1"


def test_coqc_result_output_no_stderr() -> None:
    result = CoqcResult(success=False, stdout="line1")
    assert result.output == "line1"


def test_coqc_result_str() -> None:
    str_result = str(CoqcResult(success=False, stdout="line1", stderr="line2"))
    assert str_result == "CoqcResult(success=False, stdout=line1, stderr=line2)"


# --- Polarity / statement_for_polarity ---


def test_statement_for_polarity_positive_keeps_statement() -> None:
    assert statement_for_polarity(GOAL.statement, Polarity.Positive) == GOAL.statement


def test_statement_for_polarity_negative_wraps_with_not() -> None:
    assert statement_for_polarity(GOAL.statement, Polarity.Negative) == (
        "~ (forall n : nat, n + 0 = n.)"
    )


def test_proof_attempt_defaults_to_positive_polarity() -> None:
    attempt = ProofAttempt(goal=GOAL, script="reflexivity.", new_lemmas=[])
    assert attempt.polarity is Polarity.Positive


def test_proof_attempt_target_statement_follows_polarity() -> None:
    positive = ProofAttempt(
        goal=GOAL,
        script="reflexivity.",
        new_lemmas=[],
        polarity=Polarity.Positive,
    )
    negative = ProofAttempt(
        goal=GOAL,
        script="intro H.",
        new_lemmas=[],
        polarity=Polarity.Negative,
    )
    assert positive.target_statement == GOAL.statement
    assert negative.target_statement == "~ (forall n : nat, n + 0 = n.)"


# --- status_from_frontiers ---


def test_status_from_frontiers_open_when_both_missing() -> None:
    assert status_from_frontiers(None, None) is LemmaStatus.Open


def test_status_from_frontiers_proved_prefers_positive() -> None:
    assert (
        status_from_frontiers(
            record("reflexivity.", success=True),
            record("intro H.", success=True, polarity=Polarity.Negative),
        )
        is LemmaStatus.Proved
    )


def test_status_from_frontiers_refuted_when_only_negative_succeeds() -> None:
    assert (
        status_from_frontiers(
            record("auto.", success=False),
            record("intro H.", success=True, polarity=Polarity.Negative),
        )
        is LemmaStatus.Refuted
    )


# --- LemmaNode ---


def test_lemma_node_starts_open_with_empty_histories() -> None:
    node = LemmaNode(goal=GOAL)
    assert node.positive == []
    assert node.negative == []
    assert node.status is LemmaStatus.Open
    assert node.frontier(Polarity.Positive) is None
    assert node.frontier(Polarity.Negative) is None


def test_lemma_node_append_routes_by_polarity() -> None:
    node = LemmaNode(goal=GOAL)
    pos = record("auto.", success=False)
    neg = record("intro H.", success=False, polarity=Polarity.Negative)

    node.append(pos)
    node.append(neg)

    assert node.positive == [pos]
    assert node.negative == [neg]
    assert node.history(Polarity.Positive) == [pos]
    assert node.history(Polarity.Negative) == [neg]
    assert node.frontier(Polarity.Positive) is pos
    assert node.frontier(Polarity.Negative) is neg
    assert node.status is LemmaStatus.Open


def test_lemma_node_status_proved_when_latest_positive_succeeds() -> None:
    node = LemmaNode(goal=GOAL)
    node.append(record("auto.", success=False))
    node.append(record("reflexivity.", success=True))
    assert node.status is LemmaStatus.Proved


def test_lemma_node_status_refuted_when_latest_negative_succeeds() -> None:
    node = LemmaNode(goal=GOAL)
    node.append(record("auto.", success=False))
    node.append(
        record("intro H. contradiction.", success=True, polarity=Polarity.Negative)
    )
    assert node.status is LemmaStatus.Refuted


def test_lemma_node_status_refuted_after_failed_positives_and_mixed_negatives() -> None:
    node = LemmaNode(goal=GOAL)
    node.append(record("auto.", success=False))
    node.append(record("induction n.", success=False))
    node.append(record("intro H.", success=False, polarity=Polarity.Negative))
    node.append(
        record("intro H. contradiction.", success=True, polarity=Polarity.Negative)
    )
    assert node.status is LemmaStatus.Refuted


def test_lemma_node_append_rejects_mismatched_goal() -> None:
    node = LemmaNode(goal=GOAL)
    other = AttemptRecord(
        attempt=ProofAttempt(
            goal=Goal(name="other", statement="True."),
            script="exact I.",
            new_lemmas=[],
        ),
        rocq_error=CoqcResult(success=True),
    )
    with pytest.raises(ValueError) as exc_info:
        node.append(other)
    assert str(exc_info.value) == (
        f"Attempt goal {other.attempt.goal!r} does not match lemma node goal {GOAL!r}"
    )


def test_attempt_record_can_nest_child_lemma_nodes() -> None:
    helper = Goal(name="helper", statement="True.")
    child = LemmaNode(goal=helper)
    child.append(record("exact I.", success=True, goal=helper))

    parent = AttemptRecord(
        attempt=ProofAttempt(
            goal=GOAL,
            script="apply helper.",
            new_lemmas=[helper],
        ),
        rocq_error=CoqcResult(success=False, stderr="Error."),
        lemmas=[child],
    )

    assert parent.lemmas[0].status is LemmaStatus.Proved
    assert parent.lemmas[0].goal == helper
    assert parent.fully_succeeded is False


def test_lemma_node_status_open_when_coqc_ok_but_child_still_open() -> None:
    helper = Goal(name="helper", statement="True.")
    child = LemmaNode(goal=helper)
    node = LemmaNode(goal=GOAL)
    node.append(
        AttemptRecord(
            attempt=ProofAttempt(
                goal=GOAL,
                script="apply helper.",
                new_lemmas=[helper],
            ),
            rocq_error=CoqcResult(success=True),
            lemmas=[child],
        )
    )
    assert child.status is LemmaStatus.Open
    assert node.status is LemmaStatus.Open


def test_lemma_node_status_proved_when_coqc_ok_and_children_proved() -> None:
    helper = Goal(name="helper", statement="True.")
    child = LemmaNode(goal=helper)
    child.append(record("exact I.", success=True, goal=helper))
    node = LemmaNode(goal=GOAL)
    node.append(
        AttemptRecord(
            attempt=ProofAttempt(
                goal=GOAL,
                script="apply helper.",
                new_lemmas=[helper],
            ),
            rocq_error=CoqcResult(success=True),
            lemmas=[child],
        )
    )
    assert child.status is LemmaStatus.Proved
    assert node.status is LemmaStatus.Proved


def test_lemma_node_status_open_when_coqc_ok_but_one_child_refuted() -> None:
    proved = Goal(name="ok", statement="True.")
    refuted = Goal(name="bad", statement="False.")
    proved_child = LemmaNode(goal=proved)
    proved_child.append(record("exact I.", success=True, goal=proved))
    refuted_child = LemmaNode(goal=refuted)
    refuted_child.append(
        record("intro H.", success=True, polarity=Polarity.Negative, goal=refuted)
    )
    node = LemmaNode(goal=GOAL)
    node.append(
        AttemptRecord(
            attempt=ProofAttempt(
                goal=GOAL,
                script="apply ok. apply bad.",
                new_lemmas=[proved, refuted],
            ),
            rocq_error=CoqcResult(success=True),
            lemmas=[proved_child, refuted_child],
        )
    )
    assert proved_child.status is LemmaStatus.Proved
    assert refuted_child.status is LemmaStatus.Refuted
    assert node.status is LemmaStatus.Open


def test_lemma_node_status_open_when_coqc_fails_and_one_child_refuted() -> None:
    proved = Goal(name="ok", statement="True.")
    refuted = Goal(name="bad", statement="False.")
    proved_child = LemmaNode(goal=proved)
    proved_child.append(record("exact I.", success=True, goal=proved))
    refuted_child = LemmaNode(goal=refuted)
    refuted_child.append(
        record("intro H.", success=True, polarity=Polarity.Negative, goal=refuted)
    )
    node = LemmaNode(goal=GOAL)
    node.append(
        AttemptRecord(
            attempt=ProofAttempt(
                goal=GOAL,
                script="apply ok. apply bad.",
                new_lemmas=[proved, refuted],
            ),
            rocq_error=CoqcResult(success=False, stderr="Error."),
            lemmas=[proved_child, refuted_child],
        )
    )
    assert proved_child.status is LemmaStatus.Proved
    assert refuted_child.status is LemmaStatus.Refuted
    assert node.status is LemmaStatus.Open


def test_lemma_node_status_open_when_negative_coqc_ok_but_child_open() -> None:
    helper = Goal(name="helper", statement="False.")
    child = LemmaNode(goal=helper)
    node = LemmaNode(goal=GOAL)
    node.append(
        AttemptRecord(
            attempt=ProofAttempt(
                goal=GOAL,
                polarity=Polarity.Negative,
                script="apply helper.",
                new_lemmas=[helper],
            ),
            rocq_error=CoqcResult(success=True),
            lemmas=[child],
        )
    )
    assert node.status is LemmaStatus.Open


def test_lemma_node_status_refuted_when_negative_coqc_ok_and_children_proved() -> None:
    helper = Goal(name="helper", statement="False.")
    child = LemmaNode(goal=helper)
    child.append(record("exact I.", success=True, goal=helper))
    node = LemmaNode(goal=GOAL)
    node.append(
        AttemptRecord(
            attempt=ProofAttempt(
                goal=GOAL,
                polarity=Polarity.Negative,
                script="apply helper.",
                new_lemmas=[helper],
            ),
            rocq_error=CoqcResult(success=True),
            lemmas=[child],
        )
    )
    assert node.status is LemmaStatus.Refuted
