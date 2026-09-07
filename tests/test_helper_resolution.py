"""Tests for llmprover.helper_resolution."""

from __future__ import annotations

from llmprover.coqc_output import CoqcResult
from llmprover.domain import (
    AttemptRecord,
    Goal,
    LemmaNode,
    ProofAttempt,
)
from llmprover.helper_resolution import (
    bare_new_helper_names,
    format_bare_helper_error,
    new_helper_protocol_error,
    resolve_attempt_helpers,
    script_with_child_names,
    substitute_helper_placeholders,
)

GOAL = Goal(name="plus_n0", statement="forall n : nat, n + 0 = n.")


# --- substitute_helper_placeholders ---


def test_substitute_helper_placeholders_rewrites_known_names() -> None:
    script = "exact {helper}. apply {helper_1}."
    rewritten = substitute_helper_placeholders(
        script, {"helper": "base", "helper_1": "step"}
    )
    assert rewritten == "exact base. apply step."


def test_substitute_helper_placeholders_keeps_unrelated_braces() -> None:
    script = "refine {x : nat | x > 0}."
    assert substitute_helper_placeholders(script, {"helper": "base"}) == script


# --- bare_new_helper_names ---


def test_bare_new_helper_names_accepts_placeholder() -> None:
    assert bare_new_helper_names("apply {helper}.", ["helper"]) == []


def test_bare_new_helper_names_rejects_bare_identifier() -> None:
    assert bare_new_helper_names("apply helper.", ["helper"]) == ["helper"]


def test_bare_new_helper_names_ignores_longer_identifier() -> None:
    assert bare_new_helper_names("apply helper_2.", ["helper"]) == []
    assert bare_new_helper_names("apply foobar.", ["foo"]) == []


def test_bare_new_helper_names_keeps_unrelated_rocq_braces() -> None:
    assert bare_new_helper_names("refine {apply helper}.", ["helper"]) == ["helper"]


def test_bare_new_helper_names_reports_only_faulty_helpers() -> None:
    script = "split. apply {ok}. apply bad."
    assert bare_new_helper_names(script, ["ok", "bad"]) == ["bad"]


# --- format_bare_helper_error ---


def test_format_bare_helper_error_single_helper() -> None:
    assert format_bare_helper_error(["intermediate"]) == (
        "Invalid helper reference: newly proposed helper 'intermediate' "
        "must be referenced as '{intermediate}' in SCRIPT, "
        "not as bare 'intermediate'."
    )


def test_format_bare_helper_error_lists_multiple_helpers() -> None:
    assert format_bare_helper_error(["left", "mid", "right"]) == (
        "Invalid helper reference: newly proposed helpers 'left', 'mid', 'right' "
        "must be referenced as '{left}', '{mid}', '{right}' in SCRIPT, "
        "not as bare identifiers."
    )


# --- new_helper_protocol_error ---


def test_new_helper_protocol_error_only_checks_current_new_lemmas() -> None:
    attempt = ProofAttempt(
        goal=GOAL,
        script="apply old_helper. apply {fresh}.",
        new_lemmas=[Goal(name="fresh", statement="True.")],
    )
    assert new_helper_protocol_error(attempt) is None


def test_new_helper_protocol_error_message_lists_bare_helpers() -> None:
    attempt = ProofAttempt(
        goal=GOAL,
        script="apply intermediate.",
        new_lemmas=[Goal(name="intermediate", statement="True.")],
    )
    assert new_helper_protocol_error(attempt) == (
        "Invalid helper reference: newly proposed helper 'intermediate' "
        "must be referenced as '{intermediate}' in SCRIPT, "
        "not as bare 'intermediate'."
    )


# --- resolve_attempt_helpers ---


def test_resolve_attempt_helpers_binds_placeholders_and_renames() -> None:
    helper = Goal(name="helper", statement="True.")
    attempt = ProofAttempt(
        goal=GOAL,
        script="exact {helper}.",
        new_lemmas=[helper],
    )

    resolved = resolve_attempt_helpers(attempt, ["helper_1"])

    assert resolved.script == "exact helper_1."
    assert resolved.new_lemmas[0].name == "helper_1"
    assert resolved.new_lemmas[0].statement == "True."
    assert attempt.script == "exact {helper}."


def test_resolve_attempt_helpers_deduplicates_aliased_names() -> None:
    left = Goal(name="left", statement="left_statement.")
    right = Goal(name="right", statement="right_statement.")
    attempt = ProofAttempt(
        goal=GOAL,
        script="split. exact {left}. exact {right}.",
        new_lemmas=[left, right],
    )

    resolved = resolve_attempt_helpers(attempt, ["base", "base"])

    assert resolved.script == "split. exact base. exact base."
    assert [lemma.name for lemma in resolved.new_lemmas] == ["base"]
    assert [lemma.statement for lemma in resolved.new_lemmas] == ["left_statement."]


# --- script_with_child_names ---


def test_script_with_child_names_uses_child_goal_names() -> None:
    shared = LemmaNode(goal=Goal(name="base", statement="True."))
    record = AttemptRecord(
        attempt=ProofAttempt(
            goal=GOAL,
            script="exact {dup}.",
            new_lemmas=[Goal(name="dup", statement="True.")],
        ),
        rocq_error=CoqcResult(success=False, stderr="Error on helper."),
        lemmas=[shared],
    )

    assert script_with_child_names(record) == "exact base."
    assert record.attempt.script == "exact {dup}."


def test_script_with_child_names_falls_back_to_proposed_names_when_incomplete() -> None:
    renamed = LemmaNode(goal=Goal(name="base", statement="True."))
    record = AttemptRecord(
        attempt=ProofAttempt(
            goal=GOAL,
            script="split. exact {left}. exact {right}. apply {left}.",
            new_lemmas=[
                Goal(name="left", statement="True."),
                Goal(name="right", statement="False."),
            ],
        ),
        rocq_error=CoqcResult(success=False, stderr="Error on helper."),
        lemmas=[renamed],
    )

    assert script_with_child_names(record) == (
        "split. exact left. exact right. apply left."
    )
    assert record.attempt.script == (
        "split. exact {left}. exact {right}. apply {left}."
    )
