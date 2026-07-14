"""Tests for llmprover.domain."""

from __future__ import annotations

import pytest

from llmprover.domain import (
    CoqcResult,
    Goal,
    ProofAttempt,
    ProofNode,
    ProofTree,
)


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


def test_proof_tree_accepts_matching_children() -> None:
    root_goal = Goal(statement="P", name="root")
    child_goal = Goal(statement="Q", name="child")
    attempt = ProofAttempt(goal=root_goal, script="admit.", new_lemmas=[child_goal])
    root = ProofNode(goal=root_goal, current_attempt=attempt)

    tree = ProofTree(root, children=[ProofNode(goal=child_goal)])

    assert tree.children[0].goal == child_goal


def test_proof_tree_rejects_mismatched_children() -> None:
    root_goal = Goal(statement="P", name="root")
    expected = Goal(statement="Q", name="expected")
    other = Goal(statement="R", name="other")
    attempt = ProofAttempt(goal=root_goal, script="admit.", new_lemmas=[expected])
    root = ProofNode(goal=root_goal, current_attempt=attempt)

    with pytest.raises(ValueError, match="Child goals must match"):
        ProofTree(root, children=[ProofNode(goal=other)])


def test_proof_tree_builds_children_from_new_lemmas() -> None:
    root_goal = Goal(statement="P", name="root")
    child_a = Goal(statement="Q", name="child_a")
    child_b = Goal(statement="R", name="child_b")
    attempt = ProofAttempt(
        goal=root_goal,
        script="admit.",
        new_lemmas=[child_a, child_b],
    )
    root = ProofNode(goal=root_goal, current_attempt=attempt)

    tree = ProofTree(root)

    assert [child.goal for child in tree.children] == [child_a, child_b]


def test_proof_tree_without_children_starts_empty() -> None:
    root = ProofNode(goal=Goal(statement="P", name="root"))
    tree = ProofTree(root)
    assert tree.children == []
