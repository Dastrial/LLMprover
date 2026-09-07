"""Tests for llmprover.equivalence_merger."""

from __future__ import annotations

from unittest.mock import MagicMock

from llmprover.domain import (
    AttemptRecord,
    CoqcResult,
    Goal,
    LemmaNode,
    Polarity,
    Position,
    ProofAttempt,
)
from llmprover.equivalence_merger import EquivalenceMerger

GOAL = Goal(name="plus_n0", statement="forall n : nat, n + 0 = n.")
CHILD_A = Goal(name="base", statement="0 + 0 = 0.")
CHILD_B = Goal(name="step", statement="forall n, n + 0 = n -> S n + 0 = S n.")
ROOT: Position = ()
POS_CHILD0: Position = ((Polarity.Positive, 0),)
POS_CHILD1: Position = ((Polarity.Positive, 1),)


def _canonical(statement: str) -> str:
    return statement.strip().rstrip(".").strip()


def make_merger(checker: MagicMock | None = None) -> EquivalenceMerger:
    if checker is None:
        checker = MagicMock()
        checker.statements_equivalent.side_effect = (
            lambda left, right, environment=None: _canonical(left) == _canonical(right)
        )
    return EquivalenceMerger(checker)


# --- EquivalenceMerger.analyze ---


def test_analyze_uses_rocq_convertibility_when_strings_differ() -> None:
    checker = MagicMock()
    checker.statements_equivalent.return_value = True
    root = LemmaNode(goal=GOAL)
    attempt = ProofAttempt(
        goal=GOAL,
        script="apply {norm}.",
        new_lemmas=[Goal(name="norm", statement="2 = 2.")],
    )

    cycle, aliases = make_merger(checker).analyze(attempt, root, ROOT)

    assert cycle is not None
    checker.statements_equivalent.assert_called()
    assert aliases == {}


def test_analyze_detects_cycle_with_ancestor() -> None:
    root = LemmaNode(goal=GOAL)
    attempt = ProofAttempt(
        goal=GOAL,
        script="apply {again}.",
        new_lemmas=[Goal(name="again", statement=GOAL.statement)],
    )

    cycle, aliases = make_merger().analyze(attempt, root, ROOT)

    assert cycle is not None
    helper, ancestor = cycle
    assert helper.name == "again"
    assert ancestor is root
    assert aliases == {}


def test_analyze_detects_cycle_with_grandparent() -> None:
    child_a = LemmaNode(goal=CHILD_A)
    root = LemmaNode(goal=GOAL)
    root.append(
        AttemptRecord(
            attempt=ProofAttempt(
                goal=GOAL,
                script="apply {base}.",
                new_lemmas=[CHILD_A],
            ),
            rocq_error=CoqcResult(success=True),
            lemmas=[child_a],
        )
    )
    attempt = ProofAttempt(
        goal=child_a.goal,
        script="apply {again}.",
        new_lemmas=[Goal(name="again", statement=GOAL.statement)],
    )

    cycle, aliases = make_merger().analyze(attempt, root, POS_CHILD0)

    assert cycle is not None
    helper, ancestor = cycle
    assert helper.name == "again"
    assert ancestor is root
    assert aliases == {}


def test_analyze_detects_cycle_through_aliased_node() -> None:
    """After B aliases to A, A must not introduce a helper equivalent to B."""
    child_a = LemmaNode(goal=CHILD_A)
    child_b = LemmaNode(goal=CHILD_B)
    root = LemmaNode(goal=GOAL)
    root.append(
        AttemptRecord(
            attempt=ProofAttempt(
                goal=GOAL,
                script="split.",
                new_lemmas=[CHILD_A, CHILD_B],
            ),
            rocq_error=CoqcResult(success=True),
            lemmas=[child_a, child_b],
        )
    )
    child_b.append(
        AttemptRecord(
            attempt=ProofAttempt(
                goal=CHILD_B,
                script="apply {dup}.",
                new_lemmas=[Goal(name="dup", statement=CHILD_A.statement)],
            ),
            rocq_error=CoqcResult(success=True),
            lemmas=[child_a],
        )
    )

    attempt = ProofAttempt(
        goal=child_a.goal,
        script="apply {step}.",
        new_lemmas=[Goal(name="again", statement=CHILD_B.statement)],
    )

    cycle, aliases = make_merger().analyze(attempt, root, POS_CHILD0)

    assert cycle is not None
    helper, ancestor = cycle
    assert helper.name == "again"
    assert ancestor is child_b
    assert aliases == {}


def test_analyze_aliases_incomparable_cousin() -> None:
    child_a = LemmaNode(goal=CHILD_A)
    child_b = LemmaNode(goal=CHILD_B)
    root = LemmaNode(goal=GOAL)
    root.append(
        AttemptRecord(
            attempt=ProofAttempt(
                goal=GOAL,
                script="split.",
                new_lemmas=[CHILD_A, CHILD_B],
            ),
            rocq_error=CoqcResult(success=True),
            lemmas=[child_a, child_b],
        )
    )
    attempt = ProofAttempt(
        goal=child_b.goal,
        script="apply {dup}.",
        new_lemmas=[Goal(name="dup", statement=CHILD_A.statement)],
    )

    cycle, aliases = make_merger().analyze(attempt, root, POS_CHILD1)

    assert cycle is None
    assert aliases == {0: child_a}
