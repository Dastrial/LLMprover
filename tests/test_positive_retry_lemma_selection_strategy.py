"""Tests for llmprover.strategy.positive_retry_lemma_selection_strategy."""

from __future__ import annotations

import random

from llmprover.domain import (
    AttemptRecord,
    CoqcResult,
    Goal,
    LemmaNode,
    Polarity,
    Position,
    ProofAttempt,
)
from llmprover.strategy.positive_retry_lemma_selection_strategy import (
    PARENT_RETRY_SCORE,
    PositiveRetryLemmaSelectionStrategy,
)

GOAL = Goal(name="plus_n0", statement="forall n : nat, n + 0 = n.")
CHILD_A = Goal(name="base", statement="0 + 0 = 0.")
CHILD_B = Goal(name="step", statement="forall n, True.")
ROOT: Position = ()
CHILD0: Position = ((Polarity.Positive, 0),)


def record(
    *,
    success: bool,
    lemmas: list[LemmaNode] | None = None,
    goal: Goal = GOAL,
) -> AttemptRecord:
    return AttemptRecord(
        attempt=ProofAttempt(
            goal=goal,
            script="script.",
            new_lemmas=[n.goal for n in (lemmas or [])],
        ),
        rocq_error=CoqcResult(success=success, stderr="" if success else "err"),
        lemmas=lemmas or [],
    )


def synced_strategy(
    root: LemmaNode,
    *,
    parent_retry_score: int = PARENT_RETRY_SCORE,
) -> PositiveRetryLemmaSelectionStrategy:
    strategy = PositiveRetryLemmaSelectionStrategy(
        rng=random.Random(0),
        parent_retry_score=parent_retry_score,
    )
    strategy.sync_with_tree(root)
    return strategy


def apply_attempt(
    strategy: PositiveRetryLemmaSelectionStrategy,
    root: LemmaNode,
    attempt: AttemptRecord,
    position: Position = ROOT,
) -> None:
    root.from_position(position).append(attempt)
    strategy.update(attempt, position)
    strategy.sync_with_tree(root)


def test_failed_parent_retry_increments_parent_retries_without_resetting_helpers() -> (
    None
):
    child = LemmaNode(goal=CHILD_A)
    root = LemmaNode(goal=GOAL)
    strategy = synced_strategy(root)

    apply_attempt(strategy, root, record(success=True, lemmas=[child]), ROOT)
    strategy._state_for(child).attempts_since_reset = 3

    apply_attempt(
        strategy,
        root,
        record(success=False, lemmas=[LemmaNode(goal=CHILD_B)]),
        ROOT,
    )

    assert strategy._state_for(root).parent_retries_since_decomposition == 1
    assert strategy._state_for(child).attempts_since_reset == 3
    assert strategy.selection_score(root) == PARENT_RETRY_SCORE + 1


def test_same_successful_decomposition_increments_parent_retries_without_resetting_helpers() -> (
    None
):
    child = LemmaNode(goal=CHILD_A)
    root = LemmaNode(goal=GOAL)
    strategy = synced_strategy(root)

    apply_attempt(strategy, root, record(success=True, lemmas=[child]), ROOT)
    strategy._state_for(child).attempts_since_reset = 2
    strategy._state_for(root).parent_retries_since_decomposition = 1

    apply_attempt(strategy, root, record(success=True, lemmas=[child]), ROOT)

    assert strategy._state_for(root).parent_retries_since_decomposition == 2
    assert strategy._state_for(child).attempts_since_reset == 2


def test_different_successful_decomposition_resets_parent_retries_and_helpers() -> None:
    first_child = LemmaNode(goal=CHILD_A)
    second_child = LemmaNode(goal=CHILD_B)
    root = LemmaNode(goal=GOAL)
    strategy = synced_strategy(root)

    apply_attempt(strategy, root, record(success=True, lemmas=[first_child]), ROOT)
    strategy._state_for(first_child).attempts_since_reset = 4
    strategy._state_for(root).parent_retries_since_decomposition = 2

    apply_attempt(strategy, root, record(success=True, lemmas=[second_child]), ROOT)

    assert strategy._state_for(root).parent_retries_since_decomposition == 0
    assert strategy._state_for(first_child).attempts_since_reset == 4
    assert strategy._state_for(second_child).attempts_since_reset == 0


def test_reordered_merged_helpers_count_as_same_decomposition() -> None:
    child_a = LemmaNode(goal=CHILD_A)
    child_b = LemmaNode(goal=CHILD_B)
    root = LemmaNode(goal=GOAL)
    strategy = synced_strategy(root)

    apply_attempt(strategy, root, record(success=True, lemmas=[child_a, child_b]), ROOT)
    strategy._state_for(child_a).attempts_since_reset = 5
    strategy._state_for(child_b).attempts_since_reset = 6
    strategy._state_for(root).parent_retries_since_decomposition = 1

    apply_attempt(
        strategy,
        root,
        record(success=True, lemmas=[child_b, child_a]),
        ROOT,
    )

    assert strategy._state_for(root).parent_retries_since_decomposition == 2
    assert strategy._state_for(child_a).attempts_since_reset == 5
    assert strategy._state_for(child_b).attempts_since_reset == 6


def test_active_positions_follow_successful_frontier_after_failed_parent_retry() -> (
    None
):
    child = LemmaNode(goal=CHILD_A)
    root = LemmaNode(goal=GOAL)
    strategy = synced_strategy(root)

    apply_attempt(strategy, root, record(success=True, lemmas=[child]), ROOT)
    apply_attempt(
        strategy,
        root,
        record(success=False, lemmas=[LemmaNode(goal=CHILD_B)]),
        ROOT,
    )

    assert CHILD0 in strategy.position_to_node_id
    assert strategy.position_to_node_id[CHILD0] == id(child)
