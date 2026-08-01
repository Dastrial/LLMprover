"""Tests for llmprover.simple_lemma_selection_strategy."""

from __future__ import annotations

import random
from unittest.mock import MagicMock

import pytest

from llmprover.domain import (
    AttemptRecord,
    CoqcResult,
    Goal,
    LemmaNode,
    LemmaStatus,
    Polarity,
    Position,
    ProofAttempt,
)
from llmprover.simple_lemma_selection_strategy import SimpleLemmaSelectionStrategy

GOAL = Goal(name="plus_n0", statement="forall n : nat, n + 0 = n.")
CHILD_A = Goal(name="base", statement="0 + 0 = 0.")
CHILD_B = Goal(name="step", statement="forall n, True.")
ROOT: Position = ()
CHILD0: Position = ((Polarity.Positive, 0),)
CHILD1: Position = ((Polarity.Positive, 1),)
NEG_CHILD0: Position = ((Polarity.Negative, 0),)


def record(
    *,
    success: bool,
    polarity: Polarity = Polarity.Positive,
    lemmas: list[LemmaNode] | None = None,
    goal: Goal = GOAL,
) -> AttemptRecord:
    return AttemptRecord(
        attempt=ProofAttempt(
            goal=goal,
            polarity=polarity,
            script="script.",
            new_lemmas=[n.goal for n in (lemmas or [])],
        ),
        rocq_error=CoqcResult(success=success, stderr="" if success else "err"),
        lemmas=lemmas or [],
    )


def test_select_lemma_starts_at_root_positive() -> None:
    strategy = SimpleLemmaSelectionStrategy(rng=random.Random(0))
    position, polarity = strategy.select_lemma()
    assert position == ROOT
    assert polarity is Polarity.Positive


def test_update_failed_attempt_increments_count_and_alternates_polarity() -> None:
    strategy = SimpleLemmaSelectionStrategy(rng=random.Random(0))
    strategy.update(record(success=False), ROOT)

    position, polarity = strategy.select_lemma()
    assert position == ROOT
    assert polarity is Polarity.Negative
    assert strategy.attempt_counts[ROOT] == 1


def test_select_lemma_prefers_least_attempted_open_node() -> None:
    strategy = SimpleLemmaSelectionStrategy(rng=random.Random(0))
    children = [LemmaNode(goal=CHILD_A), LemmaNode(goal=CHILD_B)]
    strategy.update(record(success=True, lemmas=children), ROOT)

    # Root has 1 attempt; children have 0 → pick a child.
    position, polarity = strategy.select_lemma()
    assert position in (CHILD0, CHILD1)
    assert polarity is Polarity.Positive


def test_select_lemma_breaks_ties_at_random() -> None:
    children = [LemmaNode(goal=CHILD_A), LemmaNode(goal=CHILD_B)]
    rng = MagicMock()
    rng.choice.side_effect = [CHILD0, CHILD1]
    strategy = SimpleLemmaSelectionStrategy(rng=rng)
    strategy.update(record(success=True, lemmas=children), ROOT)

    assert strategy.select_lemma()[0] == CHILD0
    assert strategy.select_lemma()[0] == CHILD1

    assert rng.choice.call_count == 2
    for call in rng.choice.call_args_list:
        candidates = call.args[0]
        assert set(candidates) == {CHILD0, CHILD1}
        assert len(candidates) == 2


def test_failed_attempt_with_lemmas_does_not_open_children() -> None:
    strategy = SimpleLemmaSelectionStrategy(rng=random.Random(0))
    strategy.update(
        record(
            success=False,
            lemmas=[LemmaNode(goal=CHILD_A), LemmaNode(goal=CHILD_B)],
        ),
        ROOT,
    )

    assert strategy.open == {ROOT}
    assert strategy.frontier_children[ROOT][Polarity.Positive] == []
    assert CHILD0 not in strategy.attempt_counts
    assert CHILD1 not in strategy.attempt_counts


def test_update_forgets_children_replaced_by_new_attempt_same_polarity() -> None:
    strategy = SimpleLemmaSelectionStrategy(rng=random.Random(0))
    first_children = [LemmaNode(goal=CHILD_A), LemmaNode(goal=CHILD_B)]
    strategy.update(record(success=True, lemmas=first_children), ROOT)

    replacement = [LemmaNode(goal=Goal(name="only", statement="True."))]
    strategy.update(record(success=True, lemmas=replacement), ROOT)

    # Old child positions must be gone; only the new child (and root) remain open.
    assert CHILD0 in strategy.open
    assert CHILD1 not in strategy.open
    assert strategy.attempt_counts.get(CHILD1) is None

    # Least-attempted is the new child (0) vs root (2).
    position, _ = strategy.select_lemma()
    assert position == CHILD0


def test_update_keeps_other_polarity_frontier() -> None:
    strategy = SimpleLemmaSelectionStrategy(rng=random.Random(0))
    pos_children = [LemmaNode(goal=CHILD_A)]
    strategy.update(record(success=True, lemmas=pos_children), ROOT)

    neg_children = [LemmaNode(goal=CHILD_B)]
    strategy.update(
        record(success=True, polarity=Polarity.Negative, lemmas=neg_children),
        ROOT,
    )

    assert CHILD0 in strategy.open
    assert NEG_CHILD0 in strategy.open


def test_successful_leaf_closes_position() -> None:
    strategy = SimpleLemmaSelectionStrategy(rng=random.Random(0))
    strategy.update(record(success=True, lemmas=[]), ROOT)

    assert ROOT not in strategy.open
    with pytest.raises(RuntimeError, match="No open lemma position"):
        strategy.select_lemma()


def test_proving_all_children_closes_parent() -> None:
    strategy = SimpleLemmaSelectionStrategy(rng=random.Random(0))
    children = [LemmaNode(goal=CHILD_A), LemmaNode(goal=CHILD_B)]
    strategy.update(record(success=True, lemmas=children), ROOT)

    strategy.update(
        record(success=True, lemmas=[], goal=CHILD_A),
        CHILD0,
    )
    assert ROOT in strategy.open  # sibling still open → parent stays open

    strategy.update(
        record(success=True, lemmas=[], goal=CHILD_B),
        CHILD1,
    )
    assert ROOT not in strategy.open
    assert strategy.status[ROOT] is LemmaStatus.Proved


def test_failed_child_refutation_does_not_close_parent() -> None:
    strategy = SimpleLemmaSelectionStrategy(rng=random.Random(0))
    children = [LemmaNode(goal=CHILD_A)]
    strategy.update(record(success=True, lemmas=children), ROOT)
    strategy.update(
        record(
            success=True,
            polarity=Polarity.Negative,
            lemmas=[],
            goal=CHILD_A,
        ),
        CHILD0,
    )

    assert CHILD0 not in strategy.open
    assert ROOT in strategy.open


def test_proved_lemma_drops_opposite_polarity_children() -> None:
    strategy = SimpleLemmaSelectionStrategy(rng=random.Random(0))
    strategy.update(
        record(success=True, polarity=Polarity.Negative, lemmas=[LemmaNode(goal=CHILD_B)]),
        ROOT,
    )
    assert NEG_CHILD0 in strategy.open

    strategy.update(record(success=True, lemmas=[]), ROOT)

    assert strategy.status[ROOT] is LemmaStatus.Proved
    assert ROOT not in strategy.open
    assert NEG_CHILD0 not in strategy.open
    with pytest.raises(RuntimeError, match="No open lemma position"):
        strategy.select_lemma()


def test_refuted_lemma_drops_opposite_polarity_children() -> None:
    strategy = SimpleLemmaSelectionStrategy(rng=random.Random(0))
    strategy.update(
        record(success=True, lemmas=[LemmaNode(goal=CHILD_A)]),
        ROOT,
    )
    assert CHILD0 in strategy.open

    strategy.update(
        record(success=True, polarity=Polarity.Negative, lemmas=[]),
        ROOT,
    )

    assert strategy.status[ROOT] is LemmaStatus.Refuted
    assert ROOT not in strategy.open
    assert CHILD0 not in strategy.open
    with pytest.raises(RuntimeError, match="No open lemma position"):
        strategy.select_lemma()


def test_proving_children_closes_parent_and_drops_other_polarity_work() -> None:
    strategy = SimpleLemmaSelectionStrategy(rng=random.Random(0))
    strategy.update(
        record(success=True, lemmas=[LemmaNode(goal=CHILD_A), LemmaNode(goal=CHILD_B)]),
        ROOT,
    )
    strategy.update(
        record(success=True, polarity=Polarity.Negative, lemmas=[LemmaNode(goal=CHILD_B)]),
        ROOT,
    )
    assert NEG_CHILD0 in strategy.open

    strategy.update(record(success=True, lemmas=[], goal=CHILD_A), CHILD0)
    strategy.update(record(success=True, lemmas=[], goal=CHILD_B), CHILD1)

    assert strategy.status[ROOT] is LemmaStatus.Proved
    assert ROOT not in strategy.open
    assert NEG_CHILD0 not in strategy.open


def test_recompute_stops_when_parent_stays_open() -> None:
    """Proving one child must not keep walking once the parent stays Open."""
    strategy = SimpleLemmaSelectionStrategy(rng=random.Random(0))
    strategy.update(
        record(success=True, lemmas=[LemmaNode(goal=CHILD_A), LemmaNode(goal=CHILD_B)]),
        ROOT,
    )
    strategy.update(
        record(success=True, lemmas=[LemmaNode(goal=Goal(name="g", statement="True."))]),
        CHILD0,
    )
    grandchild = CHILD0 + ((Polarity.Positive, 0),)

    strategy.update(
        record(success=True, lemmas=[], goal=Goal(name="g", statement="True.")),
        grandchild,
    )

    assert strategy.status[CHILD0] is LemmaStatus.Proved
    assert strategy.status[ROOT] is LemmaStatus.Open
    assert ROOT in strategy.open
    assert CHILD1 in strategy.open


def test_completing_negative_children_closes_parent_as_refuted_and_stops() -> None:
    strategy = SimpleLemmaSelectionStrategy(rng=random.Random(0))
    strategy.update(
        record(success=True, polarity=Polarity.Negative, lemmas=[LemmaNode(goal=CHILD_A)]),
        ROOT,
    )
    strategy.update(record(success=True, lemmas=[], goal=CHILD_A), NEG_CHILD0)

    assert strategy.status[ROOT] is LemmaStatus.Refuted
    assert ROOT not in strategy.open
    with pytest.raises(RuntimeError, match="No open lemma position"):
        strategy.select_lemma()
