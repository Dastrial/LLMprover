"""Simple lemma selection: depth-weighted open nodes, delayed negative polarity."""

from __future__ import annotations

import random

from llmprover.domain import (
    AttemptRecord,
    LemmaNode,
    LemmaStatus,
    Polarity,
    Position,
)
from llmprover.strategy.lemma_selection_strategy import LemmaSelectionStrategy

# Prefer positive polarity: at least this many failed positive attempts per
# failed negative attempt before another negative is scheduled at a position.
# The high default reproduces the first miniF2F benchmark configuration and
# makes negative-polarity attempts practically unreachable within its budgets.
POSITIVE_TO_NEGATIVE_RATIO = 10000


class SimpleLemmaSelectionStrategy(LemmaSelectionStrategy):
    """Pick an open position by depth-weighted attempt count; break ties at random.

    Selection score is ``attempts / 2**depth``. A parent therefore needs about
    twice fewer attempts than its children (four times fewer than grandchildren,
    …) before it outranks them — so open children of a working decomposition
    are exhausted further before the parent is retried.

    Polarity stays positive until there are at least
    ``positive_to_negative_ratio`` failed positive attempts per failed negative
    attempt at that position (defaults to positive when never tried). The
    default value reproduces the positive-only behavior of the first miniF2F
    benchmark; callers may use a smaller value to enable polarity alternation.
    ``known_true`` keeps the root on ``Positive``.

    ``update`` maintains a strategy-local selection frontier for each polarity.
    Each new attempt replaces the previously selectable helper subtree; if the
    new attempt fails Rocq validation, no helper from that polarity remains
    selectable.

    Once a lemma closes on a leaf success (coqc ok, no children) as ``Proved``
    or ``Refuted``, it is closed for *both* polarities: the position leaves
    ``open`` and every frontier subtree under it is dropped. Closure then
    propagates to ancestors while they keep becoming ``Proved``; the climb
    stops at the first ``Open`` or ``Refuted`` ancestor.

    Invariant: ``open`` contains exactly the selectable positions, all with
    status ``Open`` — closed or forgotten nodes are removed from it eagerly.
    """

    def __init__(
        self,
        rng: random.Random | None = None,
        *,
        positive_to_negative_ratio: int = POSITIVE_TO_NEGATIVE_RATIO,
    ) -> None:
        if positive_to_negative_ratio < 1:
            raise ValueError("positive_to_negative_ratio must be at least 1")

        self.rng = rng if rng is not None else random.Random()
        self.positive_to_negative_ratio = positive_to_negative_ratio
        self.attempt_counts: dict[Position, int] = {(): 0}
        self.failed_attempts: dict[Position, dict[Polarity, int]] = {
            (): {Polarity.Positive: 0, Polarity.Negative: 0}
        }
        self.status: dict[Position, LemmaStatus] = {(): LemmaStatus.Open}
        self.open: set[Position] = {()}
        self.frontier_children: dict[Position, dict[Polarity, list[Position]]] = {
            (): {Polarity.Positive: [], Polarity.Negative: []}
        }
        self.frontier_success: dict[Position, dict[Polarity, bool]] = {
            (): {Polarity.Positive: False, Polarity.Negative: False}
        }

    def selection_score(self, position: Position) -> float:
        """Lower is better; deeper nodes are preferred for the same attempt count."""
        return self.attempt_counts[position] / (2 ** len(position))

    def select_polarity(self, position: Position, *, known_true: bool) -> Polarity:
        if known_true and position == ():
            return Polarity.Positive
        failed = self.failed_attempts.get(
            position, {Polarity.Positive: 0, Polarity.Negative: 0}
        )
        if failed[Polarity.Positive] >= self.positive_to_negative_ratio * (
            failed[Polarity.Negative] + 1
        ):
            return Polarity.Negative
        return Polarity.Positive

    def select_lemma(self, *, known_true: bool = False) -> tuple[Position, Polarity]:
        if not self.open:
            raise RuntimeError("No open lemma position left to select")

        min_score = min(self.selection_score(pos) for pos in self.open)
        candidates = [
            pos for pos in self.open if self.selection_score(pos) == min_score
        ]
        chosen = self.rng.choice(candidates)
        polarity = self.select_polarity(chosen, known_true=known_true)
        return chosen, polarity

    def update(
        self,
        attempt_record: AttemptRecord,
        position: Position,
    ) -> None:
        polarity = attempt_record.attempt.polarity

        self.attempt_counts[position] = self.attempt_counts.get(position, 0) + 1
        if not attempt_record.rocq_error.success:
            failed = self.failed_attempts.setdefault(
                position, {Polarity.Positive: 0, Polarity.Negative: 0}
            )
            failed[polarity] += 1
        self.frontier_success.setdefault(
            position, {Polarity.Positive: False, Polarity.Negative: False}
        )[polarity] = attempt_record.rocq_error.success
        self.frontier_children.setdefault(
            position, {Polarity.Positive: [], Polarity.Negative: []}
        )

        for child in self.frontier_children[position][polarity]:
            self.forget_subtree(child)

        # Failed coqc means a proposed decomposition is unusable: do not open
        # children. Only a successful attempt may extend the frontier.
        new_children: list[Position] = []
        if attempt_record.rocq_error.success:
            for index, lemma in enumerate(attempt_record.lemmas):
                child_pos = position + ((polarity, index),)
                new_children.append(child_pos)
                self.attempt_counts[child_pos] = 0
                self.failed_attempts[child_pos] = {
                    Polarity.Positive: 0,
                    Polarity.Negative: 0,
                }
                self.frontier_children[child_pos] = {
                    Polarity.Positive: [],
                    Polarity.Negative: [],
                }
                self.frontier_success[child_pos] = {
                    Polarity.Positive: False,
                    Polarity.Negative: False,
                }
                if lemma.status is LemmaStatus.Open:
                    self.status[child_pos] = LemmaStatus.Open
                    self.open.add(child_pos)
                else:
                    self.status[child_pos] = lemma.status
        self.frontier_children[position][polarity] = new_children

        # A node itself only closes immediately on a leaf success (coqc ok, no
        # children). With children, closure is decided later via ancestors,
        # unless every child is already decided (aliased proved helpers).
        if attempt_record.rocq_error.success and not new_children:
            status = (
                LemmaStatus.Proved
                if polarity is Polarity.Positive
                else LemmaStatus.Refuted
            )
            self.close_position(position, status)
            self.recompute_ancestors(position)
            return

        parent_status = (
            self.status_from_frontiers(position)
            if attempt_record.rocq_error.success
            else LemmaStatus.Open
        )
        if parent_status is LemmaStatus.Open:
            self.status[position] = LemmaStatus.Open
            self.open.add(position)
        else:
            self.close_position(position, parent_status)
            self.recompute_ancestors(position)

    def close_position(self, position: Position, status: LemmaStatus) -> None:
        """Mark *position* decided and drop every frontier subtree under it.

        A proved or refuted lemma is finished for both polarities: continuing
        on the opposite side cannot help, and pending opposite-polarity
        children must leave the open set.
        """
        self.status[position] = status
        self.open.discard(position)
        children_by_polarity = self.frontier_children.get(
            position, {Polarity.Positive: [], Polarity.Negative: []}
        )
        for children in children_by_polarity.values():
            for child in list(children):
                self.forget_subtree(child)
        self.frontier_children[position] = {
            Polarity.Positive: [],
            Polarity.Negative: [],
        }

    def forget_subtree(self, position: Position) -> None:
        """Drop *position* and every descendant still recorded under its frontiers."""
        self.open.discard(position)
        self.attempt_counts.pop(position, None)
        self.failed_attempts.pop(position, None)
        self.status.pop(position, None)
        self.frontier_success.pop(position, None)
        children_by_polarity = self.frontier_children.pop(position, {})
        for children in children_by_polarity.values():
            for child in children:
                self.forget_subtree(child)

    def recompute_ancestors(self, position: Position) -> None:
        """Propagate closure upward while parents keep becoming ``Proved``.

        Stop at the first ``Open`` or ``Refuted`` ancestor: neither case can
        make a higher ancestor newly succeed (``Refuted`` children do not
        satisfy a parent's proof obligations).
        """
        current = position
        while current:
            parent = current[:-1]
            parent_status = self.status_from_frontiers(parent)
            if parent_status is LemmaStatus.Open:
                self.status[parent] = LemmaStatus.Open
                self.open.add(parent)
                return
            self.close_position(parent, parent_status)
            if parent_status is LemmaStatus.Refuted:
                return
            current = parent

    def status_from_frontiers(self, position: Position) -> LemmaStatus:
        success = self.frontier_success.get(
            position, {Polarity.Positive: False, Polarity.Negative: False}
        )
        children = self.frontier_children.get(
            position, {Polarity.Positive: [], Polarity.Negative: []}
        )
        if success[Polarity.Positive] and self.all_proved(children[Polarity.Positive]):
            return LemmaStatus.Proved
        if success[Polarity.Negative] and self.all_proved(children[Polarity.Negative]):
            return LemmaStatus.Refuted
        return LemmaStatus.Open

    def all_proved(self, children: list[Position]) -> bool:
        return all(self.status.get(child) is LemmaStatus.Proved for child in children)

    def sync_with_tree(self, root: LemmaNode) -> None:
        """Close open positions whose lemma nodes are already decided."""
        for position in list(self.open):
            if position not in self.open:
                continue
            node = root.from_position(position)
            if node.status is LemmaStatus.Open:
                continue
            self.close_position(position, node.status)
            self.recompute_ancestors(position)
