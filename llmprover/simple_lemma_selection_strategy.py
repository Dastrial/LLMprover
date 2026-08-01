"""Simple lemma selection: least-attempted open nodes, alternate polarity."""

from __future__ import annotations

import random

from llmprover.domain import (
    AttemptRecord,
    LemmaStatus,
    Polarity,
    Position,
)
from llmprover.lemma_selection_strategy import LemmaSelectionStrategy


class SimpleLemmaSelectionStrategy(LemmaSelectionStrategy):
    """Pick the least-attempted open position; break ties at random.

    Polarity at a position alternates from the last polarity attempted there
    (defaults to ``Positive`` when the node has never been tried).

    ``update`` mirrors the frontier of ``LemmaNode``: only children of the
    latest ``AttemptRecord`` per polarity stay reachable; older sibling
    subtrees under that polarity are forgotten.

    Once a lemma closes on a leaf success (coqc ok, no children) as ``Proved``
    or ``Refuted``, it is closed for *both* polarities: the position leaves
    ``open`` and every frontier subtree under it is dropped. Closure then
    propagates to ancestors while they keep becoming ``Proved``; the climb
    stops at the first ``Open`` or ``Refuted`` ancestor.

    Invariant: ``open`` contains exactly the selectable positions, all with
    status ``Open`` — closed or forgotten nodes are removed from it eagerly.
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng if rng is not None else random.Random()
        self.attempt_counts: dict[Position, int] = {(): 0}
        self.last_polarity: dict[Position, Polarity | None] = {(): None}
        self.status: dict[Position, LemmaStatus] = {(): LemmaStatus.Open}
        self.open: set[Position] = {()}
        self.frontier_children: dict[Position, dict[Polarity, list[Position]]] = {
            (): {Polarity.Positive: [], Polarity.Negative: []}
        }
        self.frontier_success: dict[Position, dict[Polarity, bool]] = {
            (): {Polarity.Positive: False, Polarity.Negative: False}
        }

    def select_lemma(self) -> tuple[Position, Polarity]:
        if not self.open:
            raise RuntimeError("No open lemma position left to select")

        min_attempts = min(self.attempt_counts[pos] for pos in self.open)
        candidates = [
            pos for pos in self.open if self.attempt_counts[pos] == min_attempts
        ]
        chosen = self.rng.choice(candidates)
        last = self.last_polarity.get(chosen)
        polarity = Polarity.Negative if last is Polarity.Positive else Polarity.Positive
        return chosen, polarity

    def update(
        self,
        attempt_record: AttemptRecord,
        position: Position,
    ) -> None:
        polarity = attempt_record.attempt.polarity

        self.attempt_counts[position] = self.attempt_counts.get(position, 0) + 1
        self.last_polarity[position] = polarity
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
            for index, _ in enumerate(attempt_record.lemmas):
                child_pos = position + ((polarity, index),)
                new_children.append(child_pos)
                self.attempt_counts[child_pos] = 0
                self.last_polarity[child_pos] = None
                self.status[child_pos] = LemmaStatus.Open
                self.open.add(child_pos)
                self.frontier_children[child_pos] = {
                    Polarity.Positive: [],
                    Polarity.Negative: [],
                }
                self.frontier_success[child_pos] = {
                    Polarity.Positive: False,
                    Polarity.Negative: False,
                }
        self.frontier_children[position][polarity] = new_children

        # A node itself only closes immediately on a leaf success (coqc ok, no
        # children). With children, closure is decided later via ancestors.
        if attempt_record.rocq_error.success and not new_children:
            status = (
                LemmaStatus.Proved
                if polarity is Polarity.Positive
                else LemmaStatus.Refuted
            )
            self.close_position(position, status)
            self.recompute_ancestors(position)
        else:
            self.status[position] = LemmaStatus.Open
            self.open.add(position)

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
        self.last_polarity.pop(position, None)
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
