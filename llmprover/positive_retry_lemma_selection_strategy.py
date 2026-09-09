"""Positive-only lemma selection with resettable node-level retry state.

This strategy is designed for proof-search trees that may become DAGs after
helper-equivalence merging: several active positions can refer to the same
LemmaNode. Scheduling state therefore belongs to the node identity, not to an
individual Position.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from llmprover.domain import (
    AttemptRecord,
    LemmaNode,
    LemmaStatus,
    Polarity,
    Position,
)
from llmprover.lemma_selection_strategy import LemmaSelectionStrategy

# An open lemma with active helper obligations competes with its helpers using
# this fixed score. A leaf instead uses its real number of attempts since reset.
PARENT_RETRY_SCORE = 5


@dataclass
class NodeSelectionState:
    """Ephemeral scheduling state for one LemmaNode identity."""

    attempts_since_reset: int = 0
    parent_retries_since_decomposition: int = 0


class PositiveRetryLemmaSelectionStrategy(LemmaSelectionStrategy):
    """Positive-only lemma selection with resettable node-level retry state.

    LemmaNode is the persistent source of truth for proof history and proof
    structure. Position is only an ephemeral path used by the orchestrator to
    reach a node.

    Equivalence merging may make several active positions refer to the same
    LemmaNode. Scheduling state is therefore keyed by id(node), so aliased
    occurrences always share one retry counter.

    Selection is positive-only.

    Scoring, lower is better:

    - if an open lemma currently has at least one open helper in its latest
      successful positive decomposition, its score is PARENT_RETRY_SCORE plus
      the number of parent retries since that decomposition became active;
    - otherwise its score is the number of attempts made on that lemma since
      its latest reset.

    Therefore, with PARENT_RETRY_SCORE = 5, a fresh helper normally gets five
    attempts before its parent becomes competitive again.

    A first successful decomposition, or one with a different set of helper
    node identities, resets attempts_since_reset for its immediate helpers,
    including reused nodes. Repeating the same decomposition preserves their
    counters. Resets leave proof histories, parent-retry counters, and
    descendants' counters unchanged.

    Active positions are rebuilt from the real LemmaNode frontier instead of
    being maintained as an independent mutable mirror of the proof tree.
    """

    def __init__(
        self,
        rng: random.Random | None = None,
        *,
        parent_retry_score: int = PARENT_RETRY_SCORE,
    ) -> None:
        if parent_retry_score < 0:
            raise ValueError("parent_retry_score must be non-negative")

        self.rng = rng if rng is not None else random.Random()
        self.parent_retry_score = parent_retry_score

        # Scheduling state is attached to LemmaNode identity.
        #
        # LemmaNode is a mutable dataclass and is not hashable, so id(node) is
        # used rather than the node itself.
        self.node_state: dict[int, NodeSelectionState] = {}

        # Ephemeral view of the current positive frontier.
        #
        # A node may occur at several active positions after equivalence merging,
        # but it appears only once in active_nodes and therefore only once in
        # lemma-selection tie breaking.
        self.active_nodes: dict[int, LemmaNode] = {}
        self.active_positions: dict[int, list[Position]] = {}
        self.position_to_node_id: dict[Position, int] = {}

        self._root: LemmaNode | None = None

        # The current orchestrator does not call sync_with_tree(root) before the
        # very first select_lemma(). Until the first sync, () is necessarily the
        # root position, so root attempts can temporarily be counted here.
        self._pending_root_attempts = 0

    def _state_for(self, node: LemmaNode) -> NodeSelectionState:
        """Return the scheduling state associated with node identity."""
        return self.node_state.setdefault(
            id(node),
            NodeSelectionState(),
        )

    def reset_node(self, node: LemmaNode) -> None:
        """Reset this node's attempts_since_reset counter only."""
        self._state_for(node).attempts_since_reset = 0

    @staticmethod
    def has_active_open_children(node: LemmaNode) -> bool:
        """Whether node's current positive decomposition has an open helper."""
        frontier = node.frontier(Polarity.Positive)

        if frontier is None:
            return False

        if not frontier.rocq_error.success:
            return False

        return any(child.status is LemmaStatus.Open for child in frontier.lemmas)

    def selection_score(self, node: LemmaNode) -> int:
        """Return node's current scheduling score; lower is better."""
        state = self._state_for(node)

        if self.has_active_open_children(node):
            return self.parent_retry_score + state.parent_retries_since_decomposition

        return state.attempts_since_reset

    def select_lemma(
        self,
        *,
        known_true: bool = False,
    ) -> tuple[Position, Polarity]:
        """Select one open lemma and always attack its positive polarity."""
        del known_true

        # Bootstrap compatibility with the current orchestrator.
        #
        # Before the first sync, the only lemma that can exist as a selectable
        # goal is the root.
        if self._root is None:
            return (), Polarity.Positive

        if not self.active_nodes:
            raise RuntimeError("No open lemma position left to select")

        min_score = min(
            self.selection_score(node) for node in self.active_nodes.values()
        )

        candidate_ids = [
            node_id
            for node_id, node in self.active_nodes.items()
            if self.selection_score(node) == min_score
        ]

        # Choose between unique LemmaNode identities first.
        #
        # An aliased node must not get extra probability merely because it
        # appears at several positions in the decomposition DAG.
        chosen_id = self.rng.choice(candidate_ids)

        # Any active occurrence reaches the same LemmaNode. Pick one only after
        # the node itself has been selected.
        chosen_position = self.rng.choice(self.active_positions[chosen_id])

        return chosen_position, Polarity.Positive

    def update(
        self,
        attempt_record: AttemptRecord,
        position: Position,
    ) -> None:
        """Update scheduling state after one checked positive attempt.

        The orchestrator appends attempt_record to the attacked LemmaNode before
        calling this method.
        """
        if attempt_record.attempt.polarity is not Polarity.Positive:
            raise ValueError(
                "PositiveRetryLemmaSelectionStrategy only supports positive attempts"
            )

        if self._root is None:
            if position != ():
                raise RuntimeError(
                    "Strategy has not yet been synced with the root; "
                    "only position () can be updated during bootstrap"
                )

            self._pending_root_attempts += 1

        else:
            node_id = self.position_to_node_id.get(position)

            if node_id is None:
                raise RuntimeError(
                    f"Unknown or stale lemma position in update: {position!r}"
                )

            state = self.node_state.setdefault(
                node_id,
                NodeSelectionState(),
            )
            state.attempts_since_reset += 1

            node = self.active_nodes[node_id]
            previous_success = self._previous_successful_positive(node)

            if previous_success is not None:
                if not attempt_record.rocq_error.success:
                    state.parent_retries_since_decomposition += 1
                elif self._same_decomposition(
                    attempt_record,
                    previous_success,
                ):
                    state.parent_retries_since_decomposition += 1
                else:
                    state.parent_retries_since_decomposition = 0
                    for helper in attempt_record.lemmas:
                        self.reset_node(helper)
            elif attempt_record.rocq_error.success:
                for helper in attempt_record.lemmas:
                    self.reset_node(helper)

    def sync_with_tree(self, root: LemmaNode) -> None:
        """Synchronize the strategy with the current LemmaNode DAG."""
        if self._root is None:
            self._root = root

            root_state = self._state_for(root)
            root_state.attempts_since_reset += self._pending_root_attempts
            self._pending_root_attempts = 0

        elif self._root is not root:
            # Normally one strategy instance is used for one prove() call.
            # If it is explicitly rebound to a different proof tree, scheduling
            # state from the previous proof must not leak into the new one.
            self._root = root
            self.node_state.clear()
            self.active_nodes.clear()
            self.active_positions.clear()
            self.position_to_node_id.clear()
            self._pending_root_attempts = 0

            self._state_for(root)

        self._rebuild_active_frontier()

    def _rebuild_active_frontier(self) -> None:
        """Reconstruct all selectable positive nodes from the real proof DAG."""
        root = self._root

        if root is None:
            return

        active_nodes: dict[int, LemmaNode] = {}
        active_positions: dict[int, list[Position]] = {}
        position_to_node_id: dict[Position, int] = {}

        # A shared LemmaNode only needs its descendants expanded once.
        #
        # We still record every directly encountered active occurrence of that
        # node, but descendants need only one representative active path because
        # their LemmaNode state is shared as well.
        expanded: set[int] = set()

        def visit(
            node: LemmaNode,
            position: Position,
            ancestors: frozenset[int],
        ) -> None:
            if node.status is not LemmaStatus.Open:
                return

            node_id = id(node)

            if node_id in ancestors:
                raise RuntimeError("Cycle detected while rebuilding lemma frontier")

            active_nodes[node_id] = node
            active_positions.setdefault(node_id, []).append(position)
            position_to_node_id[position] = node_id

            # Ensure scheduling state exists even for a node discovered through
            # an old alias.
            self._state_for(node)

            if node_id in expanded:
                return

            expanded.add(node_id)

            frontier = node.frontier(Polarity.Positive)

            if frontier is None or not frontier.rocq_error.success:
                return

            child_ancestors = ancestors | {node_id}

            for index, child in enumerate(frontier.lemmas):
                if child.status is not LemmaStatus.Open:
                    continue

                child_position = position + ((Polarity.Positive, index),)

                visit(
                    child,
                    child_position,
                    child_ancestors,
                )

        visit(
            root,
            (),
            frozenset(),
        )

        self.active_nodes = active_nodes
        self.active_positions = active_positions
        self.position_to_node_id = position_to_node_id

    @staticmethod
    def _previous_successful_positive(node: LemmaNode) -> AttemptRecord | None:
        records = node.positive
        if len(records) < 2:
            return None
        for record in reversed(records[:-1]):
            if record.rocq_error.success:
                return record
        return None

    @staticmethod
    def _same_decomposition(
        left: AttemptRecord,
        right: AttemptRecord,
    ) -> bool:
        return {id(helper) for helper in left.lemmas} == {
            id(helper) for helper in right.lemmas
        }
