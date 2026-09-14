"""Lemma selection strategy abstract class."""

from __future__ import annotations

from abc import ABC, abstractmethod

from llmprover.domain import AttemptRecord, LemmaNode, Polarity, Position


class LemmaSelectionStrategy(ABC):
    """Abstract class for lemma selection strategies."""

    @abstractmethod
    def update(
        self,
        attempt_record: AttemptRecord,
        position: Position,
    ) -> None:
        """Update the strategy with a checked attempt at *position*."""

    @abstractmethod
    def select_lemma(self, *, known_true: bool = False) -> tuple[Position, Polarity]:
        """Select the position and polarity of the next lemma to attack.

        When *known_true* is set and the chosen position is the root, polarity
        stays ``Positive`` (benchmark / trusted theorem: search for a proof of
        ``P``, do not try ``~P`` on the main goal).
        """

    @abstractmethod
    def sync_with_tree(self, root: LemmaNode) -> None:
        """Synchronize selection state with the proof graph rooted at `root`.

        Refresh selectable positions to reflect the current graph structure
        and node statuses.
        """
