"""Lemma selection strategy abstract class."""

from __future__ import annotations

from abc import ABC, abstractmethod

from llmprover.domain import AttemptRecord, Polarity, Position


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
    def select_lemma(self) -> tuple[Position, Polarity]:
        """Select the position and polarity of the next lemma to attack."""
