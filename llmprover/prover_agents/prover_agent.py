"""Proof agent abstract class"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..domain import LemmaNode, Polarity, ProofAttempt


class ProverAgent(ABC):
    """Abstract proof agent"""

    @abstractmethod
    def prove(self, node: LemmaNode, polarity: Polarity) -> ProofAttempt:
        """Propose a proof for ``node.goal`` at ``polarity``.

        ``node`` holds the canonical statement ``P`` and both attempt histories.
        ``polarity`` selects whether to attack ``P`` or ``~P``.
        """
