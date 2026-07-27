"""Proof agent abstract class"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import ClassVar

from ..domain import LemmaNode, Polarity, ProofAttempt


class ProverAgent(ABC):
    """Abstract proof agent"""

    DEFAULT_SPEC: ClassVar[str]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return  # laisser les intermédiaires abstraits tranquilles
        if not getattr(cls, "DEFAULT_SPEC", None):
            raise TypeError(f"{cls.__name__} must define DEFAULT_SPEC")

    @abstractmethod
    def prove(self, node: LemmaNode, polarity: Polarity) -> ProofAttempt:
        """Propose a proof for ``node.goal`` at ``polarity``.

        ``node`` holds the canonical statement ``P`` and both attempt histories.
        ``polarity`` selects whether to attack ``P`` or ``~P``.
        """
