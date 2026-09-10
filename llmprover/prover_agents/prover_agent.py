"""Proof agent abstract class"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import ClassVar

from llmprover.domain import LemmaNode, Polarity, ProofAttempt
from llmprover.llm_client import TokenUsage


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
    def prove(
        self, node: LemmaNode, polarity: Polarity
    ) -> tuple[ProofAttempt, TokenUsage]:
        """Propose a proof for ``node.goal`` at ``polarity``.

        ``node`` holds the canonical statement ``P`` and both attempt histories.
        ``polarity`` selects whether to attack ``P`` or ``~P``.
        Returns the attempt and LLM token usage spent to produce it.
        """

    def describe(self, usage: TokenUsage | None = None) -> str:
        """One-line description stored on ``ProofAttempt.agent``.

        Default: class name, plus model and reasoning tokens when this agent
        wraps an ``LLMClient``. Subclasses may override.
        """
        parts = [type(self).__name__]
        model = getattr(self, "model", None)
        if model is None:
            return parts[0]
        model_name = getattr(model, "model", None)
        if not isinstance(model_name, str) or not model_name:
            model_name = "?"
        effort = getattr(model, "reasoning_effort", None)
        if isinstance(effort, str) and effort:
            parts.append(f"model={model_name} reasoning={effort}")
        else:
            parts.append(f"model={model_name}")
        return ", ".join(parts)
