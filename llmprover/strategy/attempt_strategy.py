"""Strategy abstract class."""

from __future__ import annotations

from abc import ABC, abstractmethod

from llmprover.prover_agents.agent_registry import AgentRegistry
from llmprover.domain import LemmaNode, Polarity
from llmprover.llm.client import TokenUsage
from llmprover.prover_agents.prover_agent import ProverAgent


class AttemptStrategy(ABC):
    """Abstract class for attempt strategies."""

    def __init__(self, agent_registry: AgentRegistry) -> None:
        self.agent_registry = agent_registry

    @abstractmethod
    def decide_agent(
        self, node: LemmaNode, polarity: Polarity
    ) -> tuple[ProverAgent, TokenUsage]:
        """Choose a ready-to-use agent for *node* at *polarity*.

        Returns the agent and any LLM token usage spent deciding.
        """
