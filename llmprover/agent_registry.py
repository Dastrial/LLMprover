"""Registry of available prover agents."""

from __future__ import annotations

from llmprover.llm_client import LLMClient
from llmprover.prover_agents.prover_agent import ProverAgent


class AgentRegistry:
    """Maps agent ids to ``ProverAgent`` subclasses.

    Specs are read from ``DEFAULT_SPEC`` on the class (no instantiation).
    Instances are created on ``get`` with the caller-provided ``model``, so
    strategy can pick a different LLM per call.
    """

    def __init__(self) -> None:
        self.counter = 0
        self.classes: dict[str, type[ProverAgent]] = {}

    def register(self, cls: type[ProverAgent]) -> str:
        """Register *cls* and return its agent id."""
        self.counter += 1
        agent_id = f"agent_{self.counter}"
        self.classes[agent_id] = cls
        return agent_id

    def specs(self) -> dict[str, str]:
        """Return ``{agent_id: DEFAULT_SPEC}`` for every registered class."""
        return {agent_id: cls.DEFAULT_SPEC for agent_id, cls in self.classes.items()}

    def get(self, agent_id: str, model: LLMClient) -> ProverAgent:
        """Create a ``ProverAgent`` for *agent_id* wired to *model*."""
        if agent_id not in self.classes:
            raise KeyError(f"Unknown agent id: {agent_id!r}")
        return self.classes[agent_id](model)
