"""Registry of available prover agents."""

from __future__ import annotations

from llmprover.history.presenter import HistoryPresenter
from llmprover.llm.client import LLMClient
from llmprover.prover_agents.prover_agent import ProverAgent


class AgentRegistry:
    """Maps agent ids to ``ProverAgent`` subclasses.

    Specs are read from ``DEFAULT_SPEC`` on the class (no instantiation).
    Instances are created on ``get`` with the caller-provided ``model``, so
    strategy can pick a different LLM per call.

    Optional ``history_presenter`` instances registered with a class are
    injected on ``get`` so presenter memory survives agent recreation.
    """

    def __init__(self) -> None:
        self.counter = 0
        self.classes: dict[str, type[ProverAgent]] = {}
        self.presenters: dict[str, HistoryPresenter] = {}

    def register(
        self,
        cls: type[ProverAgent],
        *,
        history_presenter: HistoryPresenter | None = None,
    ) -> str:
        """Register *cls* and return its agent id.

        When *history_presenter* is set, ``get`` passes it to the constructor
        so several agent classes can share one presenter (and its cache).
        """
        self.counter += 1
        agent_id = f"agent_{self.counter}"
        self.classes[agent_id] = cls
        if history_presenter is not None:
            self.presenters[agent_id] = history_presenter
        return agent_id

    def specs(self) -> dict[str, str]:
        """Return ``{agent_id: DEFAULT_SPEC}`` for every registered class."""
        return {agent_id: cls.DEFAULT_SPEC for agent_id, cls in self.classes.items()}

    def get(self, agent_id: str, model: LLMClient) -> ProverAgent:
        """Create a ``ProverAgent`` for *agent_id* wired to *model*."""
        if agent_id not in self.classes:
            raise KeyError(f"Unknown agent id: {agent_id!r}")
        cls = self.classes[agent_id]
        presenter = self.presenters.get(agent_id)
        if presenter is not None:
            return cls(model, history_presenter=presenter)
        return cls(model)
