"""Tests for llmprover.agent_registry."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from llmprover.agent_registry import AgentRegistry
from llmprover.domain import LemmaNode, Polarity, ProofAttempt
from llmprover.prover_agents.prover_agent import ProverAgent


class StubAgentA(ProverAgent):
    DEFAULT_SPEC = "stub A: cheap direct proof"
    init_count = 0

    def __init__(self, model: object) -> None:
        StubAgentA.init_count += 1
        self.model = model

    def prove(self, node: LemmaNode, polarity: Polarity) -> ProofAttempt:
        raise NotImplementedError


class StubAgentB(ProverAgent):
    DEFAULT_SPEC = "stub B: expensive decomposition"

    def __init__(self, model: object) -> None:
        self.model = model

    def prove(self, node: LemmaNode, polarity: Polarity) -> ProofAttempt:
        raise NotImplementedError


@pytest.fixture(autouse=True)
def reset_stub_init_count() -> None:
    StubAgentA.init_count = 0


def test_register_returns_sequential_ids() -> None:
    registry = AgentRegistry()
    assert registry.register(StubAgentA) == "agent_1"
    assert registry.register(StubAgentB) == "agent_2"


def test_specs_reads_class_vars_without_instantiating() -> None:
    registry = AgentRegistry()
    id_a = registry.register(StubAgentA)
    id_b = registry.register(StubAgentB)

    assert registry.specs() == {
        id_a: StubAgentA.DEFAULT_SPEC,
        id_b: StubAgentB.DEFAULT_SPEC,
    }
    assert StubAgentA.init_count == 0


def test_get_creates_instance_with_provided_model() -> None:
    model = MagicMock()
    registry = AgentRegistry()
    agent_id = registry.register(StubAgentA)

    agent = registry.get(agent_id, model)

    assert isinstance(agent, StubAgentA)
    assert agent.model is model


def test_get_unknown_id_raises_key_error() -> None:
    registry = AgentRegistry()

    with pytest.raises(KeyError, match="Unknown agent id"):
        registry.get("agent_999", MagicMock())


def test_get_is_lazy_until_called() -> None:
    registry = AgentRegistry()
    agent_id = registry.register(StubAgentA)

    assert StubAgentA.init_count == 0
    registry.get(agent_id, MagicMock())
    assert StubAgentA.init_count == 1
