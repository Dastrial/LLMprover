"""Tests for llmprover.llm_attempt_strategy."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llmprover.agent_registry import AgentRegistry
from llmprover.domain import (
    AttemptRecord,
    CoqcResult,
    Goal,
    LemmaNode,
    Polarity,
    ProofAttempt,
    statement_for_polarity,
)
from llmprover.llm_attempt_strategy import LLMAttemptStrategy, format_specs
from llmprover.llm_client import CompletionResult, OpenAIClient, TokenUsage
from llmprover.model_registry import ModelRegistry
from llmprover.prompts import fill_prompt, load_prompt
from llmprover.prover_agents.prover_agent import ProverAgent
from llmprover.utils import format_histories

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "llmprover" / "prompts"
GOAL = Goal(name="plus_n0", statement="forall n : nat, n + 0 = n.")
EMPTY = "No previous attempts.\n"
EXPECTED_SYSTEM = load_prompt(PROMPTS_DIR / "strategy_system.txt")


class StubAgent(ProverAgent):
    DEFAULT_SPEC = "stub: cheap direct proof"

    def __init__(self, model: object) -> None:
        self.model = model

    def prove(
        self, node: LemmaNode, polarity: Polarity
    ) -> tuple[ProofAttempt, TokenUsage]:
        raise NotImplementedError


def attempt_record(
    script: str,
    error: str,
    *,
    polarity: Polarity = Polarity.Positive,
) -> AttemptRecord:
    return AttemptRecord(
        attempt=ProofAttempt(
            goal=GOAL,
            polarity=polarity,
            script=script,
            new_lemmas=[],
        ),
        rocq_error=CoqcResult(success=False, stderr=error),
    )


def make_strategy(
    *,
    llm_client: MagicMock | None = None,
    agent_id: str | None = None,
    model_id: str = "gpt-5-nano",
) -> tuple[LLMAttemptStrategy, str, MagicMock]:
    agent_registry = AgentRegistry()
    registered_agent_id = agent_registry.register(StubAgent)
    if agent_id is None:
        agent_id = registered_agent_id

    model_registry = ModelRegistry()
    model_registry.register(OpenAIClient, model_id, "cheap OpenAI model")

    strategy_llm = llm_client if llm_client is not None else MagicMock()
    strategy = LLMAttemptStrategy(agent_registry, model_registry, strategy_llm)
    return strategy, agent_id, strategy_llm


def expected_user(
    node: LemmaNode,
    polarity: Polarity,
    agent_specs: dict[str, str],
    model_specs: dict[str, str],
) -> str:
    this_formula_attempts, opposite_attempts = format_histories(node, empty=EMPTY)
    if polarity is Polarity.Negative:
        this_formula_attempts, opposite_attempts = (
            opposite_attempts,
            this_formula_attempts,
        )
    return fill_prompt(
        load_prompt(PROMPTS_DIR / "strategy_user.txt"),
        statement=statement_for_polarity(node.goal.statement, polarity),
        agent_specs=format_specs(agent_specs),
        model_specs=format_specs(model_specs),
        this_formula_attempts=this_formula_attempts,
        opposite_attempts=opposite_attempts,
    )


def test_format_specs_joins_id_and_description_lines() -> None:
    assert format_specs({"agent_1": "cheap", "agent_2": "heavy"}) == (
        "agent_1: cheap\nagent_2: heavy"
    )


def test_parse_response_wires_agent_to_model() -> None:
    strategy, agent_id, _ = make_strategy()
    fake_model = MagicMock()

    with patch.object(
        strategy.model_registry, "get", return_value=fake_model
    ) as get_model:
        agent = strategy.parse_response(f"{agent_id}\ngpt-5-nano")

    get_model.assert_called_once_with("gpt-5-nano")
    assert isinstance(agent, StubAgent)
    assert agent.model is fake_model


def test_parse_response_rejects_wrong_line_count() -> None:
    strategy, _, _ = make_strategy()

    with pytest.raises(ValueError, match="expected 2 lines"):
        strategy.parse_response("agent_1")


def test_parse_response_unknown_agent_raises_key_error() -> None:
    strategy, _, _ = make_strategy()
    fake_model = MagicMock()

    with patch.object(strategy.model_registry, "get", return_value=fake_model):
        with pytest.raises(KeyError, match="Unknown agent id"):
            strategy.parse_response("agent_999\ngpt-5-nano")


def test_parse_response_unknown_model_raises_key_error() -> None:
    strategy, agent_id, _ = make_strategy()

    with pytest.raises(KeyError, match="Unknown model id"):
        strategy.parse_response(f"{agent_id}\nmissing-model")


def test_decide_agent_builds_prompt_and_parses_llm_output() -> None:
    node = LemmaNode(
        goal=GOAL,
        positive=[attempt_record("induction n.", "Error on line 1.")],
        negative=[attempt_record("intro H.", "Error B.", polarity=Polarity.Negative)],
    )
    strategy_llm = MagicMock()
    strategy, agent_id, _ = make_strategy(llm_client=strategy_llm)
    strategy_llm.complete.return_value = CompletionResult(
        text=f"{agent_id}\ngpt-5-nano", usage=TokenUsage(11, 2)
    )
    fake_model = MagicMock()

    with patch.object(strategy.model_registry, "get", return_value=fake_model):
        agent, usage = strategy.decide_agent(node, Polarity.Positive)

    assert isinstance(agent, StubAgent)
    assert agent.model is fake_model
    assert usage == TokenUsage(11, 2)
    strategy_llm.complete.assert_called_once_with(
        [
            {"role": "system", "content": EXPECTED_SYSTEM},
            {
                "role": "user",
                "content": expected_user(
                    node,
                    Polarity.Positive,
                    strategy.agent_registry.specs(),
                    strategy.model_registry.specs(),
                ),
            },
        ]
    )


def test_decide_agent_negative_polarity_uses_negated_statement() -> None:
    node = LemmaNode(
        goal=GOAL,
        positive=[attempt_record("induction n.", "Error on line 1.")],
        negative=[attempt_record("intro H.", "Error B.", polarity=Polarity.Negative)],
    )
    strategy_llm = MagicMock()
    strategy, agent_id, _ = make_strategy(llm_client=strategy_llm)
    strategy_llm.complete.return_value = CompletionResult(
        text=f"{agent_id}\ngpt-5-nano", usage=TokenUsage()
    )
    fake_model = MagicMock()

    with patch.object(strategy.model_registry, "get", return_value=fake_model):
        strategy.decide_agent(node, Polarity.Negative)

    user_prompt = strategy_llm.complete.call_args.args[0][1]["content"]
    assert user_prompt == expected_user(
        node,
        Polarity.Negative,
        strategy.agent_registry.specs(),
        strategy.model_registry.specs(),
    )
    assert statement_for_polarity(GOAL.statement, Polarity.Negative) in user_prompt


def test_decide_agent_empty_histories_use_default_empty_text() -> None:
    node = LemmaNode(goal=GOAL)
    strategy_llm = MagicMock()
    strategy, agent_id, _ = make_strategy(llm_client=strategy_llm)
    strategy_llm.complete.return_value = CompletionResult(
        text=f"{agent_id}\ngpt-5-nano", usage=TokenUsage()
    )
    fake_model = MagicMock()

    with patch.object(strategy.model_registry, "get", return_value=fake_model):
        strategy.decide_agent(node, Polarity.Negative)

    user_prompt = strategy_llm.complete.call_args.args[0][1]["content"]
    assert user_prompt == expected_user(
        node,
        Polarity.Negative,
        strategy.agent_registry.specs(),
        strategy.model_registry.specs(),
    )
    assert EMPTY in user_prompt
