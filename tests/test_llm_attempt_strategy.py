"""Tests for llmprover.llm_attempt_strategy."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

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
from llmprover.history_presenter import DeterministicHistoryPresenter
from llmprover.llm_attempt_strategy import (
    MAX_AGENT_LEAD,
    LLMAttemptStrategy,
    format_specs,
)
from llmprover.llm_client import CompletionResult, OpenAIClient, TokenUsage
from llmprover.model_registry import ModelRegistry
from llmprover.prompts import (
    PromptMessage,
    PromptPart,
    fill_prompt,
)
from llmprover.prover_agents.prover_agent import ProverAgent

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "llmprover" / "prompts"
GOAL = Goal(name="plus_n0", statement="forall n : nat, n + 0 = n.")
EMPTY = "No previous attempts.\n"
EXPECTED_SYSTEM = (
    (PROMPTS_DIR / "strategy_system.txt").read_text(encoding="utf-8").strip()
)


class StubAgent(ProverAgent):
    DEFAULT_SPEC = "stub: cheap direct proof"

    def __init__(self, model: object) -> None:
        self.model = model

    def prove(
        self, node: LemmaNode, polarity: Polarity
    ) -> tuple[ProofAttempt, TokenUsage]:
        raise NotImplementedError


class StubAgentB(ProverAgent):
    DEFAULT_SPEC = "stub B: decomposition"

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
    agent: str = "",
) -> AttemptRecord:
    return AttemptRecord(
        attempt=ProofAttempt(
            goal=GOAL,
            polarity=polarity,
            script=script,
            new_lemmas=[],
            agent=agent,
        ),
        rocq_error=CoqcResult(success=False, stderr=error),
    )


def make_strategy(
    *,
    llm_client: MagicMock | None = None,
    agent_id: str | None = None,
    model_id: str = "gpt-5-nano",
    extra_agents: tuple[type[ProverAgent], ...] = (),
) -> tuple[LLMAttemptStrategy, str, MagicMock]:
    agent_registry = AgentRegistry()
    registered_agent_id = agent_registry.register(StubAgent)
    for cls in extra_agents:
        agent_registry.register(cls)
    if agent_id is None:
        agent_id = registered_agent_id

    model_registry = ModelRegistry()
    model_registry.register(OpenAIClient, model_id, "cheap OpenAI model")

    strategy_llm = llm_client if llm_client is not None else MagicMock()
    strategy = LLMAttemptStrategy(agent_registry, model_registry, strategy_llm)
    return strategy, agent_id, strategy_llm


def node_with_agent_attempts(*agents: str) -> LemmaNode:
    return LemmaNode(
        goal=GOAL,
        positive=[attempt_record("auto.", "fail", agent=agent) for agent in agents],
    )


def expected_messages(
    node: LemmaNode,
    polarity: Polarity,
    agent_specs: dict[str, str],
    model_specs: dict[str, str],
) -> list[PromptMessage]:
    presenter = DeterministicHistoryPresenter.strategy()
    attempt_history, _ = presenter.present_prompt_block(node)
    before = fill_prompt(
        (PROMPTS_DIR / "strategy_user_before.txt").read_text(encoding="utf-8").strip(),
        header=node.goal.environment.header,
        agent_specs=format_specs(agent_specs),
        model_specs=format_specs(model_specs),
    )
    if not before.endswith("\n"):
        before = f"{before}\n"
    after = fill_prompt(
        (PROMPTS_DIR / "strategy_user_after.txt").read_text(encoding="utf-8").strip(),
        statement=statement_for_polarity(node.goal.statement, polarity),
    )
    if after and not after.endswith("\n"):
        after = f"{after}\n"
    history = (
        attempt_history if attempt_history.endswith("\n") else f"{attempt_history}\n"
    )
    return [
        PromptMessage.text("system", EXPECTED_SYSTEM, cache_breakpoint=True),
        PromptMessage(
            role="user",
            parts=(
                PromptPart(before + history, cache_breakpoint=True),
                PromptPart(after),
            ),
        ),
    ]


def expected_user(
    node: LemmaNode,
    polarity: Polarity,
    agent_specs: dict[str, str],
    model_specs: dict[str, str],
) -> str:
    return expected_messages(node, polarity, agent_specs, model_specs)[1].joined_text()


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


def test_parse_response_falls_back_when_line_count_is_wrong() -> None:
    strategy, _, _ = make_strategy()
    fake_model = MagicMock()

    with patch.object(
        strategy.model_registry, "get", return_value=fake_model
    ) as get_model:
        agent = strategy.parse_response("I am unsure")

    get_model.assert_called_once_with("gpt-5-nano")
    assert isinstance(agent, StubAgent)
    assert agent.model is fake_model


def test_parse_response_unknown_agent_falls_back_to_first() -> None:
    strategy, _, _ = make_strategy()
    fake_model = MagicMock()

    with patch.object(strategy.model_registry, "get", return_value=fake_model):
        agent = strategy.parse_response("agent_999\ngpt-5-nano")

    assert isinstance(agent, StubAgent)
    assert agent.model is fake_model


def test_parse_response_unknown_model_falls_back_to_first() -> None:
    strategy, agent_id, _ = make_strategy()
    fake_model = MagicMock()

    with patch.object(
        strategy.model_registry, "get", return_value=fake_model
    ) as get_model:
        agent = strategy.parse_response(f"{agent_id}\nmissing-model")

    get_model.assert_called_once_with("gpt-5-nano")
    assert isinstance(agent, StubAgent)
    assert agent.model is fake_model


def test_parse_response_finds_ids_in_noisy_fenced_output() -> None:
    strategy, agent_id, _ = make_strategy()
    fake_model = MagicMock()

    with patch.object(
        strategy.model_registry, "get", return_value=fake_model
    ) as get_model:
        agent = strategy.parse_response(f"```\nI pick {agent_id} with gpt-5-nano\n```")

    get_model.assert_called_once_with("gpt-5-nano")
    assert isinstance(agent, StubAgent)
    assert agent.model is fake_model


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
        expected_messages(
            node,
            Polarity.Positive,
            strategy.agent_registry.specs(),
            strategy.model_registry.specs(),
        )
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

    user_prompt = strategy_llm.complete.call_args.args[0][1].joined_text()
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

    user_prompt = strategy_llm.complete.call_args.args[0][1].joined_text()
    assert user_prompt == expected_user(
        node,
        Polarity.Negative,
        strategy.agent_registry.specs(),
        strategy.model_registry.specs(),
    )
    assert EMPTY in user_prompt


def test_decide_agent_includes_agent_description_in_history() -> None:
    node = LemmaNode(
        goal=GOAL,
        positive=[
            attempt_record(
                "induction n.",
                "Error on line 1.",
                agent="DirectAgent, model=gpt-4o-mini",
            )
        ],
    )
    strategy_llm = MagicMock()
    strategy, agent_id, _ = make_strategy(llm_client=strategy_llm)
    strategy_llm.complete.return_value = CompletionResult(
        text=f"{agent_id}\ngpt-5-nano", usage=TokenUsage()
    )

    with patch.object(strategy.model_registry, "get", return_value=MagicMock()):
        strategy.decide_agent(node, Polarity.Positive)

    user_prompt = strategy_llm.complete.call_args.args[0][1].joined_text()
    assert "Agent: DirectAgent, model=gpt-4o-mini" in user_prompt


def test_decide_agent_forces_other_agent_when_lead_reaches_threshold() -> None:
    node = node_with_agent_attempts(*(["StubAgent"] * MAX_AGENT_LEAD))
    strategy_llm = MagicMock()
    strategy, agent_id, _ = make_strategy(
        llm_client=strategy_llm, extra_agents=(StubAgentB,)
    )
    strategy_llm.complete.return_value = CompletionResult(
        text=f"{agent_id}\ngpt-5-nano", usage=TokenUsage()
    )
    fake_model = MagicMock()

    with patch.object(strategy.model_registry, "get", return_value=fake_model):
        agent, _ = strategy.decide_agent(node, Polarity.Positive)

    assert isinstance(agent, StubAgentB)
    assert agent.model is fake_model


def test_decide_agent_keeps_choice_below_lead_threshold() -> None:
    node = node_with_agent_attempts(*(["StubAgent"] * (MAX_AGENT_LEAD - 1)))
    strategy_llm = MagicMock()
    strategy, agent_id, _ = make_strategy(
        llm_client=strategy_llm, extra_agents=(StubAgentB,)
    )
    strategy_llm.complete.return_value = CompletionResult(
        text=f"{agent_id}\ngpt-5-nano", usage=TokenUsage()
    )

    with patch.object(strategy.model_registry, "get", return_value=MagicMock()):
        agent, _ = strategy.decide_agent(node, Polarity.Positive)

    assert isinstance(agent, StubAgent)


def test_decide_agent_does_not_override_already_balanced_choice() -> None:
    node = node_with_agent_attempts(*(["StubAgent"] * MAX_AGENT_LEAD))
    strategy_llm = MagicMock()
    strategy, _, _ = make_strategy(llm_client=strategy_llm, extra_agents=(StubAgentB,))
    strategy_llm.complete.return_value = CompletionResult(
        text="StubAgentB\ngpt-5-nano", usage=TokenUsage()
    )

    with patch.object(strategy.model_registry, "get", return_value=MagicMock()):
        agent, _ = strategy.decide_agent(node, Polarity.Positive)

    assert isinstance(agent, StubAgentB)


def test_decide_agent_keeps_choice_when_other_agent_is_within_threshold() -> None:
    node = node_with_agent_attempts(*(["StubAgent"] * MAX_AGENT_LEAD), "StubAgentB")
    strategy_llm = MagicMock()
    strategy, agent_id, _ = make_strategy(
        llm_client=strategy_llm, extra_agents=(StubAgentB,)
    )
    strategy_llm.complete.return_value = CompletionResult(
        text=f"{agent_id}\ngpt-5-nano", usage=TokenUsage()
    )

    with patch.object(strategy.model_registry, "get", return_value=MagicMock()):
        agent, _ = strategy.decide_agent(node, Polarity.Positive)

    assert isinstance(agent, StubAgent)


def test_decide_agent_forces_when_lead_over_other_reaches_threshold() -> None:
    node = node_with_agent_attempts(
        *(["StubAgent"] * (MAX_AGENT_LEAD + 1)), "StubAgentB"
    )
    strategy_llm = MagicMock()
    strategy, agent_id, _ = make_strategy(
        llm_client=strategy_llm, extra_agents=(StubAgentB,)
    )
    strategy_llm.complete.return_value = CompletionResult(
        text=f"{agent_id}\ngpt-5-nano", usage=TokenUsage()
    )

    with patch.object(strategy.model_registry, "get", return_value=MagicMock()):
        agent, _ = strategy.decide_agent(node, Polarity.Positive)

    assert isinstance(agent, StubAgentB)


def test_decide_agent_counts_both_polarities_for_lead() -> None:
    positive_count = MAX_AGENT_LEAD // 2
    negative_count = MAX_AGENT_LEAD - positive_count
    node = LemmaNode(
        goal=GOAL,
        positive=[
            attempt_record("auto.", "fail", agent="StubAgent")
            for _ in range(positive_count)
        ],
        negative=[
            attempt_record(
                "intro H.",
                "fail",
                polarity=Polarity.Negative,
                agent="StubAgent",
            )
            for _ in range(negative_count)
        ],
    )
    strategy_llm = MagicMock()
    strategy, agent_id, _ = make_strategy(
        llm_client=strategy_llm, extra_agents=(StubAgentB,)
    )
    strategy_llm.complete.return_value = CompletionResult(
        text=f"{agent_id}\ngpt-5-nano", usage=TokenUsage()
    )

    with patch.object(strategy.model_registry, "get", return_value=MagicMock()):
        agent, _ = strategy.decide_agent(node, Polarity.Positive)

    assert isinstance(agent, StubAgentB)
