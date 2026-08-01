"""Tests for llmprover.orchestrator."""

from __future__ import annotations

from unittest.mock import MagicMock

from llmprover.domain import (
    CoqcResult,
    Goal,
    LemmaNode,
    LemmaStatus,
    Polarity,
    Position,
    ProofAttempt,
)
from llmprover.llm_client import TokenUsage
from llmprover.orchestrator import Orchestrator

GOAL = Goal(name="plus_n0", statement="forall n : nat, n + 0 = n.")
CHILD_A = Goal(name="base", statement="0 + 0 = 0.")
CHILD_B = Goal(name="step", statement="forall n, n + 0 = n -> S n + 0 = S n.")
ROOT: Position = ()
POS_CHILD0: Position = ((Polarity.Positive, 0),)
POS_CHILD1: Position = ((Polarity.Positive, 1),)


def with_usage(
    attempts: list[ProofAttempt],
    *,
    usage: TokenUsage = TokenUsage(5, 1),
) -> list[tuple[ProofAttempt, TokenUsage]]:
    return [(attempt, usage) for attempt in attempts]


def make_orchestrator(
    *,
    positions_and_polarities: list[tuple[Position, Polarity]],
    attempts: list[ProofAttempt],
    coqc_results: list[CoqcResult],
    decide_usage: TokenUsage = TokenUsage(2, 1),
    prove_usage: TokenUsage = TokenUsage(5, 1),
) -> tuple[Orchestrator, MagicMock, MagicMock, MagicMock]:
    selection = MagicMock()
    selection.select_lemma.side_effect = positions_and_polarities

    agent = MagicMock()
    agent.prove.side_effect = with_usage(attempts, usage=prove_usage)

    attempt_strategy = MagicMock()
    attempt_strategy.decide_agent.return_value = (agent, decide_usage)

    checker = MagicMock()
    checker.check_attempt.side_effect = coqc_results

    orchestrator = Orchestrator(attempt_strategy, checker, selection)
    return orchestrator, attempt_strategy, agent, selection


def test_prove_succeeds_on_single_direct_attempt() -> None:
    attempt = ProofAttempt(goal=GOAL, script="induction n.", new_lemmas=[])
    orchestrator, attempt_strategy, agent, selection = make_orchestrator(
        positions_and_polarities=[(ROOT, Polarity.Positive)],
        attempts=[attempt],
        coqc_results=[CoqcResult(success=True)],
    )

    node = orchestrator.prove(GOAL)

    assert node.status is LemmaStatus.Proved
    assert len(node.positive) == 1
    assert node.positive[0].attempt is attempt
    assert node.positive[0].rocq_error.success
    attempt_strategy.decide_agent.assert_called_once()
    agent.prove.assert_called_once()
    selection.update.assert_called_once()
    record, position = selection.update.call_args.args
    assert position == ROOT
    assert record.attempt is attempt


def test_prove_accumulates_token_usage_from_decide_and_prove() -> None:
    attempt = ProofAttempt(goal=GOAL, script="induction n.", new_lemmas=[])
    # First loop uses tokens; second would exceed max_input_tokens if still open,
    # but success stops the loop.
    orchestrator, _, _, _ = make_orchestrator(
        positions_and_polarities=[(ROOT, Polarity.Positive)],
        attempts=[attempt],
        coqc_results=[CoqcResult(success=True)],
        decide_usage=TokenUsage(100, 10),
        prove_usage=TokenUsage(200, 20),
    )

    node = orchestrator.prove(GOAL, max_input_tokens=500, max_output_tokens=500)
    assert node.status is LemmaStatus.Proved


def test_prove_stops_when_token_budget_is_exhausted() -> None:
    failed = ProofAttempt(goal=GOAL, script="admit.", new_lemmas=[])
    orchestrator, attempt_strategy, _, _ = make_orchestrator(
        positions_and_polarities=[
            (ROOT, Polarity.Positive),
            (ROOT, Polarity.Negative),
            (ROOT, Polarity.Positive),
        ],
        attempts=[failed, failed, failed],
        coqc_results=[
            CoqcResult(success=False, stderr="e"),
            CoqcResult(success=False, stderr="e"),
            CoqcResult(success=False, stderr="e"),
        ],
        decide_usage=TokenUsage(60, 0),
        prove_usage=TokenUsage(50, 0),
    )

    node = orchestrator.prove(GOAL, max_input_tokens=150, max_output_tokens=1000)

    assert node.status is LemmaStatus.Open
    # First attempt: 110 tokens; second: 220 >= 150 → stop before third.
    assert attempt_strategy.decide_agent.call_count == 2


def test_prove_stops_when_refuted() -> None:
    attempt = ProofAttempt(
        goal=GOAL,
        polarity=Polarity.Negative,
        script="intro H. discriminate.",
        new_lemmas=[],
    )
    orchestrator, _, _, _ = make_orchestrator(
        positions_and_polarities=[(ROOT, Polarity.Negative)],
        attempts=[attempt],
        coqc_results=[CoqcResult(success=True)],
    )

    node = orchestrator.prove(GOAL)

    assert node.status is LemmaStatus.Refuted
    assert len(node.negative) == 1


def test_prove_decomposition_then_child_success() -> None:
    parent_attempt = ProofAttempt(
        goal=GOAL,
        script="induction n.",
        new_lemmas=[CHILD_A, CHILD_B],
    )

    selection = MagicMock()
    selection.select_lemma.side_effect = [
        (ROOT, Polarity.Positive),
        (POS_CHILD0, Polarity.Positive),
        (POS_CHILD1, Polarity.Positive),
    ]

    agent = MagicMock()

    def prove_side_effect(
        node: LemmaNode, polarity: Polarity
    ) -> tuple[ProofAttempt, TokenUsage]:
        if node.goal.name == GOAL.name:
            return parent_attempt, TokenUsage(1, 1)
        if node.goal.statement == CHILD_A.statement:
            return (
                ProofAttempt(goal=node.goal, script="reflexivity.", new_lemmas=[]),
                TokenUsage(1, 1),
            )
        return (
            ProofAttempt(
                goal=node.goal, script="intros. reflexivity.", new_lemmas=[]
            ),
            TokenUsage(1, 1),
        )

    agent.prove.side_effect = prove_side_effect
    attempt_strategy = MagicMock()
    attempt_strategy.decide_agent.return_value = (agent, TokenUsage(1, 1))
    checker = MagicMock()
    checker.check_attempt.return_value = CoqcResult(success=True)

    orchestrator = Orchestrator(attempt_strategy, checker, selection)
    node = orchestrator.prove(GOAL)

    assert node.status is LemmaStatus.Proved
    assert len(node.positive) == 1
    assert len(node.positive[0].lemmas) == 2
    assert node.positive[0].lemmas[0].goal.name == "base_1"
    assert node.positive[0].lemmas[1].goal.name == "step_2"
    assert node.positive[0].lemmas[0].status is LemmaStatus.Proved
    assert node.positive[0].lemmas[1].status is LemmaStatus.Proved
    assert selection.update.call_count == 3


def test_prove_gives_unique_lemma_names_across_attempts() -> None:
    first = ProofAttempt(goal=GOAL, script="admit.", new_lemmas=[CHILD_A])
    second = ProofAttempt(
        goal=GOAL,
        polarity=Polarity.Negative,
        script="admit.",
        new_lemmas=[CHILD_A],
    )
    closing = ProofAttempt(goal=GOAL, script="induction n.", new_lemmas=[])

    selection = MagicMock()
    selection.select_lemma.side_effect = [
        (ROOT, Polarity.Positive),
        (ROOT, Polarity.Negative),
        (ROOT, Polarity.Positive),
    ]
    agent = MagicMock()
    agent.prove.side_effect = with_usage([first, second, closing])
    attempt_strategy = MagicMock()
    attempt_strategy.decide_agent.return_value = (agent, TokenUsage())
    checker = MagicMock()
    checker.check_attempt.side_effect = [
        CoqcResult(success=True),
        CoqcResult(success=True),
        CoqcResult(success=True),
    ]

    node = Orchestrator(attempt_strategy, checker, selection).prove(GOAL)

    assert node.positive[0].lemmas[0].goal.name == "base_1"
    assert node.negative[0].lemmas[0].goal.name == "base_2"
    assert node.status is LemmaStatus.Proved


def test_prove_failed_decomposition_keeps_child_lemmas_in_history() -> None:
    failed = ProofAttempt(goal=GOAL, script="admit.", new_lemmas=[CHILD_A, CHILD_B])
    closing = ProofAttempt(goal=GOAL, script="induction n.", new_lemmas=[])

    selection = MagicMock()
    selection.select_lemma.side_effect = [
        (ROOT, Polarity.Positive),
        (ROOT, Polarity.Positive),
    ]
    agent = MagicMock()
    agent.prove.side_effect = with_usage([failed, closing])
    attempt_strategy = MagicMock()
    attempt_strategy.decide_agent.return_value = (agent, TokenUsage())
    checker = MagicMock()
    checker.check_attempt.side_effect = [
        CoqcResult(success=False, stderr="fail"),
        CoqcResult(success=True),
    ]

    node = Orchestrator(attempt_strategy, checker, selection).prove(GOAL)

    assert len(node.positive[0].lemmas) == 2
    assert node.positive[0].lemmas[0].goal.name == "base_1"
    assert node.positive[0].lemmas[1].goal.name == "step_2"
    assert node.status is LemmaStatus.Proved
    record, _position = selection.update.call_args_list[0].args
    assert len(record.lemmas) == 2


def test_prove_attacks_selected_position_and_polarity() -> None:
    parent = ProofAttempt(goal=GOAL, script="exact I.", new_lemmas=[CHILD_A])
    closing = ProofAttempt(goal=GOAL, script="induction n.", new_lemmas=[])
    selection = MagicMock()
    selection.select_lemma.side_effect = [
        (ROOT, Polarity.Positive),
        (POS_CHILD0, Polarity.Negative),
        (ROOT, Polarity.Positive),
    ]

    calls: list[tuple[str, Polarity]] = []

    def prove_side_effect(
        node: LemmaNode, polarity: Polarity
    ) -> tuple[ProofAttempt, TokenUsage]:
        calls.append((node.goal.statement, polarity))
        if len(calls) == 1:
            return parent, TokenUsage()
        if len(calls) == 2:
            return (
                ProofAttempt(
                    goal=node.goal,
                    polarity=polarity,
                    script="intro H. exact H.",
                    new_lemmas=[],
                ),
                TokenUsage(),
            )
        return closing, TokenUsage()

    agent = MagicMock()
    agent.prove.side_effect = prove_side_effect
    attempt_strategy = MagicMock()
    attempt_strategy.decide_agent.return_value = (agent, TokenUsage())
    checker = MagicMock()
    checker.check_attempt.return_value = CoqcResult(success=True)

    node = Orchestrator(attempt_strategy, checker, selection).prove(GOAL)

    assert calls[0] == (GOAL.statement, Polarity.Positive)
    assert calls[1] == (CHILD_A.statement, Polarity.Negative)
    assert calls[2] == (GOAL.statement, Polarity.Positive)
    assert node.positive[0].lemmas[0].status is LemmaStatus.Refuted
    assert node.status is LemmaStatus.Proved
