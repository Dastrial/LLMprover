"""Tests for llmprover.orchestrator.

Ordered to follow public symbols in ``llmprover/orchestrator.py``.
"""

from __future__ import annotations

import shutil
from unittest.mock import MagicMock

import pytest

from llmprover.domain import (
    AttemptRecord,
    CoqcResult,
    Goal,
    LemmaNode,
    LemmaStatus,
    Polarity,
    Position,
    ProofAttempt,
    RocqEnvironment,
)
from llmprover.rocq.equivalence_merger import CYCLE_ERROR
from llmprover.history.presenter import DeterministicHistoryPresenter
from llmprover.llm.client import TokenUsage
from llmprover.orchestrator import (
    Orchestrator,
    assemble_complete_proof,
    format_lemma_block,
    format_usage,
)
from llmprover.rocq.proof_script import ProofScript
from llmprover.rocq.backend import CoqcBackend
from llmprover.utils import format_attempts

GOAL = Goal(name="plus_n0", statement="forall n : nat, n + 0 = n.")
CHILD_A = Goal(name="base", statement="0 + 0 = 0.")
CHILD_B = Goal(name="step", statement="forall n, n + 0 = n -> S n + 0 = S n.")
ROOT: Position = ()
POS_CHILD0: Position = ((Polarity.Positive, 0),)
POS_CHILD1: Position = ((Polarity.Positive, 1),)


def _canonical(statement: str) -> str:
    return statement.strip().rstrip(".").strip()


def _statements_equivalent_by_canonical(
    left: str, right: str, environment=None
) -> bool:
    del environment
    return _canonical(left) == _canonical(right)


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
    checker.statements_equivalent.side_effect = _statements_equivalent_by_canonical

    orchestrator = Orchestrator(attempt_strategy, checker, selection)
    return orchestrator, attempt_strategy, agent, selection


# --- format_usage ---


def test_format_usage_empty() -> None:
    assert format_usage(TokenUsage()) == (
        "tokens in=0 cache_write=0 cache_read=0 out=0 reasoning=0 "
        "cost=$0.000000 models=-"
    )


def test_format_usage_includes_models_cache_and_cost() -> None:
    usage = TokenUsage(
        1_550,
        10,
        reasoning_tokens=3,
        model="claude-haiku-4-5",
        cache_write_tokens=1_000,
        cache_read_tokens=500,
    )
    # regular = 1550 - 1000 - 500 = 50
    expected_cost = (
        50 * 1.00 / 1_000_000
        + 1_000 * 1.25 / 1_000_000
        + 500 * 0.10 / 1_000_000
        + 10 * 5.00 / 1_000_000
    )

    text = format_usage(usage)

    assert "tokens in=1550" in text
    assert "cache_write=1000" in text
    assert "cache_read=500" in text
    assert "out=10" in text
    assert "reasoning=3" in text
    assert f"cost=${expected_cost:.6f}" in text
    assert "models=claude-haiku-4-5" in text


def test_format_usage_unnamed_model_shows_question_mark() -> None:
    text = format_usage(TokenUsage(1, 0, model=""))
    assert "models=?" in text


# --- format_lemma_block ---


def test_format_lemma_block() -> None:
    assert format_lemma_block("helper", "True", "exact I.") == (
        "Lemma helper: True.\nProof.\nexact I.\nQed.\n\n"
    )


# --- assemble_complete_proof ---


def test_assemble_complete_proof_leaf() -> None:
    node = LemmaNode(goal=GOAL)
    node.append(
        AttemptRecord(
            attempt=ProofAttempt(goal=GOAL, script="induction n.", new_lemmas=[]),
            rocq_error=CoqcResult(success=True),
        )
    )

    script = assemble_complete_proof(node)

    assert script == (
        "Lemma plus_n0: forall n : nat, n + 0 = n..\nProof.\ninduction n.\nQed.\n"
    )


def test_assemble_complete_proof_ignores_later_failed_positive_retry() -> None:
    helper = Goal(name="helper", statement="True")
    helper_node = LemmaNode(goal=helper)
    helper_node.append(
        AttemptRecord(
            attempt=ProofAttempt(goal=helper, script="exact I.", new_lemmas=[]),
            rocq_error=CoqcResult(success=True),
        )
    )
    successful = AttemptRecord(
        attempt=ProofAttempt(
            goal=GOAL,
            script="apply {helper}.",
            new_lemmas=[helper],
        ),
        rocq_error=CoqcResult(success=True),
        lemmas=[helper_node],
    )
    node = LemmaNode(goal=GOAL)
    node.append(successful)
    node.append(
        AttemptRecord(
            attempt=ProofAttempt(goal=GOAL, script="broken.", new_lemmas=[]),
            rocq_error=CoqcResult(success=False, stderr="Error."),
        )
    )

    script = assemble_complete_proof(node)

    assert "Lemma helper:" in script
    assert "apply helper." in script
    assert "broken." not in script


def test_assemble_complete_proof_puts_helpers_first() -> None:
    header = "Require Import Nat."
    root = Goal(
        name="plus_n0",
        statement="forall n : nat, n + 0 = n",
        environment=RocqEnvironment(header=header),
    )
    base = Goal(name="base", statement="0 + 0 = 0")
    step = Goal(name="step", statement="forall n, n + 0 = n -> S n + 0 = S n")
    base_node = LemmaNode(goal=base)
    base_node.append(
        AttemptRecord(
            attempt=ProofAttempt(goal=base, script="reflexivity.", new_lemmas=[]),
            rocq_error=CoqcResult(success=True),
        )
    )
    step_node = LemmaNode(goal=step)
    step_node.append(
        AttemptRecord(
            attempt=ProofAttempt(
                goal=step, script="intros. reflexivity.", new_lemmas=[]
            ),
            rocq_error=CoqcResult(success=True),
        )
    )
    node = LemmaNode(goal=root)
    node.append(
        AttemptRecord(
            attempt=ProofAttempt(
                goal=root,
                script="induction n. apply {base}. apply {step}.",
                new_lemmas=[base, step],
            ),
            rocq_error=CoqcResult(success=True),
            lemmas=[base_node, step_node],
        )
    )

    script = assemble_complete_proof(node)

    assert script.startswith("Require Import Nat.\n\nLemma base:")
    assert script.index("Lemma base:") < script.index("Lemma step:")
    assert script.index("Lemma step:") < script.index("Lemma plus_n0:")
    assert "induction n. apply base. apply step." in script
    assert "{base}" not in script


def test_assemble_complete_proof_binds_placeholders_to_child_names() -> None:
    renamed = Goal(name="helper_1", statement="True")
    helper_node = LemmaNode(goal=renamed)
    helper_node.append(
        AttemptRecord(
            attempt=ProofAttempt(goal=renamed, script="exact I.", new_lemmas=[]),
            rocq_error=CoqcResult(success=True),
        )
    )
    proposed = Goal(name="helper", statement="True")
    parent = Goal(name="main", statement="True")
    node = LemmaNode(goal=parent)
    node.append(
        AttemptRecord(
            attempt=ProofAttempt(
                goal=parent,
                script="exact {helper}.",
                new_lemmas=[proposed],
            ),
            rocq_error=CoqcResult(success=True),
            lemmas=[helper_node],
        )
    )

    script = assemble_complete_proof(node)

    assert "Lemma helper_1: True." in script
    assert "exact helper_1." in script
    assert "Lemma helper:" not in script
    assert "{helper}" not in script


def test_assemble_complete_proof_reuses_aliased_helper() -> None:
    shared = Goal(name="base", statement="True")
    shared_node = LemmaNode(goal=shared)
    shared_node.append(
        AttemptRecord(
            attempt=ProofAttempt(goal=shared, script="exact I.", new_lemmas=[]),
            rocq_error=CoqcResult(success=True),
        )
    )
    dup = Goal(name="dup", statement="True")
    parent = Goal(name="main", statement="True")
    node = LemmaNode(goal=parent)
    node.append(
        AttemptRecord(
            attempt=ProofAttempt(
                goal=parent,
                script="exact {dup}.",
                new_lemmas=[dup],
            ),
            rocq_error=CoqcResult(success=True),
            lemmas=[shared_node],
        )
    )

    script = assemble_complete_proof(node)

    assert script.count("Lemma base:") == 1
    assert "Lemma dup:" not in script
    assert "exact I." in script
    assert "Lemma main: True." in script
    assert "exact base." in script
    assert "{dup}" not in script


def test_assemble_complete_proof_rejects_refuted() -> None:
    node = LemmaNode(goal=GOAL)
    node.append(
        AttemptRecord(
            attempt=ProofAttempt(
                goal=GOAL,
                polarity=Polarity.Negative,
                script="intro H. exact H.",
                new_lemmas=[],
            ),
            rocq_error=CoqcResult(success=True),
        )
    )

    with pytest.raises(ValueError, match="complete proof"):
        assemble_complete_proof(node)


def test_assemble_complete_proof_emits_shared_node_once() -> None:
    shared = Goal(name="base", statement="True")
    shared_node = LemmaNode(goal=shared)
    shared_node.append(
        AttemptRecord(
            attempt=ProofAttempt(goal=shared, script="exact I.", new_lemmas=[]),
            rocq_error=CoqcResult(success=True),
        )
    )
    left = Goal(name="left", statement="True")
    right = Goal(name="right", statement="True")
    parent = Goal(name="main", statement="True")
    node = LemmaNode(goal=parent)
    node.append(
        AttemptRecord(
            attempt=ProofAttempt(
                goal=parent,
                script="split.",
                new_lemmas=[left, right],
            ),
            rocq_error=CoqcResult(success=True),
            lemmas=[shared_node, shared_node],
        )
    )

    script = assemble_complete_proof(node)

    assert script.count("Lemma base:") == 1
    assert "Lemma left:" not in script
    assert "Lemma right:" not in script
    assert "exact I." in script
    assert "Lemma main: True." in script


@pytest.mark.skipif(shutil.which("coqc") is None, reason="coqc not installed")
def test_assemble_complete_proof_compiles_with_coqc() -> None:
    helper = Goal(name="helper", statement="True")
    helper_node = LemmaNode(goal=helper)
    helper_node.append(
        AttemptRecord(
            attempt=ProofAttempt(goal=helper, script="exact I.", new_lemmas=[]),
            rocq_error=CoqcResult(success=True),
        )
    )
    root = Goal(name="main", statement="True")
    node = LemmaNode(goal=root)
    node.append(
        AttemptRecord(
            attempt=ProofAttempt(
                goal=root, script="exact {helper}.", new_lemmas=[helper]
            ),
            rocq_error=CoqcResult(success=True),
            lemmas=[helper_node],
        )
    )

    script = assemble_complete_proof(node)
    result = CoqcBackend().check_script(ProofScript(code=script))

    assert result.success, result.stderr


# --- Orchestrator.sync_verbose ---


def test_verbose_syncs_checker_and_history_presenters() -> None:
    strategy_presenter = DeterministicHistoryPresenter.strategy()
    agent_presenter = DeterministicHistoryPresenter.full()
    attempt_strategy = MagicMock()
    attempt_strategy.history_presenter = strategy_presenter
    attempt_strategy.agent_registry = MagicMock()
    attempt_strategy.agent_registry.presenters = {"agent_1": agent_presenter}
    checker = MagicMock()

    Orchestrator(
        attempt_strategy,
        checker,
        MagicMock(),
        verbose=True,
    )

    assert checker.verbose is True
    assert strategy_presenter.verbose is True
    assert agent_presenter.verbose is True


def test_sync_verbose_noop_on_presenters_when_not_verbose() -> None:
    strategy_presenter = DeterministicHistoryPresenter.strategy()
    attempt_strategy = MagicMock()
    attempt_strategy.history_presenter = strategy_presenter
    attempt_strategy.agent_registry = MagicMock()
    attempt_strategy.agent_registry.presenters = {}
    checker = MagicMock()

    Orchestrator(attempt_strategy, checker, MagicMock(), verbose=False)

    assert checker.verbose is False
    assert strategy_presenter.verbose is False


# --- Orchestrator.log ---


def test_log_prints_only_when_verbose(capsys) -> None:
    silent = Orchestrator(MagicMock(), MagicMock(), MagicMock(), verbose=False)
    silent.log("hidden")
    assert capsys.readouterr().out == ""

    noisy = Orchestrator(MagicMock(), MagicMock(), MagicMock(), verbose=True)
    noisy.log("visible")
    assert capsys.readouterr().out == "visible\n"


# --- Orchestrator.print_attempt ---


def test_print_attempt_silent_when_disabled(capsys) -> None:
    attempt = ProofAttempt(goal=GOAL, script="induction n.", new_lemmas=[])
    orchestrator = Orchestrator(MagicMock(), MagicMock(), MagicMock())

    orchestrator.print_attempt(attempt)

    assert capsys.readouterr().out == ""


def test_print_attempts_prints_agent_script(capsys) -> None:
    attempt = ProofAttempt(
        goal=GOAL,
        script="induction n.\nreflexivity.",
        new_lemmas=[],
    )
    orchestrator, _, _, _ = make_orchestrator(
        positions_and_polarities=[(ROOT, Polarity.Positive)],
        attempts=[attempt],
        coqc_results=[CoqcResult(success=True)],
    )
    orchestrator.print_attempts = True

    orchestrator.prove(GOAL)

    assert capsys.readouterr().out == (
        "[orchestrator]   agent output:\n"
        "[orchestrator]   script:\n"
        "[orchestrator]     induction n.\n"
        "[orchestrator]     reflexivity.\n"
    )


def test_print_attempts_prints_raw_search_and_select(capsys) -> None:
    attempt = ProofAttempt(
        goal=GOAL,
        script="reflexivity.",
        new_lemmas=[],
        search_about_output='Search "gcd".\nAbout Nat.add_0_r.',
        select_output="Nat.gcd\nNat.add_0_r",
    )
    orchestrator, _, _, _ = make_orchestrator(
        positions_and_polarities=[(ROOT, Polarity.Positive)],
        attempts=[attempt],
        coqc_results=[CoqcResult(success=True)],
    )
    orchestrator.print_attempts = True

    orchestrator.prove(GOAL)

    out = capsys.readouterr().out
    assert "[orchestrator]   search/about (raw LLM):" in out
    assert 'Search "gcd".' in out
    assert "About Nat.add_0_r." in out
    assert "[orchestrator]   select (raw LLM):" in out
    assert "Nat.gcd" in out
    assert "[orchestrator]   script:" in out


def test_print_attempt_prints_new_lemmas(capsys) -> None:
    attempt = ProofAttempt(
        goal=GOAL,
        script="split.",
        new_lemmas=[CHILD_A, CHILD_B],
    )
    orchestrator = Orchestrator(MagicMock(), MagicMock(), MagicMock())
    orchestrator.print_attempts = True

    orchestrator.print_attempt(attempt)

    out = capsys.readouterr().out
    assert "[orchestrator]   new_lemmas:" in out
    assert f"[orchestrator]     {CHILD_A.name}: {CHILD_A.statement}" in out
    assert f"[orchestrator]     {CHILD_B.name}: {CHILD_B.statement}" in out
    assert "[orchestrator]   script:" in out


def test_verbose_prints_agent_script(capsys) -> None:
    attempt = ProofAttempt(
        goal=GOAL,
        script="induction n.\nreflexivity.",
        new_lemmas=[],
    )
    orchestrator, _, _, _ = make_orchestrator(
        positions_and_polarities=[(ROOT, Polarity.Positive)],
        attempts=[attempt],
        coqc_results=[CoqcResult(success=True)],
    )
    orchestrator.verbose = True
    orchestrator.print_attempts = False

    orchestrator.prove(GOAL)

    out = capsys.readouterr().out
    assert "[orchestrator]   agent output:" in out
    assert "[orchestrator]   script:" in out
    assert "induction n." in out
    assert "reflexivity." in out


# --- Orchestrator.unique_name_lemma ---


def test_unique_name_lemma_preserves_environment() -> None:
    env = RocqEnvironment(
        header="Require Import Reals.",
        allowed_axioms=frozenset({"Classical.classic"}),
    )
    lemma = Goal(name="helper", statement="True.", environment=env)
    orchestrator = Orchestrator(MagicMock(), MagicMock(), MagicMock())

    renamed = orchestrator.unique_name_lemma(lemma)

    assert renamed.name == "helper"
    assert renamed.statement == "True."
    assert renamed.environment is env

    again = orchestrator.unique_name_lemma(lemma)
    assert again.name == "helper_1"


def test_unique_name_lemma_skips_taken_suffixes() -> None:
    lemma = Goal(name="helper", statement="True.")
    orchestrator = Orchestrator(MagicMock(), MagicMock(), MagicMock())
    orchestrator._used_lemma_names = {"helper", "helper_1"}

    renamed = orchestrator.unique_name_lemma(lemma)

    assert renamed.name == "helper_2"


# --- Orchestrator.log_complete_proof ---


def test_log_complete_proof_prints_and_checks(capsys) -> None:
    node = LemmaNode(goal=GOAL)
    node.append(
        AttemptRecord(
            attempt=ProofAttempt(goal=GOAL, script="induction n.", new_lemmas=[]),
            rocq_error=CoqcResult(success=True),
        )
    )
    checker = MagicMock()
    checker.check_script.return_value = CoqcResult(success=True)
    orchestrator = Orchestrator(MagicMock(), checker, MagicMock(), verbose=True)

    result = orchestrator.log_complete_proof(node)

    assert result is not None and result.success
    checker.check_script.assert_called_once()
    script = checker.check_script.call_args.args[0]
    assert isinstance(script, ProofScript)
    assert "Lemma plus_n0:" in script.code
    out = capsys.readouterr().out
    assert "[orchestrator] complete proof:" in out
    assert "Lemma plus_n0:" in out
    assert "[orchestrator]   Lemma plus_n0:" not in out
    assert "[orchestrator] complete proof coqc=ok" in out


def test_log_complete_proof_returns_none_when_not_proved() -> None:
    node = LemmaNode(goal=GOAL)
    checker = MagicMock()
    orchestrator = Orchestrator(MagicMock(), checker, MagicMock(), verbose=True)

    assert orchestrator.log_complete_proof(node) is None
    checker.check_script.assert_not_called()


def test_log_complete_proof_logs_coqc_error(capsys) -> None:
    node = LemmaNode(goal=GOAL)
    node.append(
        AttemptRecord(
            attempt=ProofAttempt(goal=GOAL, script="induction n.", new_lemmas=[]),
            rocq_error=CoqcResult(success=True),
        )
    )
    checker = MagicMock()
    checker.check_script.return_value = CoqcResult(
        success=False,
        stderr='File "t.v", line 1:\nError: Something wrong.',
    )
    orchestrator = Orchestrator(MagicMock(), checker, MagicMock(), verbose=True)

    result = orchestrator.log_complete_proof(node)

    assert result is not None and not result.success
    out = capsys.readouterr().out
    assert "[orchestrator] complete proof coqc=fail" in out
    assert "[orchestrator]   error:" in out
    assert "Something wrong." in out


# --- Orchestrator.prove ---


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


def test_prove_records_agent_class_name_on_attempt() -> None:
    attempt = ProofAttempt(goal=GOAL, script="induction n.", new_lemmas=[])
    orchestrator, _, agent, _ = make_orchestrator(
        positions_and_polarities=[(ROOT, Polarity.Positive)],
        attempts=[attempt],
        coqc_results=[CoqcResult(success=True)],
        prove_usage=TokenUsage(5, 1, 3),
    )

    node = orchestrator.prove(GOAL)

    assert node.positive[0].attempt.agent == type(agent).__name__


def test_prove_resyncs_checker_verbose() -> None:
    attempt = ProofAttempt(goal=GOAL, script="induction n.", new_lemmas=[])
    orchestrator, _, _, _ = make_orchestrator(
        positions_and_polarities=[(ROOT, Polarity.Positive)],
        attempts=[attempt],
        coqc_results=[CoqcResult(success=True)],
    )
    orchestrator.verbose = True
    orchestrator.checker.verbose = False

    orchestrator.prove(GOAL, max_attempts=1)

    assert orchestrator.checker.verbose is True


def test_prove_accumulates_token_usage_from_decide_and_prove() -> None:
    attempt = ProofAttempt(goal=GOAL, script="induction n.", new_lemmas=[])
    # First loop spends a few tokens; success stops before a second attempt.
    orchestrator, _, _, _ = make_orchestrator(
        positions_and_polarities=[(ROOT, Polarity.Positive)],
        attempts=[attempt],
        coqc_results=[CoqcResult(success=True)],
        decide_usage=TokenUsage(100, 10),
        prove_usage=TokenUsage(200, 20),
    )

    node = orchestrator.prove(GOAL, max_cost_usd=1.0)
    assert node.status is LemmaStatus.Proved
    assert orchestrator.last_attempts_used == 1
    assert orchestrator.last_usage == TokenUsage(300, 30)


def test_prove_stops_when_cost_budget_is_exhausted() -> None:
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

    # Each attempt: 110 unnamed input tokens at DEFAULT_PRICE $1 / 1M.
    cost_per_attempt = TokenUsage(110, 0).cost_usd()
    node = orchestrator.prove(GOAL, max_cost_usd=cost_per_attempt * 1.5)

    assert node.status is LemmaStatus.Open
    # First attempt: 1x cost; 1.5x budget allows a second, then stops.
    assert attempt_strategy.decide_agent.call_count == 2


def test_prove_none_max_attempts_is_unlimited() -> None:
    failed = ProofAttempt(goal=GOAL, script="admit.", new_lemmas=[])
    orchestrator, attempt_strategy, _, _ = make_orchestrator(
        positions_and_polarities=[(ROOT, Polarity.Positive)] * 3,
        attempts=[failed, failed, failed],
        coqc_results=[CoqcResult(success=False, stderr="e")] * 3,
        decide_usage=TokenUsage(50, 0),
        prove_usage=TokenUsage(50, 0),
    )

    cost_per_attempt = TokenUsage(100, 0).cost_usd()
    node = orchestrator.prove(
        GOAL,
        max_cost_usd=cost_per_attempt * 2.5,
        max_attempts=None,
    )

    assert node.status is LemmaStatus.Open
    assert attempt_strategy.decide_agent.call_count == 3
    assert orchestrator.last_attempts_used == 3


def test_prove_cost_budget_prices_each_model_separately() -> None:
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
        decide_usage=TokenUsage(1_000_000, 0, model="gpt-4o-mini"),
        prove_usage=TokenUsage(1_000_000, 0, model="mystery-model"),
    )

    # gpt-4o-mini 1M in = $0.15; unknown 1M in = DEFAULT $1 → $1.15 / attempt.
    # $2 allows two attempts; a flat $1/1M on all 2M tokens would stop after one.
    node = orchestrator.prove(GOAL, max_cost_usd=2.0)

    assert node.status is LemmaStatus.Open
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
            ProofAttempt(goal=node.goal, script="intros. reflexivity.", new_lemmas=[]),
            TokenUsage(1, 1),
        )

    agent.prove.side_effect = prove_side_effect
    attempt_strategy = MagicMock()
    attempt_strategy.decide_agent.return_value = (agent, TokenUsage(1, 1))
    checker = MagicMock()
    checker.check_attempt.return_value = CoqcResult(success=True)
    checker.statements_equivalent.side_effect = _statements_equivalent_by_canonical

    orchestrator = Orchestrator(attempt_strategy, checker, selection)
    node = orchestrator.prove(GOAL)

    assert node.status is LemmaStatus.Proved
    assert len(node.positive) == 1
    assert len(node.positive[0].lemmas) == 2
    assert node.positive[0].lemmas[0].goal.name == "base"
    assert node.positive[0].lemmas[1].goal.name == "step"
    assert node.positive[0].lemmas[0].status is LemmaStatus.Proved
    assert node.positive[0].lemmas[1].status is LemmaStatus.Proved
    assert selection.update.call_count == 3


def test_prove_gives_unique_lemma_names_across_attempts() -> None:
    other = Goal(name=CHILD_A.name, statement="1 + 0 = 1.")
    first = ProofAttempt(goal=GOAL, script="admit.", new_lemmas=[CHILD_A])
    second = ProofAttempt(
        goal=GOAL,
        polarity=Polarity.Negative,
        script="admit.",
        new_lemmas=[other],
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
        CoqcResult(success=True),
    ]
    checker.statements_equivalent.side_effect = _statements_equivalent_by_canonical

    node = Orchestrator(attempt_strategy, checker, selection).prove(GOAL)

    assert node.positive[0].lemmas[0].goal.name == "base"
    assert node.negative[0].lemmas[0].goal.name == "base_1"
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
    checker.statements_equivalent.side_effect = _statements_equivalent_by_canonical

    node = Orchestrator(attempt_strategy, checker, selection).prove(GOAL)

    assert len(node.positive[0].lemmas) == 2
    assert node.positive[0].lemmas[0].goal.name == "base"
    assert node.positive[0].lemmas[1].goal.name == "step"
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
    checker.statements_equivalent.side_effect = _statements_equivalent_by_canonical

    node = Orchestrator(attempt_strategy, checker, selection).prove(GOAL)

    assert calls[0] == (GOAL.statement, Polarity.Positive)
    assert calls[1] == (CHILD_A.statement, Polarity.Negative)
    assert calls[2] == (GOAL.statement, Polarity.Positive)
    assert node.positive[0].lemmas[0].status is LemmaStatus.Refuted
    assert node.status is LemmaStatus.Proved


def test_prove_forwards_known_true_to_select_lemma() -> None:
    attempt = ProofAttempt(goal=GOAL, script="induction n.", new_lemmas=[])
    orchestrator, _, _, selection = make_orchestrator(
        positions_and_polarities=[(ROOT, Polarity.Positive)],
        attempts=[attempt],
        coqc_results=[CoqcResult(success=True)],
    )
    orchestrator.known_true = True

    orchestrator.prove(GOAL)

    selection.select_lemma.assert_called_once_with(known_true=True)


def test_prove_known_true_false_by_default() -> None:
    attempt = ProofAttempt(goal=GOAL, script="induction n.", new_lemmas=[])
    orchestrator, _, _, selection = make_orchestrator(
        positions_and_polarities=[(ROOT, Polarity.Positive)],
        attempts=[attempt],
        coqc_results=[CoqcResult(success=True)],
    )

    orchestrator.prove(GOAL)

    selection.select_lemma.assert_called_once_with(known_true=False)


def test_prove_rejects_bare_new_helper_reference() -> None:
    invalid = ProofAttempt(
        goal=GOAL,
        script="apply intermediate.",
        new_lemmas=[Goal(name="intermediate", statement="True.")],
    )
    closing = ProofAttempt(goal=GOAL, script="induction n.", new_lemmas=[])
    orchestrator, _, agent, _ = make_orchestrator(
        positions_and_polarities=[
            (ROOT, Polarity.Positive),
            (ROOT, Polarity.Positive),
        ],
        attempts=[invalid, closing],
        coqc_results=[CoqcResult(success=True)],
    )

    node = orchestrator.prove(GOAL)

    assert len(node.positive) == 2
    failed = node.positive[0]
    assert failed.lemmas == []
    assert failed.rocq_error.success is False
    assert failed.attempt.script == "apply intermediate."
    assert "Invalid helper reference" in failed.rocq_error.stderr
    assert "{intermediate}" in failed.rocq_error.stderr
    assert orchestrator.checker.check_attempt.call_count == 1
    history = format_attempts([failed])
    assert "Invalid helper reference" in history
    assert "1 | apply intermediate." in history
    second_call_node = agent.prove.call_args_list[1].args[0]
    assert second_call_node.positive[0] is failed


def test_prove_accepts_braced_new_helper_reference() -> None:
    attempt = ProofAttempt(
        goal=GOAL,
        script="apply {intermediate}.",
        new_lemmas=[Goal(name="intermediate", statement="True.")],
    )
    closing = ProofAttempt(goal=GOAL, script="induction n.", new_lemmas=[])
    orchestrator, _, _, _ = make_orchestrator(
        positions_and_polarities=[
            (ROOT, Polarity.Positive),
            (ROOT, Polarity.Positive),
        ],
        attempts=[attempt, closing],
        coqc_results=[
            CoqcResult(success=True),
            CoqcResult(success=True),
        ],
    )

    node = orchestrator.prove(GOAL)

    assert node.positive[0].lemmas[0].goal.name == "intermediate"
    assert node.positive[0].rocq_error.success
    assert orchestrator.checker.check_attempt.call_count == 2


def test_prove_drops_attempt_when_helper_equals_ancestor() -> None:
    cyclic = ProofAttempt(
        goal=GOAL,
        script="apply {again}.",
        new_lemmas=[Goal(name="again", statement=GOAL.statement)],
    )
    closing = ProofAttempt(goal=GOAL, script="induction n.", new_lemmas=[])
    orchestrator, _, _, _ = make_orchestrator(
        positions_and_polarities=[
            (ROOT, Polarity.Positive),
            (ROOT, Polarity.Positive),
        ],
        attempts=[cyclic, closing],
        coqc_results=[CoqcResult(success=True)],
    )

    node = orchestrator.prove(GOAL)

    assert len(node.positive) == 2
    assert node.positive[0].lemmas == []
    assert node.positive[0].rocq_error.success is False
    assert node.positive[0].rocq_error.stderr == CYCLE_ERROR.format(
        helper="again",
        ancestor=GOAL.name,
    )
    assert node.status is LemmaStatus.Proved
    assert orchestrator.checker.check_attempt.call_count == 1


def test_prove_aliases_helper_equivalent_to_cousin() -> None:
    parent = ProofAttempt(goal=GOAL, script="split.", new_lemmas=[CHILD_A, CHILD_B])
    selection = MagicMock()
    selection.select_lemma.side_effect = [
        (ROOT, Polarity.Positive),
        (POS_CHILD1, Polarity.Positive),
    ]

    def prove_side_effect(
        node: LemmaNode, polarity: Polarity
    ) -> tuple[ProofAttempt, TokenUsage]:
        del polarity
        if node.goal.statement == GOAL.statement:
            return parent, TokenUsage()
        return (
            ProofAttempt(
                goal=node.goal,
                script="apply {dup}.",
                new_lemmas=[Goal(name="dup", statement=CHILD_A.statement)],
            ),
            TokenUsage(),
        )

    agent = MagicMock()
    agent.prove.side_effect = prove_side_effect
    attempt_strategy = MagicMock()
    attempt_strategy.decide_agent.return_value = (agent, TokenUsage())
    checker = MagicMock()
    checker.check_attempt.side_effect = [
        CoqcResult(success=True),
        CoqcResult(success=True),
    ]
    checker.statements_equivalent.side_effect = _statements_equivalent_by_canonical

    node = Orchestrator(attempt_strategy, checker, selection).prove(
        GOAL, max_attempts=2
    )

    child_a = node.positive[0].lemmas[0]
    child_b = node.positive[0].lemmas[1]
    assert child_b.positive[0].lemmas[0] is child_a
    assert checker.check_attempt.call_count == 2
    helper_names = checker.check_attempt.call_args_list[1].kwargs["helper_names"]
    assert helper_names == [child_a.goal.name]


def test_prove_unique_name_avoids_root_name() -> None:
    helper = Goal(name=GOAL.name, statement="True.")
    parent = ProofAttempt(goal=GOAL, script="exact {plus_n0}.", new_lemmas=[helper])
    child = ProofAttempt(
        goal=Goal(name=f"{GOAL.name}_1", statement="True."),
        script="exact I.",
        new_lemmas=[],
    )
    selection = MagicMock()
    selection.select_lemma.side_effect = [
        (ROOT, Polarity.Positive),
        (POS_CHILD0, Polarity.Positive),
    ]
    agent = MagicMock()

    def prove_side_effect(
        node: LemmaNode, polarity: Polarity
    ) -> tuple[ProofAttempt, TokenUsage]:
        del polarity
        if node.goal.name == GOAL.name:
            return parent, TokenUsage()
        return child, TokenUsage()

    agent.prove.side_effect = prove_side_effect
    attempt_strategy = MagicMock()
    attempt_strategy.decide_agent.return_value = (agent, TokenUsage())
    checker = MagicMock()
    checker.check_attempt.return_value = CoqcResult(success=True)
    checker.statements_equivalent.side_effect = _statements_equivalent_by_canonical

    node = Orchestrator(attempt_strategy, checker, selection).prove(GOAL)

    assert node.positive[0].lemmas[0].goal.name == "plus_n0_1"
