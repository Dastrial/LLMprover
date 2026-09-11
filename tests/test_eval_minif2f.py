"""Tests for eval_minif2f helpers (order follows eval_minif2f.py)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from eval_minif2f import (
    DEFAULT_LEMMA_SELECTOR,
    ProblemResult,
    build_lemma_selection_strategy,
    build_openai_orchestrator,
    format_report,
    load_completed_results,
    parse_args,
    parse_result_from_log,
    prove_one,
    proved_percent_curve,
)
from llmprover.domain import Goal, LemmaStatus
from llmprover.history_presenter import DeterministicHistoryPresenter
from llmprover.llm_attempt_strategy import LLMAttemptStrategy
from llmprover.llm_client import TokenUsage
from llmprover.orchestrator import Orchestrator
from llmprover.positive_retry_lemma_selection_strategy import (
    PositiveRetryLemmaSelectionStrategy,
)
from llmprover.prover_agents.decomposition_about_agent import DecompositionAboutAgent
from llmprover.prover_agents.repair_direct_about_agent import RepairDirectAboutAgent
from llmprover.rocq import CoqcBackend
from llmprover.simple_lemma_selection_strategy import SimpleLemmaSelectionStrategy


def result(
    name: str,
    status: str,
    cost_usd: float,
    attempts: int = 1,
    input_tokens: int = 10,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
    output_tokens: int = 4,
) -> ProblemResult:
    return ProblemResult(
        name=name,
        status=status,
        cost_usd=cost_usd,
        attempts=attempts,
        input_tokens=input_tokens,
        cache_write_tokens=cache_write_tokens,
        cache_read_tokens=cache_read_tokens,
        output_tokens=output_tokens,
    )


# --- build_lemma_selection_strategy ------------------------------------------


def test_build_lemma_selection_strategy() -> None:
    assert isinstance(
        build_lemma_selection_strategy("simple"), SimpleLemmaSelectionStrategy
    )
    assert isinstance(
        build_lemma_selection_strategy("positive-retry"),
        PositiveRetryLemmaSelectionStrategy,
    )

    with pytest.raises(ValueError, match="Unknown lemma selector"):
        build_lemma_selection_strategy("unknown")


# --- build_openai_orchestrator -----------------------------------------------


@pytest.mark.parametrize(
    ("lemma_selector", "expected_cls"),
    [
        ("simple", SimpleLemmaSelectionStrategy),
        ("positive-retry", PositiveRetryLemmaSelectionStrategy),
    ],
)
def test_build_openai_orchestrator_wiring(
    lemma_selector: str,
    expected_cls: type,
) -> None:
    fake_llm = MagicMock(name="strategy_llm")

    with patch(
        "eval_minif2f.OpenAIClient.from_env",
        return_value=fake_llm,
    ) as from_env:
        orchestrator = build_openai_orchestrator(
            verbose=False,
            lemma_selector=lemma_selector,
        )

    from_env.assert_called_once_with(model="gpt-5.6-luna", reasoning_effort="none")

    assert isinstance(orchestrator, Orchestrator)
    assert orchestrator.verbose is False
    assert orchestrator.print_attempts is True
    assert orchestrator.known_true is True
    assert isinstance(orchestrator.checker, CoqcBackend)
    assert isinstance(orchestrator.lemma_selection_strategy, expected_cls)

    strategy = orchestrator.attempt_strategy
    assert isinstance(strategy, LLMAttemptStrategy)
    assert strategy.llm_client is fake_llm
    assert isinstance(strategy.history_presenter, DeterministicHistoryPresenter)

    agent_classes = set(strategy.agent_registry.classes.values())
    assert agent_classes == {RepairDirectAboutAgent, DecompositionAboutAgent}
    assert list(strategy.model_registry.specs()) == ["gpt-5.6-luna-none"]


def test_build_openai_orchestrator_rejects_unknown_selector() -> None:
    with patch("eval_minif2f.OpenAIClient.from_env", return_value=MagicMock()):
        with pytest.raises(ValueError, match="Unknown lemma selector"):
            build_openai_orchestrator(lemma_selector="unknown")


# --- parse_result_from_log ---------------------------------------------------


def test_parse_result_from_log_reads_orchestrator_done(tmp_path: Path) -> None:
    log = tmp_path / "foo.log"
    log.write_text(
        "[orchestrator] start goal statement=True max_cost_usd=0.04 "
        "max_attempts=unlimited\n"
        "[orchestrator] done status=proved attempts=3 "
        "(tokens in=120 cache_write=0 cache_read=0 out=45 reasoning=2 "
        "cost=$0.012345 models=gpt-5.6-luna)\n",
        encoding="utf-8",
    )

    parsed = parse_result_from_log(log)

    assert parsed == ProblemResult(
        name="foo",
        status="proved",
        cost_usd=0.012345,
        attempts=3,
        input_tokens=120,
        cache_write_tokens=0,
        cache_read_tokens=0,
        output_tokens=45,
    )


def test_parse_result_from_log_reads_cache_tokens(tmp_path: Path) -> None:
    log = tmp_path / "cached.log"
    log.write_text(
        "[orchestrator] done status=proved attempts=2 "
        "(tokens in=500 cache_write=100 cache_read=300 out=20 reasoning=0 "
        "cost=$0.001000 models=gpt-5.6-luna)\n",
        encoding="utf-8",
    )

    parsed = parse_result_from_log(log)

    assert parsed is not None
    assert parsed.input_tokens == 500
    assert parsed.cache_write_tokens == 100
    assert parsed.cache_read_tokens == 300


def test_parse_result_from_log_returns_none_when_incomplete(tmp_path: Path) -> None:
    log = tmp_path / "bar.log"
    log.write_text("[orchestrator] attempt 1/unlimited lemma='bar'\n", encoding="utf-8")
    assert parse_result_from_log(log) is None


# --- load_completed_results --------------------------------------------------


def test_load_completed_results_skips_incomplete_logs(tmp_path: Path) -> None:
    (tmp_path / "foo.log").write_text(
        "[orchestrator] done status=open attempts=7 "
        "(tokens in=10 cache_write=0 cache_read=0 out=2 reasoning=0 "
        "cost=$0.040000 models=gpt-5-nano)\n",
        encoding="utf-8",
    )
    (tmp_path / "bar.log").write_text(
        "[orchestrator] attempt 1/unlimited lemma='bar'\n",
        encoding="utf-8",
    )

    completed = load_completed_results(tmp_path)

    assert set(completed) == {"foo"}
    assert completed["foo"].status == "open"
    assert completed["foo"].attempts == 7
    assert completed["foo"].input_tokens == 10
    assert completed["foo"].cache_write_tokens == 0
    assert completed["foo"].cache_read_tokens == 0
    assert completed["foo"].output_tokens == 2


# --- format_report -----------------------------------------------------------


def test_format_report_counts_proved_by_cost_interval() -> None:
    results = [
        result("cheap", "proved", 0.005),
        result("mid", "proved", 0.015, attempts=2),
        result("edge", "proved", 0.04, attempts=4),
        result("over", "proved", 0.041, attempts=5),
        result("open", "open", 0.04, attempts=3),
        result("boom", "error", 0.01),
    ]

    report = format_report(results)

    assert "Theorems proved    : 4 / 6" in report
    assert "[$0.00, $0.01) : 1 theorem(s) proved" in report
    assert "[$0.01, $0.02) : 1 theorem(s) proved" in report
    assert "[$0.03, $0.04] : 1 theorem(s) proved" in report
    assert "($0.04, +∞) : 1 theorem(s) proved" in report
    assert "≤ $0.01 : 1 theorem(s) proved" in report
    assert "≤ $0.04 : 3 theorem(s) proved" in report
    assert "Open               : 1" in report
    assert "Errors             : 1" in report
    assert "Input tokens       : 60 (cache_write=0 cache_read=0)" in report
    assert "Output tokens      : 24" in report


# --- proved_percent_curve ----------------------------------------------------


def test_proved_percent_curve_is_cumulative_over_all_problems() -> None:
    results = [
        result("a", "proved", 0.01),
        result("b", "proved", 0.02),
        result("c", "open", 0.04),
        result("d", "error", 0.03),
    ]

    xs, ys = proved_percent_curve(results)

    assert xs[0] == 0.0
    assert ys[0] == 0.0
    assert xs[1] == pytest.approx(0.01)
    assert ys[1] == pytest.approx(25.0)
    assert xs[2] == pytest.approx(0.02)
    assert ys[2] == pytest.approx(50.0)
    assert xs[-1] == pytest.approx(0.04)
    assert ys[-1] == pytest.approx(50.0)


# --- plot_proved_percent_vs_cost ---------------------------------------------


def test_plot_proved_percent_vs_cost_writes_png(tmp_path: Path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    del matplotlib
    from eval_minif2f import plot_proved_percent_vs_cost

    path = tmp_path / "proved_vs_cost.png"
    plot_proved_percent_vs_cost(
        [result("a", "proved", 0.01), result("b", "open", 0.04)],
        path,
    )
    assert path.is_file()
    assert path.stat().st_size > 0


# --- prove_one ---------------------------------------------------------------


def _fake_orchestrator(
    *,
    status: LemmaStatus,
    usage: TokenUsage | None = None,
    attempts: int = 3,
) -> MagicMock:
    orchestrator = MagicMock()
    node = MagicMock()
    node.status = status
    orchestrator.prove.return_value = node
    orchestrator.last_usage = usage or TokenUsage(
        input_tokens=100,
        output_tokens=20,
        cache_write_tokens=5,
        cache_read_tokens=10,
        model="gpt-5.6-luna",
    )
    orchestrator.last_attempts_used = attempts
    return orchestrator


def test_prove_one_returns_proved_and_logs_complete_proof() -> None:
    goal = Goal(name="foo", statement="True")
    orchestrator = _fake_orchestrator(status=LemmaStatus.Proved, attempts=2)

    with patch(
        "eval_minif2f.build_openai_orchestrator",
        return_value=orchestrator,
    ) as build:
        outcome = prove_one(
            "foo",
            goal,
            max_cost_usd=0.04,
            lemma_selector="simple",
        )

    build.assert_called_once_with(verbose=True, lemma_selector="simple")
    orchestrator.prove.assert_called_once_with(
        goal, max_cost_usd=0.04, max_attempts=None
    )
    orchestrator.log_complete_proof.assert_called_once_with(
        orchestrator.prove.return_value
    )
    assert outcome.name == "foo"
    assert outcome.status == "proved"
    assert outcome.attempts == 2
    assert outcome.input_tokens == 100
    assert outcome.cache_write_tokens == 5
    assert outcome.cache_read_tokens == 10
    assert outcome.output_tokens == 20
    assert outcome.cost_usd == pytest.approx(orchestrator.last_usage.cost_usd())


def test_prove_one_open_skips_complete_proof() -> None:
    goal = Goal(name="bar", statement="False")
    orchestrator = _fake_orchestrator(status=LemmaStatus.Open)

    with patch(
        "eval_minif2f.build_openai_orchestrator",
        return_value=orchestrator,
    ):
        outcome = prove_one("bar", goal, max_cost_usd=0.01)

    orchestrator.log_complete_proof.assert_not_called()
    assert outcome.status == "open"
    assert outcome.name == "bar"


def test_prove_one_exception_returns_error_status() -> None:
    goal = Goal(name="boom", statement="True")
    orchestrator = _fake_orchestrator(status=LemmaStatus.Open, attempts=1)
    orchestrator.prove.side_effect = RuntimeError("coqc missing")

    with patch(
        "eval_minif2f.build_openai_orchestrator",
        return_value=orchestrator,
    ):
        outcome = prove_one("boom", goal, max_cost_usd=0.04)

    orchestrator.log_complete_proof.assert_not_called()
    assert outcome == ProblemResult(
        name="boom",
        status="error",
        cost_usd=orchestrator.last_usage.cost_usd(),
        attempts=1,
        input_tokens=100,
        cache_write_tokens=5,
        cache_read_tokens=10,
        output_tokens=20,
    )


def test_prove_one_defaults_lemma_selector_to_positive_retry() -> None:
    goal = Goal(name="baz", statement="True")
    orchestrator = _fake_orchestrator(status=LemmaStatus.Refuted)

    with patch(
        "eval_minif2f.build_openai_orchestrator",
        return_value=orchestrator,
    ) as build:
        outcome = prove_one("baz", goal, max_cost_usd=0.04)

    build.assert_called_once_with(verbose=True, lemma_selector=DEFAULT_LEMMA_SELECTOR)
    assert outcome.status == "refuted"


# --- parse_args --------------------------------------------------------------


def test_lemma_selector_defaults_to_positive_retry() -> None:
    args = parse_args([])

    assert args.lemma_selector == DEFAULT_LEMMA_SELECTOR == "positive-retry"
    assert args.output_dir is None
