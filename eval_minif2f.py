"""Run the live OpenAI orchestrator on every miniF2F-rocq *test* problem (shuffled).

Uses deterministic history and gpt-5.6-luna with reasoning disabled. The lemma
selector is explicit; outputs default to a selector-specific directory under
``benchmark_runs/``. A rerun skips problems whose log already contains an
orchestrator ``done`` line.

Requires ``OPENAI_API_KEY``, ``coqc``, and ``datasets`` / ``matplotlib``
(``pip install -e '.[bench]'``)::

    python eval_minif2f.py
    python eval_minif2f.py --lemma-selector positive-retry --limit 3
    python eval_minif2f.py --lemma-selector simple
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TextIO

from dotenv import load_dotenv

from llmprover.prover_agents.agent_registry import AgentRegistry
from llmprover.domain import Goal, LemmaStatus
from llmprover.history.presenter import DeterministicHistoryPresenter
from llmprover.strategy.llm_attempt_strategy import LLMAttemptStrategy
from llmprover.llm.client import OpenAIClient
from llmprover.minif2f.loader import load_minif2f
from llmprover.llm.model_registry import ModelRegistry
from llmprover.orchestrator import Orchestrator
from llmprover.strategy.positive_retry_lemma_selection_strategy import (
    PositiveRetryLemmaSelectionStrategy,
)
from llmprover.prover_agents.decomposition_about_agent import DecompositionAboutAgent
from llmprover.prover_agents.repair_direct_about_agent import RepairDirectAboutAgent
from llmprover.rocq.backend import CoqcBackend
from llmprover.strategy.simple_lemma_selection_strategy import SimpleLemmaSelectionStrategy

load_dotenv()

DEFAULT_MAX_COST_USD = 0.04
DEFAULT_OUTPUT_ROOT = Path("benchmark_runs")
DEFAULT_LEMMA_SELECTOR = "positive-retry"
LEMMA_SELECTOR_NAMES = ("simple", "positive-retry")
DONE_RE = re.compile(
    r"^\[orchestrator\] done status=(?P<status>\S+) "
    r"attempts=(?P<attempts>\d+) "
    r"\(tokens in=(?P<input_tokens>\d+) "
    r"(?:cache_write=(?P<cache_write>\d+) cache_read=(?P<cache_read>\d+) )?"
    r"out=(?P<output_tokens>\d+) "
    r"reasoning=(?P<reasoning>\d+) cost=\$(?P<cost>[0-9.]+) "
)
COST_INTERVALS_USD: tuple[tuple[float, float, str], ...] = (
    (0.00, 0.01, "[$0.00, $0.01)"),
    (0.01, 0.02, "[$0.01, $0.02)"),
    (0.02, 0.03, "[$0.02, $0.03)"),
    (0.03, 0.04, "[$0.03, $0.04]"),
)


@dataclass(frozen=True)
class ProblemResult:
    name: str
    status: str
    cost_usd: float
    attempts: int
    input_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    output_tokens: int


class Tee:
    """Write to several text streams (console + problem log)."""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return False


@contextmanager
def tee_stdout(log_file: TextIO) -> Iterator[None]:
    """Duplicate stdout to *log_file* for the duration of the block."""
    original = sys.stdout
    sys.stdout = Tee(original, log_file)
    try:
        yield
    finally:
        sys.stdout = original


def build_lemma_selection_strategy(
    name: str,
) -> SimpleLemmaSelectionStrategy | PositiveRetryLemmaSelectionStrategy:
    if name == "simple":
        return SimpleLemmaSelectionStrategy()
    if name == "positive-retry":
        return PositiveRetryLemmaSelectionStrategy()
    raise ValueError(f"Unknown lemma selector: {name}")


def build_openai_orchestrator(
    *,
    verbose: bool = True,
    print_attempts: bool = True,
    lemma_selector: str = DEFAULT_LEMMA_SELECTOR,
) -> Orchestrator:
    """Live miniF2F prover wiring: deterministic history, gpt-5.6-luna reasoning=none."""
    detailed_history = DeterministicHistoryPresenter.full()
    repair_history = DeterministicHistoryPresenter.full()
    strategy_history = DeterministicHistoryPresenter.strategy()

    agent_registry = AgentRegistry()
    agent_registry.register(RepairDirectAboutAgent, history_presenter=repair_history)
    agent_registry.register(DecompositionAboutAgent, history_presenter=detailed_history)

    model_registry = ModelRegistry()
    model_registry.register(
        OpenAIClient,
        "gpt-5.6-luna",
        "Strongest available model; use for all attempts.",
        reasoning_effort="none",
    )
    strategy_llm = OpenAIClient.from_env(model="gpt-5.6-luna", reasoning_effort="none")
    return Orchestrator(
        attempt_strategy=LLMAttemptStrategy(
            agent_registry,
            model_registry,
            strategy_llm,
            history_presenter=strategy_history,
        ),
        checker=CoqcBackend(),
        lemma_selection_strategy=build_lemma_selection_strategy(lemma_selector),
        verbose=verbose,
        print_attempts=print_attempts,
        known_true=True,
    )


def parse_result_from_log(path: Path) -> ProblemResult | None:
    """Return the result encoded in the last orchestrator ``done`` line, if any."""
    if not path.is_file():
        return None
    last_match: re.Match[str] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        match = DONE_RE.match(line.strip())
        if match is not None:
            last_match = match
    if last_match is None:
        return None
    cache_write = last_match.group("cache_write")
    cache_read = last_match.group("cache_read")
    return ProblemResult(
        name=path.stem,
        status=last_match.group("status"),
        cost_usd=float(last_match.group("cost")),
        attempts=int(last_match.group("attempts")),
        input_tokens=int(last_match.group("input_tokens")),
        cache_write_tokens=int(cache_write) if cache_write is not None else 0,
        cache_read_tokens=int(cache_read) if cache_read is not None else 0,
        output_tokens=int(last_match.group("output_tokens")),
    )


def load_completed_results(output_dir: Path) -> dict[str, ProblemResult]:
    """Map problem name → result for logs that already contain a ``done`` line."""
    if not output_dir.is_dir():
        return {}
    completed: dict[str, ProblemResult] = {}
    for path in output_dir.glob("*.log"):
        result = parse_result_from_log(path)
        if result is not None:
            completed[result.name] = result
    return completed


def result_from_orchestrator(
    name: str, orchestrator: Orchestrator, status: str
) -> ProblemResult:
    usage = orchestrator.last_usage
    return ProblemResult(
        name=name,
        status=status,
        cost_usd=usage.cost_usd(),
        attempts=orchestrator.last_attempts_used,
        input_tokens=usage.input_tokens,
        cache_write_tokens=usage.cache_write_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        output_tokens=usage.output_tokens,
    )


def count_proved_in_interval(
    proved: list[ProblemResult],
    low: float,
    high: float,
    *,
    right_closed: bool,
) -> int:
    if high == float("inf"):
        return sum(1 for item in proved if item.cost_usd > low)
    if right_closed:
        return sum(1 for item in proved if low <= item.cost_usd <= high)
    return sum(1 for item in proved if low <= item.cost_usd < high)


def format_report(results: list[ProblemResult]) -> str:
    total = len(results)
    proved = [item for item in results if item.status == LemmaStatus.Proved.value]
    open_ = [item for item in results if item.status == LemmaStatus.Open.value]
    refuted = [item for item in results if item.status == LemmaStatus.Refuted.value]
    errors = [item for item in results if item.status == "error"]
    total_cost = sum(item.cost_usd for item in results)
    proved_cost = sum(item.cost_usd for item in proved)
    input_tokens = sum(item.input_tokens for item in results)
    cache_write_tokens = sum(item.cache_write_tokens for item in results)
    cache_read_tokens = sum(item.cache_read_tokens for item in results)
    output_tokens = sum(item.output_tokens for item in results)

    lines = [
        "miniF2F-rocq report (test split)",
        f"Problems evaluated : {total}",
        (
            f"Theorems proved    : {len(proved)} / {total}"
            + (f" ({100 * len(proved) / total:.1f}%)" if total else "")
        ),
        f"Open               : {len(open_)}",
        f"Refuted            : {len(refuted)}",
        f"Errors             : {len(errors)}",
        f"Total cost         : ${total_cost:.6f}",
        f"Proved cost        : ${proved_cost:.6f}",
        (
            f"Input tokens       : {input_tokens} "
            f"(cache_write={cache_write_tokens} cache_read={cache_read_tokens})"
        ),
        f"Output tokens      : {output_tokens}",
        "",
        "Theorems proved by per-problem cost interval:",
    ]
    for index, (low, high, label) in enumerate(COST_INTERVALS_USD):
        last = index == len(COST_INTERVALS_USD) - 1
        count = count_proved_in_interval(proved, low, high, right_closed=last)
        lines.append(f"  {label} : {count} theorem(s) proved")
    overflow = count_proved_in_interval(proved, 0.04, float("inf"), right_closed=False)
    lines.append(f"  ($0.04, +∞) : {overflow} theorem(s) proved")

    lines.extend(
        ["", "Cumulative (theorems proved with per-problem cost ≤ threshold):"]
    )
    for threshold in (0.01, 0.02, 0.03, 0.04):
        count = sum(1 for item in proved if item.cost_usd <= threshold)
        lines.append(f"  ≤ ${threshold:.2f} : {count} theorem(s) proved")
    return "\n".join(lines)


def proved_percent_curve(
    results: list[ProblemResult],
) -> tuple[list[float], list[float]]:
    """Return (cost, % proved) points for a step plot over all evaluated problems."""
    total = len(results)
    if total == 0:
        return [0.0], [0.0]
    proved = sorted(
        (item for item in results if item.status == LemmaStatus.Proved.value),
        key=lambda item: item.cost_usd,
    )
    xs = [0.0]
    ys = [0.0]
    for index, item in enumerate(proved, start=1):
        xs.append(item.cost_usd)
        ys.append(100.0 * index / total)
    max_cost = max((item.cost_usd for item in results), default=0.0)
    if xs[-1] < max_cost:
        xs.append(max_cost)
        ys.append(ys[-1])
    return xs, ys


def plot_proved_percent_vs_cost(results: list[ProblemResult], path: Path) -> None:
    """Save a matplotlib plot of cumulative % proved vs per-problem cost."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs, ys = proved_percent_curve(results)
    figure, axes = plt.subplots()
    axes.step(xs, ys, where="post")
    axes.set_xlabel("Cost (USD)")
    axes.set_ylabel("Theorems proved (%)")
    axes.set_ylim(0, 100)
    axes.set_title("miniF2F-rocq test: proved theorems vs cost")
    axes.grid(True, linestyle=":")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def prove_one(
    name: str,
    goal: Goal,
    *,
    max_cost_usd: float,
    lemma_selector: str = DEFAULT_LEMMA_SELECTOR,
) -> ProblemResult:
    orchestrator = build_openai_orchestrator(
        verbose=True, lemma_selector=lemma_selector
    )
    try:
        node = orchestrator.prove(
            goal,
            max_cost_usd=max_cost_usd,
            max_attempts=None,
        )
        if node.status is LemmaStatus.Proved:
            orchestrator.log_complete_proof(node)
        return result_from_orchestrator(name, orchestrator, node.status.value)
    except Exception:
        traceback.print_exc()
        return result_from_orchestrator(name, orchestrator, "error")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the live orchestrator with an explicit lemma selector "
            "on the miniF2F-rocq test split."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for per-problem logs. By default, use "
            "benchmark_runs/<lemma-selector>."
        ),
    )
    parser.add_argument(
        "--lemma-selector",
        choices=LEMMA_SELECTOR_NAMES,
        default=DEFAULT_LEMMA_SELECTOR,
        help="Lemma selection strategy (default: positive-retry).",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=DEFAULT_MAX_COST_USD,
        help="Per-problem orchestrator USD budget (default: 0.04).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N problems (debug).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="0-based index into the test split.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-run problems even if a completed log already exists.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling problems (default: 42). Pass -1 to disable shuffling.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir or (
        DEFAULT_OUTPUT_ROOT / args.lemma_selector.replace("-", "_")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    by_split = load_minif2f(splits=["test"])
    if "test" not in by_split:
        print(
            f"Split 'test' not found. Available splits: {list(by_split)}",
            file=sys.stderr,
        )
        return 1
    problems = list(by_split["test"])
    if args.seed >= 0:
        random.seed(args.seed)
        random.shuffle(problems)
    problems = problems[args.start :]
    if args.limit is not None:
        problems = problems[: args.limit]
    if not problems:
        print("No test problems to evaluate.", file=sys.stderr)
        return 1

    completed = {} if args.no_resume else load_completed_results(output_dir)
    results: list[ProblemResult] = []

    for index, problem in enumerate(problems, start=1):
        log_path = output_dir / f"{problem.name}.log"
        if problem.name in completed:
            result = completed[problem.name]
            results.append(result)
            print(
                f"[eval] skip {index}/{len(problems)} {problem.name} "
                f"({result.status}, ${result.cost_usd:.6f})",
                flush=True,
            )
            continue

        print(
            f"[eval] {index}/{len(problems)} {problem.name}",
            flush=True,
        )
        with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
            with tee_stdout(log_file):
                print(
                    f"[eval] config lemma_selector={args.lemma_selector} "
                    f"max_cost_usd={args.max_cost} seed={args.seed}"
                )
                result = prove_one(
                    problem.name,
                    problem.to_goal(),
                    max_cost_usd=args.max_cost,
                    lemma_selector=args.lemma_selector,
                )
        results.append(result)
        proved_so_far = sum(1 for r in results if r.status == LemmaStatus.Proved.value)
        pct = 100.0 * proved_so_far / len(results)
        print(
            f"[eval] saved {log_path} status={result.status} "
            f"cost=${result.cost_usd:.6f} attempts={result.attempts} "
            f"in={result.input_tokens} "
            f"cache_write={result.cache_write_tokens} "
            f"cache_read={result.cache_read_tokens} "
            f"out={result.output_tokens} "
            f"| proved {proved_so_far}/{len(results)} ({pct:.1f}%)",
            flush=True,
        )

    report = format_report(results)
    report_path = output_dir / "report.txt"
    report_path.write_text(report + "\n", encoding="utf-8")
    print()
    print(report)

    plot_path = output_dir / "proved_vs_cost.png"
    try:
        plot_proved_percent_vs_cost(results, plot_path)
        print(f"Plot written to {plot_path.resolve()}")
    except ImportError:
        print(
            "matplotlib is not installed; skip plot. "
            "Install with: pip install 'llmprover[bench]'",
            file=sys.stderr,
        )
    print(f"Logs written to {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
