"""Per-policy views of attempt history, with presenter-local memory.

Raw ``AttemptRecord`` lists stay on ``LemmaNode``. Each ``HistoryPresenter``
holds its own per-node append-only text so the same work is not redone when
agents are recreated (via a shared presenter instance injected at construction
time).

The rendered history is a single growing string: legend first, then attempts
from both polarities appended as they appear (no per-polarity sections). That
keeps a stable cacheable prefix across calls.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from llmprover.domain import AttemptRecord, LemmaNode
from llmprover.llm.client import TokenUsage
from llmprover.utils import EMPTY_ATTEMPTS, format_attempt

REPAIR_FORMAT_LEGEND = """\
Previous failed attempts are shown below (both polarities, in order).
Each attempt uses blank lines between fields:
Attempt N:

Statement: <formula that was attacked>

Script:
N | <tactics that were tried>

Rocq errors: <coqc output>"""

FULL_FORMAT_LEGEND = """\
Previous failed attempts are shown below (direct proofs and decompositions, both polarities, in order).
Each attempt uses blank lines between top-level fields; helper lemmas are indented:
Attempt N:

Statement: <formula that was attacked>

Helper lemmas:
  LemmaName [status]: statement
  (present only for decomposition attempts; status is open, proved, or refuted)

Script:
N | <tactics that were tried>

Rocq errors: <coqc output>"""

STRATEGY_FORMAT_LEGEND = """\
Previous failed attempts are shown below (both polarities, in order).
Each attempt uses blank lines between top-level fields; helper lemmas are indented:
Attempt N:

Statement: <formula that was attacked>

Agent: <prover type, model, reasoning tokens>

Helper lemmas:
  LemmaName [status]: statement
  (present only for decomposition attempts; status is open, proved, or refuted)

Script:
N | <tactics that were tried>

Rocq errors: <coqc output>"""


@dataclass
class CacheEntry:
    """Append-only history text for one lemma node."""

    text: str
    covered_positive: int = 0
    covered_negative: int = 0
    next_index: int = 1


class HistoryPresenter(ABC):
    """Turn attempt records into a prompt string, with per-node memory."""

    def __init__(self, policy_id: str, *, verbose: bool = False) -> None:
        self.policy_id = policy_id
        self.verbose = verbose
        self.memory: dict[int, CacheEntry] = {}

    def log_render(self, index: int, text: str) -> None:
        """Print the rendered chunk when ``verbose`` is enabled."""
        if not self.verbose:
            return
        print(
            f"[orchestrator]   history summary ({self.policy_id}, attempt {index}):",
            flush=True,
        )
        body = text.rstrip()
        if not body:
            print("[orchestrator]     (empty)", flush=True)
            return
        for line in body.splitlines():
            print(f"[orchestrator]     {line}", flush=True)

    def present_prompt_block(self, node: LemmaNode) -> tuple[str, TokenUsage]:
        """Return the append-only history block for *node*.

        Seeds the cache with ``format_legend()``, then for each polarity appends
        ``render`` of every attempt past ``covered_*``. Empty render chunks are
        skipped (no text, no attempt index).
        """
        if not node.positive and not node.negative:
            return EMPTY_ATTEMPTS, TokenUsage()

        key = id(node)
        entry = self.memory.get(key)
        if entry is None:
            legend = self.format_legend().rstrip()
            entry = CacheEntry(text=f"{legend}\n\n" if legend else "")
            self.memory[key] = entry

        usage = TokenUsage()
        for records, attr in (
            (node.positive, "covered_positive"),
            (node.negative, "covered_negative"),
        ):
            covered = getattr(entry, attr)
            while covered < len(records):
                record = records[covered]
                covered += 1
                chunk, one_usage = self.render(record, index=entry.next_index)
                usage += one_usage
                if not chunk:
                    continue
                self.log_render(entry.next_index, chunk)
                entry.text += chunk
                entry.next_index += 1
            setattr(entry, attr, covered)

        if entry.next_index == 1:
            return EMPTY_ATTEMPTS, usage
        return entry.text, usage

    @abstractmethod
    def format_legend(self) -> str:
        """Explain how each attempt block is structured in the prompt."""

    @abstractmethod
    def render(
        self,
        record: AttemptRecord,
        *,
        index: int,
    ) -> tuple[str, TokenUsage]:
        """Build the prompt text for one *record* (``Attempt {index}:`` …).

        Return an empty string to skip the record (e.g. repair ignoring
        decompositions).
        """


class NullHistoryPresenter(HistoryPresenter):
    """Always returns the empty-history placeholder; ignores records."""

    def __init__(self) -> None:
        super().__init__(policy_id="null")

    def format_legend(self) -> str:
        return ""

    def render(
        self,
        record: AttemptRecord,
        *,
        index: int,
    ) -> tuple[str, TokenUsage]:
        return "", TokenUsage()

    def present_prompt_block(self, node: LemmaNode) -> tuple[str, TokenUsage]:
        return EMPTY_ATTEMPTS, TokenUsage()


class DeterministicHistoryPresenter(HistoryPresenter):
    """Format attempts with ``format_attempt`` (no LLM)."""

    def __init__(
        self,
        policy_id: str,
        *,
        include_agent: bool = False,
        direct_only: bool = False,
        format_legend_text: str,
        verbose: bool = False,
    ) -> None:
        super().__init__(policy_id, verbose=verbose)
        self.include_agent = include_agent
        self.direct_only = direct_only
        self.format_legend_text = format_legend_text

    def format_legend(self) -> str:
        return self.format_legend_text

    def render(
        self,
        record: AttemptRecord,
        *,
        index: int,
    ) -> tuple[str, TokenUsage]:
        if self.direct_only and record.attempt.new_lemmas:
            return "", TokenUsage()
        return (
            format_attempt(record, index, include_agent=self.include_agent),
            TokenUsage(),
        )

    @classmethod
    def full(cls) -> DeterministicHistoryPresenter:
        """Full history, no agent line (decomposition agents)."""
        return cls(
            policy_id="full",
            format_legend_text=FULL_FORMAT_LEGEND,
        )

    @classmethod
    def repair(cls) -> DeterministicHistoryPresenter:
        """Direct attempts only (repair agents)."""
        return cls(
            policy_id="repair",
            format_legend_text=REPAIR_FORMAT_LEGEND,
            direct_only=True,
        )

    @classmethod
    def strategy(cls) -> DeterministicHistoryPresenter:
        """Full history with agent descriptions (attempt strategy)."""
        return cls(
            policy_id="strategy",
            include_agent=True,
            format_legend_text=STRATEGY_FORMAT_LEGEND,
        )
