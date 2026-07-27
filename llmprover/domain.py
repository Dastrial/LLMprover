"""Domain types for proof search: goals, attempts, and lemma nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Polarity(Enum):
    """Whether an attempt targets ``P`` or ``~P`` for a canonical goal statement ``P``."""

    Positive = "positive"
    Negative = "negative"


class LemmaStatus(Enum):
    """Outcome of search on a lemma node."""

    Open = "open"
    Proved = "proved"
    Refuted = "refuted"


def statement_for_polarity(statement: str, polarity: Polarity) -> str:
    """Return the Rocq statement to prove for *polarity*.

    Positive polarity keeps *statement* (``P``). Negative polarity wraps it as
    ``~ (P)`` (v1 encoding).
    """
    if polarity is Polarity.Positive:
        return statement
    return f"~ ({statement})"


@dataclass
class CoqcResult:
    """Outcome of running ``coqc`` on a script."""

    success: bool
    stdout: str = ""
    stderr: str = ""

    @property
    def output(self) -> str:
        """Full coqc output (stdout + stderr) — pass this back to the LLM on failure."""
        return "\n".join(part for part in (self.stdout, self.stderr) if part).strip()

    def __str__(self) -> str:
        return f"CoqcResult(success={self.success}, stdout={self.stdout}, stderr={self.stderr})"


@dataclass
class Goal:
    """A Rocq statement to be proved and its name.

    The statement is always the canonical formula ``P``. Attempts that attack
    ``~P`` keep this same ``Goal`` and set ``ProofAttempt.polarity`` to
    ``Negative``.
    """

    statement: str
    name: str


@dataclass
class ProofAttempt:
    """A proposed proof for ``goal`` at a given polarity.

    ``goal`` is the canonical lemma (``P``). ``polarity`` selects whether the
    script is meant to prove ``P`` or ``~P``.

    ``script`` is the tactic body (between ``Proof.`` and ``Qed.``).
    ``new_lemmas`` lists sub-goals introduced by decomposition; ``make_script``
    emits them as ``Admitted`` before the main lemma. For sound decomposition,
    the main proof must depend only on those lemmas — to be enforced in
    ``CoqcBackend.check_attempt`` (not yet).
    """

    goal: Goal
    polarity: Polarity = Polarity.Positive
    script: str = field(kw_only=True)
    new_lemmas: list[Goal] = field(kw_only=True)

    @property
    def target_statement(self) -> str:
        """Rocq statement actually attacked by this attempt (``P`` or ``~ (P)``)."""
        return statement_for_polarity(self.goal.statement, self.polarity)


@dataclass
class AttemptRecord:
    """A checked proof attempt, with Rocq outcome and optional child searches.

    ``lemmas`` holds a ``LemmaNode`` per helper introduced by decomposition
    (nested search state for each child goal).
    """

    attempt: ProofAttempt
    rocq_error: CoqcResult
    lemmas: list[LemmaNode] = field(default_factory=list)

    @property
    def fully_succeeded(self) -> bool:
        """Whether this attempt is a complete success.

        Requires a successful ``coqc`` run and every child lemma to be
        ``Proved``. Child status is recursive (nested ``LemmaNode`` trees);
        the search tree must stay acyclic.
        """
        if not self.rocq_error.success:
            return False
        return all(lemma.status is LemmaStatus.Proved for lemma in self.lemmas)


def status_from_frontiers(
    latest_positive: AttemptRecord | None,
    latest_negative: AttemptRecord | None,
) -> LemmaStatus:
    """Derive lemma status from the latest attempt of each polarity.

    For now we only look at the frontier (last attempt) of each polarity —
    simpler and enough in practice, since we normally stop once a polarity
    succeeds. Checking whether *any* past attempt succeeded might be more
    robust later.

    A frontier attempt only counts if ``fully_succeeded`` (coqc ok and all
    child lemmas proved), so status walks the nested lemma tree.
    """
    if latest_positive is not None and latest_positive.fully_succeeded:
        return LemmaStatus.Proved
    if latest_negative is not None and latest_negative.fully_succeeded:
        return LemmaStatus.Refuted
    return LemmaStatus.Open


@dataclass
class LemmaNode:
    """Search state for one lemma: canonical goal plus dual attempt histories.

    Positive and negative attempts share the same ``goal`` (``P``). Status is
    derived from the latest attempt of each polarity (search frontier), and
    only once that attempt's child lemmas are themselves proved.
    Child searches live on ``AttemptRecord.lemmas`` as nested ``LemmaNode``s.
    """

    goal: Goal
    positive: list[AttemptRecord] = field(default_factory=list)
    negative: list[AttemptRecord] = field(default_factory=list)

    def history(self, polarity: Polarity) -> list[AttemptRecord]:
        if polarity is Polarity.Positive:
            return self.positive
        return self.negative

    def frontier(self, polarity: Polarity) -> AttemptRecord | None:
        records = self.history(polarity)
        return records[-1] if records else None

    def append(self, record: AttemptRecord) -> None:
        """Append *record* to the history matching ``record.attempt.polarity``."""
        if record.attempt.goal != self.goal:
            raise ValueError(
                f"Attempt goal {record.attempt.goal!r} does not match "
                f"lemma node goal {self.goal!r}"
            )
        self.history(record.attempt.polarity).append(record)

    @property
    def status(self) -> LemmaStatus:
        return status_from_frontiers(
            self.frontier(Polarity.Positive),
            self.frontier(Polarity.Negative),
        )
