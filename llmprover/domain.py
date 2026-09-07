"""Domain types for proof search: goals, attempts, and lemma nodes."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias

from llmprover.coqc_output import CoqcResult


class Polarity(Enum):
    """Whether an attempt targets ``P`` or ``~P`` for a canonical goal statement ``P``."""

    Positive = "positive"
    Negative = "negative"


PositionStep: TypeAlias = tuple[Polarity, int]
Position: TypeAlias = tuple[PositionStep, ...]


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


@dataclass(frozen=True)
class RocqEnvironment:
    """Compilation context and assumption policy for a goal.

    ``header`` is the full Rocq prelude emitted before lemmas (``Require``,
    scopes, local definitions, axioms, …). ``allowed_axioms`` is an optional
    extra allowlist (fully qualified names as Rocq prints them). By default
    ``check_attempt`` also accepts assumptions that are usable after
    ``header`` (probed with ``coqc`` via ``Check @``).
    """

    header: str = ""
    allowed_axioms: frozenset[str] = frozenset()


@dataclass
class Goal:
    """A Rocq statement to be proved and its name.

    The statement is always the canonical formula ``P``. Attempts that attack
    ``~P`` keep this same ``Goal`` and set ``ProofAttempt.polarity`` to
    ``Negative``.

    ``environment`` is shared with child lemmas created during decomposition.
    """

    statement: str
    name: str
    environment: RocqEnvironment = field(default_factory=RocqEnvironment)


@dataclass
class ProofAttempt:
    """A proposed proof for ``goal`` at a given polarity.

    ``goal`` is the canonical lemma (``P``). ``polarity`` selects whether the
    script is meant to prove ``P`` or ``~P``.

    ``script`` is the tactic body (between ``Proof.`` and ``Qed.``).
    Decomposition scripts refer to proposed helpers as ``{helper_name}``
    placeholders so checking and proof assembly can bind them to the real
    ``LemmaNode`` names. ``new_lemmas`` lists sub-goals introduced by
    decomposition; ``make_script`` emits them as ``Admitted`` before the main
    lemma. For sound decomposition, the main proof must depend only on those
    lemmas, on names allowed by ``goal.environment.header``, or on
    ``goal.environment.allowed_axioms``.

    ``agent`` is the prover class name that produced this attempt. The
    orchestrator stores ``type(prover).__name__`` so later strategy can count
    calls per agent on this lemma. It is shown in strategy histories, not in
    ProverAgent prompts.

    ``about_lemmas`` is set by About-augmented agents to the lemma names
    queried via Rocq ``About`` before the proof call (``None`` otherwise).

    ``search_about_output`` / ``select_output`` hold the raw LLM replies for the
    Search/About and optional select turns (``None`` when unused).
    """

    goal: Goal
    polarity: Polarity = Polarity.Positive
    script: str = field(kw_only=True)
    new_lemmas: list[Goal] = field(kw_only=True)
    agent: str = field(default="", kw_only=True)
    about_lemmas: list[str] | None = field(default=None, kw_only=True)
    search_about_output: str | None = field(default=None, kw_only=True)
    select_output: str | None = field(default=None, kw_only=True)

    @property
    def target_statement(self) -> str:
        """Rocq statement actually attacked by this attempt (``P`` or ``~ (P)``)."""
        return statement_for_polarity(self.goal.statement, self.polarity)


@dataclass
class AttemptRecord:
    """A checked proof attempt, with Rocq outcome and optional child searches.

    ``lemmas`` holds a ``LemmaNode`` per helper introduced by decomposition
    (nested search state for each child goal). An entry may be an existing
    node already in the tree when two statements are equivalent up to
    reflexivity and there is no ancestor relation.
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
    """Derive lemma status from the frontier attempt of each polarity.

    Status is derived from each polarity's frontier: the latest Rocq-successful
    attempt when one exists, otherwise the latest chronological attempt.

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
    derived from each polarity's frontier (latest Rocq-successful attempt, or
    the latest chronological attempt when none succeeded), and only once that
    attempt's child lemmas are themselves proved.
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
        """Return the attempt whose helper subtree is active for this polarity.

        This is the latest Rocq-successful attempt when one exists, otherwise
        the latest chronological attempt.
        """
        records = self.history(polarity)
        if not records:
            return None
        for record in reversed(records):
            if record.rocq_error.success:
                return record
        return records[-1]

    def append(self, record: AttemptRecord) -> None:
        """Append *record* to the history matching ``record.attempt.polarity``."""
        if record.attempt.goal != self.goal:
            raise ValueError(
                f"Attempt goal {record.attempt.goal!r} does not match "
                f"lemma node goal {self.goal!r}"
            )
        self.history(record.attempt.polarity).append(record)

    def agent_call_counts(self) -> dict[str, int]:
        """Count prover class names on this node's ``AttemptRecord``s.

        Reads ``ProofAttempt.agent`` from both polarities. Empty names are
        ignored.
        """
        counts: dict[str, int] = {}
        for record in (*self.positive, *self.negative):
            name = record.attempt.agent
            if name:
                counts[name] = counts.get(name, 0) + 1
        return counts

    @property
    def status(self) -> LemmaStatus:
        return status_from_frontiers(
            self.frontier(Polarity.Positive),
            self.frontier(Polarity.Negative),
        )

    def from_position(self, position: Position) -> LemmaNode:
        return self.nodes_along_position(position)[-1]

    def nodes_along_position(self, position: Position) -> list[LemmaNode]:
        """Return nodes from this node to *position*, inclusive."""
        nodes = [self]
        current = self
        for polarity, index in position:
            frontier_record = current.frontier(polarity)
            if frontier_record is None:
                raise ValueError(f"No frontier attempt for polarity {polarity.value!r}")
            if index < 0 or index >= len(frontier_record.lemmas):
                raise ValueError(
                    f"Helper index {index} out of range for "
                    f"{len(frontier_record.lemmas)} lemmas"
                )
            current = frontier_record.lemmas[index]
            nodes.append(current)
        return nodes

    def nodes_reaching(self, target: LemmaNode) -> dict[int, LemmaNode]:
        """Return *target* and every node in this tree that can reach it.

        Follows every helper edge, including alias links (the same child
        ``LemmaNode`` stored on several attempts). After a merge, the aliased
        node and all of its ancestors reach the source, even if they are not
        on ``nodes_along_position`` from the root to the source.
        """
        parents: dict[int, list[LemmaNode]] = {}
        seen: set[int] = set()

        def visit(node: LemmaNode) -> None:
            if id(node) in seen:
                return
            seen.add(id(node))
            for record in (*node.positive, *node.negative):
                for child in record.lemmas:
                    parents.setdefault(id(child), []).append(node)
                    visit(child)

        visit(self)
        reaching = {id(target): target}
        stack = [target]
        while stack:
            node = stack.pop()
            for parent in parents.get(id(node), []):
                if id(parent) not in reaching:
                    reaching[id(parent)] = parent
                    stack.append(parent)
        return reaching

    def walk(self) -> Iterator[LemmaNode]:
        """Yield this node and every nested lemma, skipping aliased repeats."""
        seen: set[int] = set()

        def inner(node: LemmaNode) -> Iterator[LemmaNode]:
            if id(node) in seen:
                return
            seen.add(id(node))
            yield node
            for record in (*node.positive, *node.negative):
                for child in record.lemmas:
                    yield from inner(child)

        yield from inner(self)
