"""Dataclass for a proof tree. This tree represents the advancement of the proof search."""

from __future__ import annotations

from dataclasses import dataclass, field

from llmprover.proof_script import ProofScript


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
    """A Rocq statement to be proved and its name."""

    statement: str
    name: str


@dataclass
class ProofAttempt:
    """A proposed proof for ``goal``.

    ``script`` is the tactic body (between ``Proof.`` and ``Qed.``).
    ``new_lemmas`` lists sub-goals introduced by decomposition; ``make_script``
    emits them as ``Admitted`` before the main lemma. For sound decomposition,
    the main proof must depend only on those lemmas — to be enforced in
    ``CoqcBackend.check_attempt`` (not yet).
    """

    goal: Goal
    script: str
    new_lemmas: list[Goal]


@dataclass
class ProofContext:
    """The context of a Goal."""

    goal: Goal
    previous_attempts: list[ProofAttempt]
    rocq_errors: list[CoqcResult] = field(default_factory=list)
    proved_lemmas: list[tuple[Goal, ProofScript]] = field(default_factory=list)
    refuted_lemmas: list[Goal] = field(default_factory=list)


class ProofNode:
    """The content of a node in the proof tree."""

    def __init__(
        self,
        goal: Goal,
        proof_attempts: list[ProofAttempt] | None = None,
        current_attempt: ProofAttempt | None = None,
    ) -> None:
        self.goal = goal
        self.proof_attempts = proof_attempts if proof_attempts is not None else []
        self.current_attempt = current_attempt


class ProofTree:
    def __init__(
        self, root: ProofNode, children: list[ProofNode] | None = None
    ) -> None:
        self.root = root
        new_lemmas = (
            root.current_attempt.new_lemmas if root.current_attempt is not None else []
        )
        if children is None:
            children = [ProofNode(goal=lemma) for lemma in new_lemmas]
        else:
            actual_goals = [child.goal for child in children]
            if actual_goals != new_lemmas:
                raise ValueError(
                    "Child goals must match new_lemmas from root proof_attempts: "
                    f"expected {[goal.name for goal in new_lemmas]!r}, "
                    f"got {[goal.name for goal in actual_goals]!r}"
                )
        self.children = children
