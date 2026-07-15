"""Dataclass for a proof tree. This tree represents the advancement of the proof search."""

from __future__ import annotations

from dataclasses import dataclass, field


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
class AttemptRecord:
    """The context of a Goal."""

    attempt: ProofAttempt
    rocq_errors: CoqcResult
    lemmas_attempts: list[list[AttemptRecord]] = field(default_factory=list)


ProofRecord = AttemptRecord
