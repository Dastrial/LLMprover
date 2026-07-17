"""Batch Rocq checking via ``coqc`` — used for oneshot LLM proofs."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from llmprover.domain import CoqcResult, ProofAttempt
from llmprover.proof_script import ProofScript


class CoqcBackend:
    """Compile a full ``.v`` script through ``coqc``."""

    def __init__(self, coqc_path: str = "coqc", timeout: float = 30.0) -> None:
        self.coqc_path = coqc_path
        self.timeout = timeout

    def check_script(self, source: ProofScript) -> CoqcResult:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "check.v"
            path.write_text(source.code, encoding="utf-8")

            try:
                proc = subprocess.run(
                    [self.coqc_path, str(path)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return CoqcResult(
                    success=False,
                    stderr=f"coqc timed out after {self.timeout}s",
                )

            return CoqcResult(
                success=proc.returncode == 0,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )

    def make_script(self, attempt: ProofAttempt) -> ProofScript:
        """Make a ProofScript from a proof attempt that may contain admitted lemmas.

        The main lemma statement follows ``attempt.polarity`` (``P`` or ``~ (P)``).
        Helper lemmas in ``new_lemmas`` keep their own canonical statements.
        """
        code = ""
        for lemma in attempt.new_lemmas:
            code += f"Lemma {lemma.name}: {lemma.statement}.\n Proof.\n admit.\n Admitted.\n\n"
        code += f"Lemma {attempt.goal.name}: {attempt.target_statement}.\n"
        code += f"Proof.\n{attempt.script}\nQed."
        return ProofScript(code=code)

    def check_attempt(self, attempt: ProofAttempt) -> CoqcResult:
        """Check a proof attempt.

        Builds a full script via ``make_script`` and runs ``coqc``. On success,
        compilation is necessary but not sufficient for decomposition: the main
        goal's assumptions should be exactly the lemmas in ``attempt.new_lemmas``
        (no stdlib lemmas, axioms, or undeclared admits). That assumption check
        is planned but not implemented yet.
        """
        return self.check_script(self.make_script(attempt))
