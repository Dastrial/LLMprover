"""Batch Rocq checking via ``coqc`` — used for oneshot LLM proofs."""

from __future__ import annotations

import re
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

from llmprover.rocq.coqc_output import (
    CoqcResult,
    parse_coqc_locations,
    parse_print_assumptions,
    remap_coqc_stderr,
)
from llmprover.domain import ProofAttempt, RocqEnvironment
from llmprover.rocq.helper_resolution import resolve_attempt_helpers
from llmprover.rocq.proof_script import ProofScript, shift_tactic_region, tactic_line_span

# Module wrapper so generated lemmas / admitted helpers get unambiguous
# qualified names (e.g. ``LLMProver.helper`` instead of a bare ``helper``).
CHECK_MODULE = "LLMProver"

STMT_EQ_LEMMA = "__llmprover_stmt_eq"
STMT_EQ_LHS = "__llmprover_lhs"
STMT_EQ_RHS = "__llmprover_rhs"


def prepend_tactic_prelude_if_needed(code: str) -> str:
    """Prepend the minimal Rocq imports required by the tactics found in *code*.

    Inspects *code* for the presence of tactics that are not available by
    default in the Rocq standard library (``lia``, ``nia``, ``lra``, ``nra``,
    ``ring``, ``ring_simplify``, ``field``, ``field_simplify``) and prepends
    only the ``Require Import`` lines that are actually needed.

    ``auto``, ``eauto``, ``firstorder``, and ``congruence`` are built-in and
    require no import.

    When none of the guarded tactics appear in *code*, the string is returned
    unchanged (preserves coqc error line numbers for tests).
    """
    lower = code.lower()

    # `Lia` covers both `lia` (linear integer/nat) and `nia` (nonlinear).
    # `Lra` covers both `lra` (linear reals) and `nra` (nonlinear reals).
    need_int_arith = bool(re.search(r"\blia\b", lower) or re.search(r"\bnia\b", lower))
    need_real_arith = bool(re.search(r"\blra\b", lower) or re.search(r"\bnra\b", lower))
    need_ring = bool(
        re.search(r"\bring_simplify\b", lower) or re.search(r"\bring\b", lower)
    )
    need_field = bool(
        re.search(r"\bfield_simplify\b", lower) or re.search(r"\bfield\b", lower)
    )

    if not (need_int_arith or need_real_arith or need_ring or need_field):
        return code

    parts: list[str] = []

    if need_int_arith:
        parts.append("From Stdlib Require Import Lia.")
    if need_real_arith:
        parts.append("From Stdlib Require Import Lra.")
    if need_ring:
        parts.append("From Stdlib Require Import Ring.")
    if need_field:
        parts.append("From Stdlib Require Import Field.")

    parts.append("")
    return "\n".join(parts) + code


class CoqcBackend:
    """Compile a full ``.v`` script through ``coqc``."""

    def __init__(
        self,
        coqc_path: str = "coqc",
        timeout: float = 30.0,
        *,
        verbose: bool = False,
    ) -> None:
        self.coqc_path = coqc_path
        self.timeout = timeout
        self.verbose = verbose

    def check_script(
        self,
        source: ProofScript,
        *,
        tactic_prelude: bool = True,
        verbose: bool | None = None,
    ) -> CoqcResult:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "check.v"
            if tactic_prelude:
                code = prepend_tactic_prelude_if_needed(source.code)
                prefix_len = len(code) - len(source.code)
                prefix = code[:prefix_len]
                tactic_start, tactic_end = shift_tactic_region(source, prefix)
            else:
                code = source.code
                tactic_start = source.tactic_start_line
                tactic_end = source.tactic_end_line
            log = self.verbose if verbose is None else verbose
            if log:
                print("[coqc] script:", flush=True)
                print(code, end="" if code.endswith("\n") else "\n", flush=True)
            path.write_text(code, encoding="utf-8")

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

            locations = parse_coqc_locations(
                proc.stderr,
                tactic_start_line=tactic_start,
                tactic_end_line=tactic_end,
            )
            script_stderr = None
            if tactic_start is not None and locations:
                script_stderr = remap_coqc_stderr(proc.stderr, locations)
            result = CoqcResult(
                success=proc.returncode == 0,
                stdout=proc.stdout,
                stderr=proc.stderr,
                locations=locations,
                script_stderr=script_stderr,
            )
            if log and not result.success and result.display_stderr.strip():
                print("[coqc] error:", flush=True)
                print(
                    result.display_stderr,
                    end="" if result.display_stderr.endswith("\n") else "\n",
                    flush=True,
                )
            return result

    def statements_equivalent(
        self,
        left: str,
        right: str,
        environment: RocqEnvironment | None = None,
    ) -> bool:
        """Return whether *left* and *right* are convertible (``reflexivity``).

        Identical canonical forms skip ``coqc``. Otherwise compiles a script
        that defines both statements and proves they are equal by
        ``reflexivity`` — kernel convertibility, including normalization.
        """
        lhs = left.strip().rstrip(".").strip()
        rhs = right.strip().rstrip(".").strip()
        if lhs == rhs:
            return True
        if not lhs or not rhs:
            return False

        header = ""
        if environment is not None:
            header = environment.header.strip()
        parts: list[str] = []
        if header:
            parts.append(header)
            parts.append("")
        parts.append(f"Definition {STMT_EQ_LHS} := {lhs}.")
        parts.append(f"Definition {STMT_EQ_RHS} := {rhs}.")
        parts.append(f"Lemma {STMT_EQ_LEMMA} : {STMT_EQ_LHS} = {STMT_EQ_RHS}.")
        parts.append("Proof.")
        parts.append("reflexivity.")
        parts.append("Qed.")
        parts.append("")
        return self.check_script(
            ProofScript(code="\n".join(parts)), verbose=False
        ).success

    def probe_names_under_header(
        self, header: str, names: Sequence[str]
    ) -> frozenset[str]:
        """Return assumption names usable after compiling *header*.

        For each name, compiles ``header`` + ``Check @name.`` with ``coqc``
        (same tool as proof checking). ``@`` disables implicit-argument
        inference so the probe only checks that the global is resolvable.
        """
        available: set[str] = set()
        prelude = header.strip()
        for name in sorted(set(names)):
            parts: list[str] = []
            if prelude:
                parts.append(prelude)
                parts.append("")
            parts.append(f"Check @{name}.")
            parts.append("")
            if self.check_script(
                ProofScript(code="\n".join(parts)),
                tactic_prelude=False,
                verbose=False,
            ).success:
                available.add(name)
        return frozenset(available)

    def make_script(
        self,
        attempt: ProofAttempt,
        *,
        check_assumptions: bool = False,
        helper_names: Sequence[str] | None = None,
    ) -> ProofScript:
        """Make a ProofScript from a proof attempt that may contain admitted lemmas.

        Emits ``goal.environment.header`` first (when non-empty), then helper
        lemmas as ``Admitted``, then the main lemma. The main lemma statement
        follows ``attempt.polarity`` (``P`` or ``~ (P)``). Helpers keep their
        own canonical statements. ``{helper_name}`` placeholders in the tactic
        script are replaced by *helper_names* (default: proposed lemma names).

        When *check_assumptions* is true, keeps the header outside
        ``CHECK_MODULE``, wraps lemmas in the module (fully qualified local
        admits), and appends ``Print Assumptions`` for ``check_attempt``.

        The returned ``ProofScript`` records ``tactic_start_line`` /
        ``tactic_end_line`` (1-based, inclusive) for the LLM tactic body so
        coqc error loci can be remapped into ``attempt.script``.
        """
        attempt = resolve_attempt_helpers(attempt, helper_names)
        body_prefix = ""
        for lemma in attempt.new_lemmas:
            body_prefix += (
                f"Lemma {lemma.name}: {lemma.statement}.\n"
                " Proof.\n admit.\n Admitted.\n\n"
            )
        body_prefix += f"Lemma {attempt.goal.name}: {attempt.target_statement}.\n"
        body_prefix += "Proof.\n"
        body = f"{body_prefix}{attempt.script}\nQed."
        tactic_start, tactic_end = tactic_line_span(body_prefix, attempt.script)

        header = attempt.goal.environment.header.strip()
        if not check_assumptions:
            if header:
                prefix = f"{header}\n\n"
                code = f"{prefix}{body}"
                shift = prefix.count("\n")
                return ProofScript(
                    code=code,
                    tactic_start_line=tactic_start + shift,
                    tactic_end_line=tactic_end + shift,
                )
            return ProofScript(
                code=body,
                tactic_start_line=tactic_start,
                tactic_end_line=tactic_end,
            )

        parts: list[str] = []
        if header:
            parts.append(header)
            parts.append("")
        parts.append(f"Module {CHECK_MODULE}.")
        parts.append(body)
        parts.append(f"End {CHECK_MODULE}.")
        parts.append(f"Print Assumptions {CHECK_MODULE}.{attempt.goal.name}.")
        parts.append("")
        prefix_parts: list[str] = []
        if header:
            prefix_parts.append(header)
            prefix_parts.append("")
        prefix_parts.append(f"Module {CHECK_MODULE}.")
        prefix = "\n".join(prefix_parts) + "\n"
        shift = prefix.count("\n")
        return ProofScript(
            code="\n".join(parts),
            tactic_start_line=tactic_start + shift,
            tactic_end_line=tactic_end + shift,
        )

    def check_attempt(
        self,
        attempt: ProofAttempt,
        *,
        check_assumptions: bool = True,
        helper_names: Sequence[str] | None = None,
    ) -> CoqcResult:
        """Check a proof attempt: compile, then optionally validate assumptions.

        *helper_names* binds ``{helper}`` placeholders (and admitted lemma
        names) to the real identifiers that will appear in the search tree.
        Defaults to the proposed ``new_lemmas`` names.

        When *check_assumptions* is false, only runs ``coqc`` on the plain
        script (no ``Print Assumptions`` / header probe).

        When true (default), allowed logical dependencies are:

        - the resolved helpers (as ``LLMProver.<name>``);
        - ``attempt.goal.environment.allowed_axioms`` (optional explicit extras);
        - any other name reported by ``Print Assumptions`` that is usable after
          ``goal.environment.header`` (``Check @`` probe via ``coqc``).

        Compilation success alone is not enough in that mode: any remaining
        assumption fails the check.
        """
        attempt = resolve_attempt_helpers(attempt, helper_names)
        result = self.check_script(
            self.make_script(attempt, check_assumptions=check_assumptions)
        )
        if not result.success or not check_assumptions:
            return result

        found = parse_print_assumptions(result.output)
        allowed = {
            f"{CHECK_MODULE}.{lemma.name}" for lemma in attempt.new_lemmas
        } | set(attempt.goal.environment.allowed_axioms)

        need_probe = sorted(found - allowed)
        if need_probe:
            available = self.probe_names_under_header(
                attempt.goal.environment.header,
                need_probe,
            )
            forbidden = sorted(set(need_probe) - set(available))
            if forbidden:
                names = ", ".join(forbidden)
                return CoqcResult(
                    success=False,
                    stdout=result.stdout,
                    stderr=(
                        "Disallowed assumptions "
                        f"(not in new_lemmas, allowed_axioms, or header): {names}"
                    ),
                    locations=result.locations,
                    script_stderr=result.script_stderr,
                )
        return result
