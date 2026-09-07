"""Tests for llmprover.rocq (CoqcBackend)."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llmprover.coqc_output import OUTSIDE_TACTIC_LOCUS, CoqcLocation, CoqcResult
from llmprover.domain import Goal, Polarity, ProofAttempt, RocqEnvironment
from llmprover.proof_script import ProofScript
from llmprover.rocq import (
    CHECK_MODULE,
    STMT_EQ_LEMMA,
    STMT_EQ_LHS,
    STMT_EQ_RHS,
    CoqcBackend,
    prepend_tactic_prelude_if_needed,
)

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
COQC_AVAILABLE = shutil.which("coqc") is not None

VALID_IMPORT = "From Stdlib Require Import PeanoNat.\n"

VALID_PROOF = """\
From Stdlib Require Import PeanoNat.

Lemma plus_n0 : forall n : nat, n + 0 = n.
Proof.
  induction n as [|n IHn].
  - reflexivity.
  - simpl. rewrite IHn. reflexivity.
Qed.
"""

INVALID_PROOF = """\
Lemma bad : True.
Proof.
  oops.
Qed.
"""

INVALID_PROOF_ERROR_RE = re.compile(
    r'^File "[^"]+", line 3, characters 2-6:\n'
    r"Error: The reference oops was not found in the current environment\.\Z"
)

SLOW_CHECK_TIMEOUT = 0.2


# --- prepend_tactic_prelude_if_needed ---


def test_prepend_tactic_prelude_unchanged_without_guarded_tactics() -> None:
    code = "Lemma t: True.\nProof.\nauto.\nQed.\n"
    assert prepend_tactic_prelude_if_needed(code) == code


def test_prepend_tactic_prelude_for_lia() -> None:
    code = "Lemma t: True.\nProof.\nlia.\nQed.\n"
    assert prepend_tactic_prelude_if_needed(code) == (
        "From Stdlib Require Import Lia.\nLemma t: True.\nProof.\nlia.\nQed.\n"
    )


def test_prepend_tactic_prelude_for_nia() -> None:
    code = "Lemma t: True.\nProof.\nnia.\nQed.\n"
    assert prepend_tactic_prelude_if_needed(code) == (
        "From Stdlib Require Import Lia.\nLemma t: True.\nProof.\nnia.\nQed.\n"
    )


def test_prepend_tactic_prelude_for_lra() -> None:
    code = "Lemma t: True.\nProof.\nlra.\nQed.\n"
    assert prepend_tactic_prelude_if_needed(code) == (
        "From Stdlib Require Import Lra.\nLemma t: True.\nProof.\nlra.\nQed.\n"
    )


def test_prepend_tactic_prelude_for_nra() -> None:
    code = "Lemma t: True.\nProof.\nnra.\nQed.\n"
    assert prepend_tactic_prelude_if_needed(code) == (
        "From Stdlib Require Import Lra.\nLemma t: True.\nProof.\nnra.\nQed.\n"
    )


def test_prepend_tactic_prelude_for_ring_simplify() -> None:
    code = "Lemma t: True.\nProof.\nring_simplify.\nQed.\n"
    assert prepend_tactic_prelude_if_needed(code) == (
        "From Stdlib Require Import Ring.\n"
        "Lemma t: True.\nProof.\nring_simplify.\nQed.\n"
    )


def test_prepend_tactic_prelude_for_ring() -> None:
    code = "Lemma t: True.\nProof.\nring.\nQed.\n"
    assert prepend_tactic_prelude_if_needed(code) == (
        "From Stdlib Require Import Ring.\nLemma t: True.\nProof.\nring.\nQed.\n"
    )


def test_prepend_tactic_prelude_for_field_simplify() -> None:
    code = "Lemma t: True.\nProof.\nfield_simplify.\nQed.\n"
    assert prepend_tactic_prelude_if_needed(code) == (
        "From Stdlib Require Import Field.\n"
        "Lemma t: True.\nProof.\nfield_simplify.\nQed.\n"
    )


def test_prepend_tactic_prelude_for_field() -> None:
    code = "Lemma t: True.\nProof.\nfield.\nQed.\n"
    assert prepend_tactic_prelude_if_needed(code) == (
        "From Stdlib Require Import Field.\nLemma t: True.\nProof.\nfield.\nQed.\n"
    )


def test_prepend_tactic_prelude_adds_three_requires() -> None:
    code = "Lemma t: True.\nProof.\nlia.\nlra.\nring.\nQed.\n"
    assert prepend_tactic_prelude_if_needed(code) == (
        "From Stdlib Require Import Lia.\n"
        "From Stdlib Require Import Lra.\n"
        "From Stdlib Require Import Ring.\n"
        "Lemma t: True.\nProof.\nlia.\nlra.\nring.\nQed.\n"
    )


# --- CoqcBackend.check_script ---


@pytest.mark.skipif(not COQC_AVAILABLE, reason="coqc not installed")
def test_check_script_valid_import() -> None:
    backend = CoqcBackend()
    result = backend.check_script(ProofScript(code=VALID_IMPORT))
    assert result.success is True


@pytest.mark.skipif(not COQC_AVAILABLE, reason="coqc not installed")
def test_check_script_valid_proof() -> None:
    backend = CoqcBackend()
    result = backend.check_script(ProofScript(code=VALID_PROOF))
    assert result.success is True


@pytest.mark.skipif(not COQC_AVAILABLE, reason="coqc not installed")
def test_check_script_invalid_proof_returns_error_output() -> None:
    backend = CoqcBackend()
    result = backend.check_script(ProofScript(code=INVALID_PROOF))
    assert result.success is False
    assert INVALID_PROOF_ERROR_RE.match(result.output)


@pytest.mark.skipif(not COQC_AVAILABLE, reason="coqc not installed")
def test_check_script_from_example_file() -> None:
    backend = CoqcBackend()
    script = ProofScript.from_file(EXAMPLES / "plus_n0.v")
    result = backend.check_script(script)
    assert result.success is True


@pytest.mark.skipif(not COQC_AVAILABLE, reason="coqc not installed")
def test_check_script_from_temp_file(tmp_path: Path) -> None:
    file_path = tmp_path / "lemma.v"
    file_path.write_text(VALID_PROOF, encoding="utf-8")

    backend = CoqcBackend()
    script = ProofScript.from_file(file_path)
    result = backend.check_script(script)

    assert script.code == VALID_PROOF
    assert result.success is True


@patch(
    "llmprover.rocq.subprocess.run",
    return_value=MagicMock(
        returncode=1,
        stdout="",
        stderr=(
            'File "/tmp/check.v", line 4, characters 2-6:\n'
            "Error: The reference oops was not found in the current environment.\n"
        ),
    ),
)
def test_check_script_verbose_prints_full_remapped_error(
    mock_run: MagicMock, capsys
) -> None:
    backend = CoqcBackend(verbose=True)
    source = ProofScript(
        code="Lemma t: True.\nProof.\nidtac.\n  oops.\nQed.\n",
        tactic_start_line=3,
        tactic_end_line=4,
    )

    result = backend.check_script(source, tactic_prelude=False)

    assert capsys.readouterr().out == (
        "[coqc] script:\n"
        "Lemma t: True.\n"
        "Proof.\n"
        "idtac.\n"
        "  oops.\n"
        "Qed.\n"
        "[coqc] error:\n"
        "Script line 2, characters 2-6:\n"
        "Error: The reference oops was not found in the current environment.\n"
    )
    assert result.locations == [
        CoqcLocation(line=4, start_char=2, end_char=6, script_line=2)
    ]
    mock_run.assert_called_once()


@patch(
    "llmprover.rocq.subprocess.run",
    return_value=MagicMock(
        returncode=1,
        stdout="",
        stderr=(
            'File "/tmp/check.v", line 5, characters 0-3:\n'
            "Warning: some-warning [some,default]\n"
            'File "/tmp/check.v", line 7, characters 4-8:\n'
            "Error: The reference foo was not found in the current environment.\n"
        ),
    ),
)
def test_check_script_remaps_multiple_loci_including_warning(
    mock_run: MagicMock,
) -> None:
    source = ProofScript(
        code=(
            "Lemma t: True.\nProof.\nidtac.\nidtac.\n   bad.\nidtac.\n    foo.\nQed.\n"
        ),
        tactic_start_line=3,
        tactic_end_line=7,
    )
    result = CoqcBackend().check_script(source, tactic_prelude=False)

    assert result.locations == [
        CoqcLocation(line=5, start_char=0, end_char=3, script_line=3),
        CoqcLocation(line=7, start_char=4, end_char=8, script_line=5),
    ]
    assert result.output == (
        "Script line 3, characters 0-3:\n"
        "Warning: some-warning [some,default]\n"
        "Script line 5, characters 4-8:\n"
        "Error: The reference foo was not found in the current environment."
    )
    mock_run.assert_called_once()


def test_check_script_tactic_prelude_on_prepends_require() -> None:
    received: list[str] = []

    def capture_run(args, **kwargs):  # type: ignore[no-untyped-def]
        import pathlib

        received.append(pathlib.Path(args[1]).read_text(encoding="utf-8"))
        return MagicMock(returncode=0, stdout="", stderr="")

    source = ProofScript(
        code="Lemma t: True.\nProof.\nlia.\nQed.\n",
        tactic_start_line=3,
        tactic_end_line=3,
    )
    with patch("llmprover.rocq.subprocess.run", side_effect=capture_run):
        CoqcBackend().check_script(source, tactic_prelude=True)

    assert received[0] == (
        "From Stdlib Require Import Lia.\nLemma t: True.\nProof.\nlia.\nQed.\n"
    )


def test_check_script_tactic_prelude_off_leaves_code_unchanged() -> None:
    received: list[str] = []

    def capture_run(args, **kwargs):  # type: ignore[no-untyped-def]
        import pathlib

        received.append(pathlib.Path(args[1]).read_text(encoding="utf-8"))
        return MagicMock(returncode=0, stdout="", stderr="")

    source = ProofScript(
        code="Lemma t: True.\nProof.\nlia.\nQed.\n",
        tactic_start_line=3,
        tactic_end_line=3,
    )
    with patch("llmprover.rocq.subprocess.run", side_effect=capture_run):
        CoqcBackend().check_script(source, tactic_prelude=False)

    assert received[0] == "Lemma t: True.\nProof.\nlia.\nQed.\n"


@patch(
    "llmprover.rocq.subprocess.run",
    return_value=MagicMock(returncode=0, stdout="", stderr=""),
)
def test_check_script_verbose_prints_exact_code_sent_to_coqc(
    mock_run: MagicMock, capsys
) -> None:
    backend = CoqcBackend(verbose=True)
    source = ProofScript(code="Lemma t: True.\nProof.\nlia.\nQed.\n")

    backend.check_script(source)

    assert capsys.readouterr().out == (
        "[coqc] script:\n"
        "From Stdlib Require Import Lia.\n"
        "Lemma t: True.\n"
        "Proof.\n"
        "lia.\n"
        "Qed.\n"
    )
    mock_run.assert_called_once()


@patch(
    "llmprover.rocq.subprocess.run",
    return_value=MagicMock(returncode=0, stdout="", stderr=""),
)
def test_check_script_silent_when_not_verbose(mock_run: MagicMock, capsys) -> None:
    backend = CoqcBackend(verbose=False)
    backend.check_script(ProofScript(code=VALID_IMPORT))
    assert capsys.readouterr().out == ""
    mock_run.assert_called_once()


@patch(
    "llmprover.rocq.subprocess.run",
    side_effect=subprocess.TimeoutExpired(cmd=["coqc"], timeout=SLOW_CHECK_TIMEOUT),
)
def test_check_script_timeout(mock_run: MagicMock) -> None:
    backend = CoqcBackend(timeout=SLOW_CHECK_TIMEOUT)
    result = backend.check_script(ProofScript(code=VALID_IMPORT))

    assert result.success is False
    assert result.output == f"coqc timed out after {SLOW_CHECK_TIMEOUT}s"


# --- CoqcBackend.statements_equivalent ---


def test_statements_equivalent_true_for_identical_canonical_form() -> None:
    backend = CoqcBackend()
    backend.check_script = MagicMock()  # type: ignore[method-assign]

    assert backend.statements_equivalent("True.", "True") is True
    backend.check_script.assert_not_called()


def test_statements_equivalent_builds_reflexivity_goal() -> None:
    backend = CoqcBackend()
    backend.check_script = MagicMock(return_value=CoqcResult(success=True))  # type: ignore[method-assign]
    env = RocqEnvironment(header="From Stdlib Require Import PeanoNat.")

    assert backend.statements_equivalent("True", "1 = 1", env) is True

    script = backend.check_script.call_args.args[0]
    assert script.code == (
        "From Stdlib Require Import PeanoNat.\n"
        "\n"
        f"Definition {STMT_EQ_LHS} := True.\n"
        f"Definition {STMT_EQ_RHS} := 1 = 1.\n"
        f"Lemma {STMT_EQ_LEMMA} : {STMT_EQ_LHS} = {STMT_EQ_RHS}.\n"
        "Proof.\n"
        "reflexivity.\n"
        "Qed.\n"
    )


def test_statements_equivalent_both_with_trailing_dot() -> None:
    backend = CoqcBackend()
    backend.check_script = MagicMock()  # type: ignore[method-assign]

    assert backend.statements_equivalent("True.", "True.") is True
    backend.check_script.assert_not_called()


def test_statements_equivalent_both_without_trailing_dot() -> None:
    backend = CoqcBackend()
    backend.check_script = MagicMock()  # type: ignore[method-assign]

    assert backend.statements_equivalent("True", "True") is True
    backend.check_script.assert_not_called()


def test_statements_equivalent_asymmetric_whitespace_and_dots() -> None:
    backend = CoqcBackend()
    backend.check_script = MagicMock()  # type: ignore[method-assign]

    assert backend.statements_equivalent("  True .  ", " True  . ") is True
    backend.check_script.assert_not_called()


def test_statements_equivalent_false_when_coqc_fails() -> None:
    backend = CoqcBackend()
    backend.check_script = MagicMock(return_value=CoqcResult(success=False))  # type: ignore[method-assign]

    assert backend.statements_equivalent("True", "False") is False


@pytest.mark.skipif(not COQC_AVAILABLE, reason="coqc not installed")
def test_statements_equivalent_live_convertibility() -> None:
    backend = CoqcBackend()
    env = RocqEnvironment(header="From Stdlib Require Import PeanoNat.")

    assert backend.statements_equivalent(
        "forall n : nat, n + 0 = n",
        "forall n : nat, n + 0 = n.",
    )
    assert backend.statements_equivalent("1 + 1 = 2", "2 = 2", env)
    assert not backend.statements_equivalent("True", "False")


# --- CoqcBackend.probe_names_under_header ---


@pytest.mark.skipif(not COQC_AVAILABLE, reason="coqc not installed")
def test_probe_names_under_header_resolves_implicits_via_check_at() -> None:
    """Globals with non-inferrable implicits must still probe as available."""
    header = "Require Import Reals.\nOpen Scope R_scope.\n"
    names = (
        "FunctionalExtensionality.functional_extensionality_dep",
        "ClassicalDedekindReals.inexistant",
        "ClassicalDedekindReals.sig_forall_dec",
    )
    expected = (
        "FunctionalExtensionality.functional_extensionality_dep",
        "ClassicalDedekindReals.sig_forall_dec",
    )
    backend = CoqcBackend()

    available = backend.probe_names_under_header(header, names)

    assert available == frozenset(expected)


# --- CoqcBackend.make_script ---


def test_make_script_builds_main_lemma_only() -> None:
    goal = Goal(statement="True", name="trivial")
    attempt = ProofAttempt(goal=goal, script="exact I.", new_lemmas=[])
    backend = CoqcBackend()

    script = backend.make_script(attempt, check_assumptions=False)

    assert script.code == "Lemma trivial: True.\nProof.\nexact I.\nQed."


def test_make_script_includes_header() -> None:
    goal = Goal(
        statement="forall n : nat, n = n",
        name="refl_nat",
        environment=RocqEnvironment(header="From Stdlib Require Import PeanoNat."),
    )
    attempt = ProofAttempt(goal=goal, script="intros n. reflexivity.", new_lemmas=[])
    backend = CoqcBackend()

    script = backend.make_script(attempt, check_assumptions=False)

    assert script.code == (
        "From Stdlib Require Import PeanoNat.\n"
        "\n"
        "Lemma refl_nat: forall n : nat, n = n.\n"
        "Proof.\n"
        "intros n. reflexivity.\n"
        "Qed."
    )


def test_make_script_includes_admitted_lemmas() -> None:
    main = Goal(statement="forall n : nat, n = n", name="main")
    first_helper = Goal(statement="forall n : nat, n = n", name="first_helper")
    second_helper = Goal(statement="forall n : nat, n = n", name="second_helper")
    attempt = ProofAttempt(
        goal=main,
        script="intros n. reflexivity.",
        new_lemmas=[first_helper, second_helper],
    )
    backend = CoqcBackend()

    script = backend.make_script(attempt)

    expected = (
        "Lemma first_helper: forall n : nat, n = n.\n"
        " Proof.\n"
        " admit.\n"
        " Admitted.\n"
        "\n"
        "Lemma second_helper: forall n : nat, n = n.\n"
        " Proof.\n"
        " admit.\n"
        " Admitted.\n"
        "\n"
        "Lemma main: forall n : nat, n = n.\n"
        "Proof.\n"
        "intros n. reflexivity.\n"
        "Qed."
    )
    assert script.code == expected


def test_make_script_substitutes_helper_placeholders() -> None:
    main = Goal(statement="True", name="main")
    helper = Goal(statement="True", name="helper")
    attempt = ProofAttempt(
        goal=main,
        script="exact {helper}.",
        new_lemmas=[helper],
    )
    backend = CoqcBackend()

    script = backend.make_script(attempt, helper_names=["helper_1"])

    assert script.code == (
        "Lemma helper_1: True.\n"
        " Proof.\n"
        " admit.\n"
        " Admitted.\n"
        "\n"
        "Lemma main: True.\n"
        "Proof.\n"
        "exact helper_1.\n"
        "Qed."
    )


def test_make_script_uses_negated_statement_for_negative_polarity() -> None:
    goal = Goal(statement="forall n : nat, n + 0 = n.", name="plus_n0")
    attempt = ProofAttempt(
        goal=goal,
        script="intro H. discriminate.",
        new_lemmas=[],
        polarity=Polarity.Negative,
    )
    backend = CoqcBackend()

    script = backend.make_script(attempt)

    assert script.code == (
        "Lemma plus_n0: ~ (forall n : nat, n + 0 = n.).\n"
        "Proof.\n"
        "intro H. discriminate.\n"
        "Qed."
    )


def test_make_script_negative_polarity_keeps_helper_statements() -> None:
    main = Goal(statement="True", name="main")
    helper = Goal(statement="False", name="helper")
    attempt = ProofAttempt(
        goal=main,
        script="apply helper.",
        new_lemmas=[helper],
        polarity=Polarity.Negative,
    )
    backend = CoqcBackend()

    script = backend.make_script(attempt)

    assert script.code == (
        "Lemma helper: False.\n"
        " Proof.\n"
        " admit.\n"
        " Admitted.\n"
        "\n"
        "Lemma main: ~ (True).\n"
        "Proof.\n"
        "apply helper.\n"
        "Qed."
    )


def test_make_script_check_assumptions_wraps_module_and_prints_assumptions() -> None:
    goal = Goal(statement="True", name="trivial")
    helper = Goal(statement="False", name="helper")
    attempt = ProofAttempt(goal=goal, script="exact I.", new_lemmas=[helper])
    backend = CoqcBackend()

    script = backend.make_script(attempt, check_assumptions=True)

    assert script.code == (
        f"Module {CHECK_MODULE}.\n"
        "Lemma helper: False.\n"
        " Proof.\n"
        " admit.\n"
        " Admitted.\n"
        "\n"
        "Lemma trivial: True.\n"
        "Proof.\n"
        "exact I.\n"
        "Qed.\n"
        f"End {CHECK_MODULE}.\n"
        f"Print Assumptions {CHECK_MODULE}.trivial.\n"
    )


def test_make_script_prefix_collision_helpers_with_header_and_negative_polarity() -> (
    None
):
    inner = Goal(name="helper_1", statement="nat -> nat.")
    outer = Goal(name="helper_1_base", statement="nat.")
    main = Goal(
        name="main",
        statement="True.",
        environment=RocqEnvironment(header="From Stdlib Require Import PeanoNat."),
    )
    attempt = ProofAttempt(
        goal=main,
        script="apply {helper_1_base}. apply {helper_1}.",
        new_lemmas=[inner, outer],
        polarity=Polarity.Negative,
    )

    script = CoqcBackend().make_script(attempt, check_assumptions=True)

    assert script.code == (
        "From Stdlib Require Import PeanoNat.\n"
        "\n"
        f"Module {CHECK_MODULE}.\n"
        "Lemma helper_1: nat -> nat..\n"
        " Proof.\n"
        " admit.\n"
        " Admitted.\n"
        "\n"
        "Lemma helper_1_base: nat..\n"
        " Proof.\n"
        " admit.\n"
        " Admitted.\n"
        "\n"
        "Lemma main: ~ (True.).\n"
        "Proof.\n"
        "apply helper_1_base. apply helper_1.\n"
        "Qed.\n"
        f"End {CHECK_MODULE}.\n"
        f"Print Assumptions {CHECK_MODULE}.main.\n"
    )


def test_make_script_check_assumptions_puts_header_outside_module() -> None:
    goal = Goal(
        statement="True",
        name="trivial",
        environment=RocqEnvironment(header="From Stdlib Require Import PeanoNat."),
    )
    attempt = ProofAttempt(goal=goal, script="exact I.", new_lemmas=[])
    backend = CoqcBackend()

    script = backend.make_script(attempt, check_assumptions=True)

    assert script.code == (
        "From Stdlib Require Import PeanoNat.\n"
        "\n"
        f"Module {CHECK_MODULE}.\n"
        "Lemma trivial: True.\n"
        "Proof.\n"
        "exact I.\n"
        "Qed.\n"
        f"End {CHECK_MODULE}.\n"
        f"Print Assumptions {CHECK_MODULE}.trivial.\n"
    )


def test_check_attempt_uses_new_lemmas_names_when_helper_names_not_specified() -> None:
    main = Goal(statement="True", name="main")
    helper = Goal(statement="True", name="myhelper")
    attempt = ProofAttempt(goal=main, script="exact {myhelper}.", new_lemmas=[helper])

    backend = CoqcBackend()
    received: list[str] = []

    def fake_check_script(
        source: ProofScript, *, tactic_prelude: bool = True, verbose: bool | None = None
    ) -> CoqcResult:
        received.append(source.code)
        return CoqcResult(success=True)

    backend.check_script = fake_check_script  # type: ignore[method-assign]
    backend.check_attempt(attempt, check_assumptions=False)

    assert received[0] == (
        "Lemma myhelper: True.\n"
        " Proof.\n"
        " admit.\n"
        " Admitted.\n"
        "\n"
        "Lemma main: True.\n"
        "Proof.\n"
        "exact myhelper.\n"
        "Qed."
    )


# --- CoqcBackend.check_attempt ---


@pytest.mark.skipif(not COQC_AVAILABLE, reason="coqc not installed")
def test_check_attempt_valid_proof() -> None:
    goal = Goal(statement="True", name="trivial")
    attempt = ProofAttempt(goal=goal, script="exact I.", new_lemmas=[])
    backend = CoqcBackend()

    result = backend.check_attempt(attempt)

    assert result.success is True


@pytest.mark.skipif(not COQC_AVAILABLE, reason="coqc not installed")
def test_check_attempt_allows_new_lemmas_as_assumptions() -> None:
    main = Goal(statement="False", name="main")
    helper = Goal(statement="False", name="helper")
    attempt = ProofAttempt(
        goal=main,
        script="exact helper.",
        new_lemmas=[helper],
    )
    backend = CoqcBackend()

    result = backend.check_attempt(attempt)

    assert result.success is True


@pytest.mark.skipif(not COQC_AVAILABLE, reason="coqc not installed")
def test_check_attempt_substitutes_helper_placeholders() -> None:
    main = Goal(statement="True", name="main")
    helper = Goal(statement="True", name="helper")
    attempt = ProofAttempt(
        goal=main,
        script="exact {helper}.",
        new_lemmas=[helper],
    )
    backend = CoqcBackend()

    result = backend.check_attempt(attempt, helper_names=["helper_1"])

    assert result.success is True


@pytest.mark.skipif(not COQC_AVAILABLE, reason="coqc not installed")
def test_check_attempt_rejects_assumption_outside_allowlist() -> None:
    denied_goal = Goal(statement="True", name="main")
    allowed_goal = Goal(
        statement="True",
        name="main",
        environment=RocqEnvironment(allowed_axioms=frozenset({"lib_ax"})),
    )
    backend = CoqcBackend()
    axiom_script = ProofScript(
        code=(
            "Axiom lib_ax : True.\n"
            f"Module {CHECK_MODULE}.\n"
            "Lemma main: True.\n"
            "Proof.\nexact lib_ax.\nQed.\n"
            f"End {CHECK_MODULE}.\n"
            f"Print Assumptions {CHECK_MODULE}.main.\n"
        )
    )

    def fake_make_script(
        attempt: ProofAttempt, *, check_assumptions: bool = False
    ) -> ProofScript:
        del attempt, check_assumptions
        return axiom_script

    backend.make_script = fake_make_script  # type: ignore[method-assign]

    denied = backend.check_attempt(
        ProofAttempt(goal=denied_goal, script="exact lib_ax.", new_lemmas=[])
    )
    assert denied.success is False
    assert (
        denied.stderr
        == "Disallowed assumptions (not in new_lemmas, allowed_axioms, or header): lib_ax"
    )

    allowed = backend.check_attempt(
        ProofAttempt(goal=allowed_goal, script="exact lib_ax.", new_lemmas=[])
    )
    assert allowed.success is True


@pytest.mark.skipif(not COQC_AVAILABLE, reason="coqc not installed")
def test_check_attempt_allows_environment_axiom_via_header() -> None:
    goal = Goal(
        statement="True",
        name="main",
        environment=RocqEnvironment(header="Axiom lib_ax : True."),
    )
    attempt = ProofAttempt(goal=goal, script="exact lib_ax.", new_lemmas=[])
    backend = CoqcBackend()

    result = backend.check_attempt(attempt)

    assert result.success is True


@pytest.mark.skipif(not COQC_AVAILABLE, reason="coqc not installed")
def test_check_attempt_allows_stdlib_axiom_via_require_header() -> None:
    goal = Goal(
        statement="forall P : Prop, P \\/ ~ P",
        name="main",
        environment=RocqEnvironment(header="Require Import Classical.\n"),
    )
    attempt = ProofAttempt(goal=goal, script="intros. apply classic.", new_lemmas=[])
    backend = CoqcBackend()

    result = backend.check_attempt(attempt)

    assert result.success is True


@pytest.mark.skipif(not COQC_AVAILABLE, reason="coqc not installed")
def test_check_attempt_compile_error_skips_assumption_policy() -> None:
    goal = Goal(statement="True", name="bad")
    attempt = ProofAttempt(goal=goal, script="oops.", new_lemmas=[])
    backend = CoqcBackend()

    result = backend.check_attempt(attempt)

    assert result.success is False
    assert result.output == (
        "Script line 1, characters 0-4:\n"
        "Error: The reference oops was not found in the current environment."
    )


@pytest.mark.skipif(not COQC_AVAILABLE, reason="coqc not installed")
def test_check_attempt_can_skip_assumption_policy() -> None:
    """With check_assumptions=False, probe_names_under_header is not called."""
    goal = Goal(
        statement="True",
        name="main",
        environment=RocqEnvironment(header="Axiom lib_ax : True.\n"),
    )
    attempt = ProofAttempt(goal=goal, script="exact lib_ax.", new_lemmas=[])
    backend = CoqcBackend()

    with patch.object(CoqcBackend, "probe_names_under_header") as mock_probe:
        backend.check_attempt(attempt, check_assumptions=False)

    mock_probe.assert_not_called()


@pytest.mark.skipif(not COQC_AVAILABLE, reason="coqc not installed")
def test_check_attempt_maps_error_to_script_line() -> None:
    """coqc file line numbers are remapped into the LLM tactic script."""
    goal = Goal(
        statement="True",
        name="bad",
        environment=RocqEnvironment(header="From Stdlib Require Import PeanoNat."),
    )
    attempt = ProofAttempt(
        goal=goal,
        script="idtac.\n  oops.",
        new_lemmas=[],
    )
    result = CoqcBackend().check_attempt(attempt, check_assumptions=False)

    assert result.success is False
    assert result.locations == [
        CoqcLocation(line=6, start_char=2, end_char=6, script_line=2)
    ]
    assert result.output == (
        "Script line 2, characters 2-6:\n"
        "Error: The reference oops was not found in the current environment."
    )


@pytest.mark.skipif(not COQC_AVAILABLE, reason="coqc not installed")
def test_check_attempt_hides_file_line_for_incomplete_proof() -> None:
    """Qed. errors must not expose generated-file coordinates to the LLM."""
    goal = Goal(statement="True", name="incomplete")
    attempt = ProofAttempt(goal=goal, script="idtac.", new_lemmas=[])
    result = CoqcBackend().check_attempt(attempt, check_assumptions=False)

    assert result.success is False
    assert result.locations == [
        CoqcLocation(line=4, start_char=0, end_char=4, script_line=None)
    ]
    assert result.output == (
        f"{OUTSIDE_TACTIC_LOCUS}\n"
        "Error:  (in proof incomplete): Attempt to save an incomplete proof\n"
        "(there are remaining open goals)."
    )
