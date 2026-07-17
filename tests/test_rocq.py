"""Tests for llmprover.rocq (CoqcBackend)."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llmprover.domain import Goal, Polarity, ProofAttempt
from llmprover.proof_script import ProofScript
from llmprover.rocq import CoqcBackend

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


# --- make_script (no coqc needed) ---


def test_make_script_builds_main_lemma_only() -> None:
    goal = Goal(statement="True", name="trivial")
    attempt = ProofAttempt(goal=goal, script="exact I.", new_lemmas=[])
    backend = CoqcBackend()

    script = backend.make_script(attempt)

    assert script.code == "Lemma trivial: True.\nProof.\nexact I.\nQed."


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


# --- CoqcBackend (real coqc) ---


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


@pytest.mark.skipif(not COQC_AVAILABLE, reason="coqc not installed")
def test_check_attempt_valid_proof() -> None:
    goal = Goal(statement="True", name="trivial")
    attempt = ProofAttempt(goal=goal, script="exact I.", new_lemmas=[])
    backend = CoqcBackend()

    result = backend.check_attempt(attempt)

    assert result.success is True


@patch(
    "llmprover.rocq.subprocess.run",
    side_effect=subprocess.TimeoutExpired(cmd=["coqc"], timeout=SLOW_CHECK_TIMEOUT),
)
def test_check_script_timeout(mock_run: MagicMock) -> None:
    backend = CoqcBackend(timeout=SLOW_CHECK_TIMEOUT)
    result = backend.check_script(ProofScript(code=VALID_IMPORT))

    assert result.success is False
    assert result.output == f"coqc timed out after {SLOW_CHECK_TIMEOUT}s"
