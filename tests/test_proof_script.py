"""Tests for llmprover.rocq.proof_script."""

from __future__ import annotations

from pathlib import Path

import pytest

from llmprover.rocq.proof_script import (
    ProofScript,
    shift_tactic_region,
    tactic_line_span,
)

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


# --- ProofScript ---


def test_proof_script_from_file_reads_example() -> None:
    source = "From Stdlib Require Import PeanoNat.\n\nLemma plus_n0 : forall n : nat, n + 0 = n.\nProof.\ninduction n as [|n IHn].\n- reflexivity.\n- cbn. rewrite IHn. reflexivity.\nQed."
    script = ProofScript.from_file(EXAMPLES / "plus_n0.v")
    assert source == script.code


def test_proof_script_from_file_rejects_non_v_extension(tmp_path: Path) -> None:
    file_path = tmp_path / "not_rocq.txt"
    file_path.write_text("hello", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Expected a \.v file"):
        ProofScript.from_file(file_path)


def test_proof_script_from_file_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        ProofScript.from_file(EXAMPLES / "does_not_exist.v")


def test_proof_script_from_file_with_directory(tmp_path: Path) -> None:
    dir_path = tmp_path / "dir.v"
    dir_path.mkdir()
    with pytest.raises(FileNotFoundError):
        ProofScript.from_file(dir_path)


# --- tactic_line_span / shift_tactic_region ---


def test_tactic_line_span_by_prefix_and_tactic_sizes() -> None:
    assert tactic_line_span("a\nb\n", "exact I.") == (3, 3)
    assert tactic_line_span("a\nb\n", "idtac.\nfoo.\nbar.") == (3, 5)
    assert tactic_line_span("", "idtac.\nfoo.\nbar.") == (1, 3)


def test_shift_tactic_region_applies_prefix_newlines() -> None:
    source = ProofScript(
        code="Lemma t: True.\nProof.\nidtac.\nQed.\n",
        tactic_start_line=3,
        tactic_end_line=4,
    )
    prefix = "From Stdlib Require Import Lia.\n\n"
    assert shift_tactic_region(source, prefix) == (5, 6)


def test_shift_tactic_region_unchanged_with_empty_prefix() -> None:
    source = ProofScript(
        code="Lemma t: True.\nProof.\nidtac.\nQed.\n",
        tactic_start_line=3,
        tactic_end_line=4,
    )
    assert shift_tactic_region(source, "") == (3, 4)


def test_shift_tactic_region_returns_none_without_span() -> None:
    source = ProofScript(code="Lemma t: True.\nProof.\nlia.\nQed.\n")
    assert shift_tactic_region(source, "From Stdlib Require Import Lia.\n\n") == (
        None,
        None,
    )
