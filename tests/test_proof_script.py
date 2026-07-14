"""Tests for llmprover.proof_script."""

from __future__ import annotations

from pathlib import Path

import pytest

from llmprover.proof_script import ProofScript

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
