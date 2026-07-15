"""Tests for llmprover.domain."""

from __future__ import annotations

from llmprover.domain import CoqcResult


def test_coqc_result_output_combines_stdout_and_stderr() -> None:
    result = CoqcResult(success=False, stdout="line1", stderr="line2")
    assert result.output == "line1\nline2"


def test_coqc_result_output_empty_when_no_output() -> None:
    result = CoqcResult(success=True)
    assert result.output == ""


def test_coqc_result_output_no_stdout() -> None:
    result = CoqcResult(success=False, stderr="line1")
    assert result.output == "line1"


def test_coqc_result_output_no_stderr() -> None:
    result = CoqcResult(success=False, stdout="line1")
    assert result.output == "line1"


def test_coqc_result_str() -> None:
    str_result = str(CoqcResult(success=False, stdout="line1", stderr="line2"))
    assert str_result == "CoqcResult(success=False, stdout=line1, stderr=line2)"
