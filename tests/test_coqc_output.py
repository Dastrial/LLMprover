"""Tests for llmprover.coqc_output."""

from __future__ import annotations

from llmprover.coqc_output import (
    OUTSIDE_TACTIC_LOCUS,
    CoqcLocation,
    CoqcResult,
    _skip_coqc_warning_block,
    parse_coqc_locations,
    parse_print_assumptions,
    remap_coqc_stderr,
    strip_coqc_warnings,
)

# --- _skip_coqc_warning_block ---


def test_skip_coqc_warning_block_from_locus_and_bare_warning() -> None:
    lines = [
        "Script line 1, characters 0-3:",
        "Warning: deprecated",
        "[deprecated,default]",
        "Script line 2, characters 0-3:",
        "Error: boom",
    ]
    assert _skip_coqc_warning_block(lines, 0) == 3

    bare = [
        "Warning: alone",
        "[foo,default]",
        "Error: boom",
    ]
    assert _skip_coqc_warning_block(bare, 0) == 2


# --- strip_coqc_warnings ---


def test_strip_coqc_warnings_removes_warning_blocks_keeps_errors() -> None:
    text = (
        "Warning: bare prelude warning\n"
        "[prelude,default]\n"
        'File "/tmp/check.v", line 2, characters 0-4:\n'
        "Error: first error after bare warning.\n"
        "Script line 1, characters 0-3:\n"
        "Warning: deprecated tactic\n"
        "Use something else.\n"
        "[deprecated,default]\n"
        "Script line 2, characters 4-8:\n"
        "Error: second error between warnings.\n"
        "Unable to unify.\n"
        "Generated Rocq code outside the tactic script:\n"
        "Warning: Loading Stdlib without prefix is deprecated.\n"
        'Use "From Stdlib Require Import ZArith" instead.\n'
        "[deprecated-missing-stdlib,deprecated,default]\n"
        'File "/tmp/check.v", line 17, characters 0-4:\n'
        "\n"
        "Warning: spaced locus warning\n"
        "[spaced,default]\n"
        "Script line 7, characters 5-16:\n"
        "Error:\n"
        "third error at the end.\n"
    )
    assert strip_coqc_warnings(text) == (
        'File "/tmp/check.v", line 2, characters 0-4:\n'
        "Error: first error after bare warning.\n"
        "Script line 2, characters 4-8:\n"
        "Error: second error between warnings.\n"
        "Unable to unify.\n"
        "Script line 7, characters 5-16:\n"
        "Error:\n"
        "third error at the end."
    )


def test_strip_coqc_warnings_handles_warning_without_locus() -> None:
    assert strip_coqc_warnings("Warning: alone\n[foo,default]\nError: boom") == (
        "Error: boom"
    )


# --- parse_coqc_locations / remap_coqc_stderr ---


def test_parse_coqc_locations_extracts_loci() -> None:
    stderr = (
        'File "/tmp/check.v", line 10, characters 0-3:\n'
        "Error: The reference lia was not found in the current environment.\n"
    )
    locs = parse_coqc_locations(stderr)
    assert len(locs) == 1
    assert locs[0].line == 10
    assert locs[0].start_char == 0
    assert locs[0].end_char == 3
    assert locs[0].script_line is None


def test_parse_coqc_locations_remaps_into_tactic_script() -> None:
    stderr = (
        'File "/tmp/check.v", line 11, characters 0-3:\n'
        "Error: boom\n"
        'File "/tmp/check.v", line 3, characters 15-20:\n'
        "Warning: deprecated\n"
    )
    locs = parse_coqc_locations(stderr, tactic_start_line=10, tactic_end_line=11)
    assert locs[0].script_line == 2
    assert locs[1].script_line is None


def test_remap_coqc_stderr_rewrites_loci() -> None:
    stderr = (
        'File "/tmp/check.v", line 11, characters 0-3:\n'
        "Error: boom\n"
        'File "/tmp/check.v", line 3, characters 15-20:\n'
        "Warning: deprecated\n"
    )
    locs = parse_coqc_locations(stderr, tactic_start_line=10, tactic_end_line=11)
    assert remap_coqc_stderr(stderr, locs) == (
        "Script line 2, characters 0-3:\n"
        "Error: boom\n"
        f"{OUTSIDE_TACTIC_LOCUS}\n"
        "Warning: deprecated\n"
    )


def test_remap_coqc_stderr_hides_qed_file_coordinates() -> None:
    stderr = (
        'File "/tmp/check.v", line 17, characters 0-4:\n'
        "Error: Attempt to save an incomplete proof.\n"
    )
    locs = parse_coqc_locations(stderr, tactic_start_line=10, tactic_end_line=16)
    assert locs[0].script_line is None
    assert remap_coqc_stderr(stderr, locs) == (
        f"{OUTSIDE_TACTIC_LOCUS}\nError: Attempt to save an incomplete proof.\n"
    )


# --- CoqcResult ---


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
    result = CoqcResult(
        success=False,
        stdout="line1",
        stderr="line2",
        locations=[
            CoqcLocation(line=10, start_char=0, end_char=3, script_line=2),
            CoqcLocation(line=3, start_char=15, end_char=20),
        ],
    )
    assert str(result) == (
        "CoqcResult(success=False, stdout=line1, stderr=line2, locations=["
        "CoqcLocation(line=10, start_char=0, end_char=3, script_line=2), "
        "CoqcLocation(line=3, start_char=15, end_char=20, script_line=None)])"
    )


def test_coqc_result_output_prefers_script_stderr() -> None:
    result = CoqcResult(
        success=False,
        stderr='File "x.v", line 10, characters 0-3:\nError: boom',
        script_stderr="Script line 2, characters 0-3:\nError: boom",
    )
    assert result.output == "Script line 2, characters 0-3:\nError: boom"
    assert result.display_stderr == "Script line 2, characters 0-3:\nError: boom"


def test_coqc_result_output_without_warnings() -> None:
    result = CoqcResult(
        success=False,
        script_stderr=(
            "Script line 1, characters 0-3:\n"
            "Warning: deprecated\n"
            "[deprecated,default]\n"
            "Script line 2, characters 0-3:\n"
            "Error: boom"
        ),
    )
    assert result.output == (
        "Script line 1, characters 0-3:\n"
        "Warning: deprecated\n"
        "[deprecated,default]\n"
        "Script line 2, characters 0-3:\n"
        "Error: boom"
    )
    assert result.output_without_warnings == (
        "Script line 2, characters 0-3:\nError: boom"
    )


# --- parse_print_assumptions ---


def test_parse_print_assumptions_closed() -> None:
    assert parse_print_assumptions("Closed under the global context\n") == set()


def test_parse_print_assumptions_extracts_fully_qualified_names() -> None:
    output = """\
Axioms:
LLMProver.helper : False
ClassicalDedekindReals.sig_forall_dec :
  forall P : nat -> Prop,
  (forall n : nat, {P n} + {~ P n}) ->
  {n : nat | ~ P n} + {forall n : nat, P n}
FunctionalExtensionality.functional_extensionality_dep :
  forall (A : Type) (B : A -> Type) (f g : forall x : A, B x),
  (forall x : A, f x = g x) -> f = g
"""
    assert parse_print_assumptions(output) == {
        "LLMProver.helper",
        "ClassicalDedekindReals.sig_forall_dec",
        "FunctionalExtensionality.functional_extensionality_dep",
    }


def test_parse_print_assumptions_skips_warning_error_file_names() -> None:
    output = """\
Axioms:
Warning: deprecated axiom listing
Error: unexpected coqc noise
File: not an axiom either
LLMProver.helper : False
"""
    assert parse_print_assumptions(output) == {"LLMProver.helper"}
