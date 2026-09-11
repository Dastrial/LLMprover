"""Tests for miniF2F loader."""

from __future__ import annotations

import pytest

from llmprover.minif2f.loader import parse_rocq_statement, problem_from_row


def test_parse_rocq_statement_strips_theorem() -> None:
    statement = parse_rocq_statement(
        "Theorem plus_n0 :\n  forall n : nat, n + 0 = n.",
    )
    assert statement == "forall n : nat, n + 0 = n"


def test_parse_rocq_statement_binders_become_forall() -> None:
    statement = parse_rocq_statement(
        "Theorem foo (n : nat) (h : n > 0) :\n  n + 0 = n.",
    )
    assert statement == "forall (n : nat) (h : n > 0), n + 0 = n"


def test_parse_rocq_statement_multi_name_binder() -> None:
    statement = parse_rocq_statement(
        "Theorem amc12_2001_p2 (a b n : nat) :\n  (1 <= a <= 9) -> True.",
    )
    assert statement == "forall (a b n : nat), (1 <= a <= 9) -> True"


def test_parse_rocq_statement_bare_identifier_binder() -> None:
    statement = parse_rocq_statement(
        "Theorem mathd_algebra_158 a :\n"
        "   list_sum (map (fun k => 2 * k + 1) (seq 0 8)) = 4 ->\n"
        "  a = 8.",
    )
    assert statement == (
        "forall a, list_sum (map (fun k => 2 * k + 1) (seq 0 8)) = 4 ->\n  a = 8"
    )


def test_parse_rocq_statement_rejects_malformed_row() -> None:
    """Corrupt HF rows (e.g. amc12_2001_p5) raise so the loader can skip them."""
    with pytest.raises(ValueError, match="Not a Theorem/Lemma"):
        parse_rocq_statement("Theorem:**\n   - *Natural Language Description:* foo.")


def test_load_skips_malformed_rows() -> None:
    """Simulate load_minif2f skip behaviour without HuggingFace."""
    good = {
        "name": "ok",
        "split": "valid",
        "rocq_statement": "Theorem ok : True.",
        "header": "",
    }
    bad = {
        "name": "amc12_2001_p5",
        "split": "valid",
        "rocq_statement": "Theorem:**\n   - *Natural Language Description:* foo.",
        "header": "",
    }
    assert problem_from_row(good).statement == "True"
    with pytest.raises(ValueError):
        problem_from_row(bad)

    problems = []
    for row in (good, bad):
        try:
            problems.append(problem_from_row(row))
        except ValueError:
            continue
    assert [p.name for p in problems] == ["ok"]


def test_problem_from_row_format_statement() -> None:
    problem = problem_from_row(
        {
            "name": "mathd_algebra_405",
            "split": "valid",
            "rocq_statement": (
                "Theorem mathd_algebra_405 :\n  forall x : nat, 0 < x -> True."
            ),
            "header": "Require Import Arith.",
            "informal_statement": "…",
            "informal_proof": "…",
        }
    )
    assert problem.format_statement() == (
        "mathd_algebra_405: forall x : nat, 0 < x -> True"
    )
    goal = problem.to_goal()
    assert goal.name == "mathd_algebra_405"
    assert goal.statement == "forall x : nat, 0 < x -> True"
    assert goal.environment.header == "Require Import Arith."
    assert goal.environment.allowed_axioms == frozenset()


def test_problem_to_goal_keeps_header_axioms_in_prelude() -> None:
    header = (
        "Require Import Coq.Reals.Reals.\n"
        "Parameter Rfloor : R -> Z.\n"
        "Axiom Rfloor_spec : forall x : R, True.\n"
    )
    problem = problem_from_row(
        {
            "name": "amc12a_2016_p3",
            "split": "valid",
            "rocq_statement": "Theorem amc12a_2016_p3 : True.",
            "header": header,
        }
    )
    goal = problem.to_goal()
    assert goal.environment.header == header
    assert goal.environment.allowed_axioms == frozenset()
