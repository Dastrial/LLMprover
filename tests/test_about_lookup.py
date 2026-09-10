"""Tests for llmprover.prover_agents.about_lookup."""

from __future__ import annotations

from unittest.mock import MagicMock

from llmprover.coqc_output import CoqcResult
from llmprover.llm_client import CompletionResult, TokenUsage
from llmprover.prompts import (
    PromptMessage,
    PromptPart,
)
from llmprover.prover_agents.about_lookup import (
    EMPTY_ABOUT,
    EMPTY_SEARCH,
    MAX_SEARCH_HITS_COLLECT,
    MAX_SELECTED_SEARCH_HITS,
    SEARCH_SELECTION_THRESHOLD,
    build_about_script,
    build_search_script,
    collect_names_from_searches,
    parse_lemma_names,
    parse_lookup_request,
    parse_search_commands,
    parse_search_hit_names,
    request_abouts,
    request_search_then_abouts,
    request_searches,
    run_abouts,
    run_searches,
)


def test_parse_lemma_names_one_per_line_strips_about_and_dots() -> None:
    text = "```\nAbout Nat.add_comm.\nNat.mul_comm\n\nNat.add_comm\n```"
    assert parse_lemma_names(text) == ["Nat.add_comm", "Nat.mul_comm"]


def test_parse_search_commands_keeps_only_search_family() -> None:
    text = (
        "```\n"
        'Search "gcd".\n'
        "About Nat.gcd.\n"
        "SearchPattern (_ + 0).\n"
        "Require Import Foo.\n"
        "SearchRewrite (_ + 0 = _).\n"
        "SearchHead plus.\n"
        "search \"add\" \"0\"\n"
        "```"
    )
    assert parse_search_commands(text) == [
        'Search "gcd".',
        "SearchPattern (_ + 0).",
        "SearchRewrite (_ + 0 = _).",
        'Search "add" "0".',
    ]


def test_parse_search_commands_keeps_first_statement_only() -> None:
    line = 'Search "gcd". About Nat.gcd.'
    assert parse_search_commands(line) == ['Search "gcd".']


def test_parse_lookup_request_keeps_search_and_about_lines() -> None:
    text = (
        "```\n"
        'Search "gcd".\n'
        "About Nat.gcd.\n"
        "SearchPattern (_ + 0).\n"
        "Nat.mul_comm\n"
        "Require Import Foo.\n"
        "About Nat.add_0_r.\n"
        "SearchRewrite (_ + 0 = _).\n"
        "SearchHead plus.\n"
        "```"
    )
    commands, names = parse_lookup_request(text)
    assert commands == [
        'Search "gcd".',
        "SearchPattern (_ + 0).",
        "SearchRewrite (_ + 0 = _).",
    ]
    assert names == ["Nat.gcd", "Nat.add_0_r"]


def test_parse_lookup_request_caps_search_commands_at_three() -> None:
    text = (
        'Search "a".\n'
        'Search "b".\n'
        'Search "c".\n'
        'Search "d".\n'
        "About Nat.gcd.\n"
    )
    commands, names = parse_lookup_request(text)
    assert commands == ['Search "a".', 'Search "b".', 'Search "c".']
    assert names == ["Nat.gcd"]


def test_parse_search_hit_names_name_only_one_per_line() -> None:
    output = (
        "BinInt.Z.add_shuffle0\n"
        "Nat.add_0_r\n"
        "  indented_ignored\n"
        "Warning\n"
        "Error: ignored\n"
    )
    assert parse_search_hit_names(output) == [
        "BinInt.Z.add_shuffle0",
        "Nat.add_0_r",
    ]


def test_parse_search_hit_names_rejects_type_suffix_lines() -> None:
    # Name-only mode: lines with ": type" are not valid whole-line names.
    output = "Nat.add_0_r: forall n : nat, n + 0 = n\nNat.add_comm\n"
    assert parse_search_hit_names(output) == ["Nat.add_comm"]


def test_build_about_script_includes_header_and_abouts() -> None:
    script = build_about_script("Require Import Nat.", ["Nat.gcd", "Nat.add"])
    assert script == (
        "Require Import Nat.\n"
        "\n"
        "About Nat.gcd.\n"
        "About Nat.add.\n"
    )


def test_build_search_script_includes_header_name_only_and_one_command() -> None:
    script = build_search_script("Require Import Nat.", 'Search "gcd".')
    assert script == (
        "Require Import Nat.\n"
        "\n"
        "Set Search Output Name Only.\n"
        'Search "gcd".\n'
    )


def test_build_search_script_can_omit_name_only() -> None:
    script = build_search_script(
        "Require Import Nat.", 'Search "gcd".', name_only=False
    )
    assert script == (
        "Require Import Nat.\n"
        "\n"
        'Search "gcd".\n'
    )


def test_run_abouts_empty_names_skips_coqc() -> None:
    checker = MagicMock()
    assert run_abouts(checker, "Require Import Nat.", []) == EMPTY_ABOUT
    checker.check_script.assert_not_called()


def test_run_searches_empty_commands_skips_coqc() -> None:
    checker = MagicMock()
    assert run_searches(checker, "Require Import Nat.", []) == EMPTY_SEARCH
    checker.check_script.assert_not_called()


def test_run_abouts_returns_coqc_output() -> None:
    checker = MagicMock()
    checker.check_script.return_value = CoqcResult(
        success=True, stdout="gcd : nat -> nat -> nat\n"
    )
    assert run_abouts(checker, "Require Import Nat.", ["Nat.gcd"]) == (
        "gcd : nat -> nat -> nat\n"
    )


def test_run_searches_returns_labeled_coqc_output() -> None:
    checker = MagicMock()
    checker.check_script.return_value = CoqcResult(
        success=True, stdout="Nat.gcd\n"
    )
    assert run_searches(checker, "Require Import Nat.", ['Search "gcd".']) == (
        '=== Search "gcd". ===\n'
        "Nat.gcd\n"
    )
    script = checker.check_script.call_args.args[0].code
    assert "Set Search Output Name Only." in script


def test_run_searches_continues_after_failed_command() -> None:
    checker = MagicMock()
    checker.check_script.side_effect = [
        CoqcResult(success=False, stderr="Error: bad pattern\n"),
        CoqcResult(success=True, stdout="Nat.gcd\n"),
    ]
    output = run_searches(
        checker,
        "Require Import Nat.",
        ["SearchPattern (_ +++ 0).", 'Search "gcd".'],
    )
    assert checker.check_script.call_count == 2
    assert output == (
        "=== SearchPattern (_ +++ 0). [failed] ===\n"
        "Error: bad pattern\n"
        "\n"
        '=== Search "gcd". ===\n'
        "Nat.gcd\n"
    )
    scripts = [call.args[0].code for call in checker.check_script.call_args_list]
    assert scripts[0].count("SearchPattern") == 1
    assert "Set Search Output Name Only." in scripts[0]
    assert 'SearchPattern (_ +++ 0).' in scripts[0]
    assert 'Search "gcd".' not in scripts[0]
    assert 'Search "gcd".' in scripts[1]


def test_request_abouts_calls_llm_then_coqc(tmp_path) -> None:
    system_path = tmp_path / "system.txt"
    before_path = tmp_path / "before.txt"
    after_path = tmp_path / "after.txt"
    system_path.write_text("SYSTEM", encoding="utf-8")
    before_path.write_text("User header={header}", encoding="utf-8")
    after_path.write_text("Statement:\n{statement}", encoding="utf-8")

    model = MagicMock()
    model.complete.return_value = CompletionResult(
        text="Nat.gcd\n", usage=TokenUsage(1, 2)
    )
    checker = MagicMock()
    checker.check_script.return_value = CoqcResult(
        success=True, stdout="gcd : nat -> nat -> nat\n"
    )

    results, usage, names = request_abouts(
        model,
        checker,
        system_prompt=system_path,
        user_before_prompt=before_path,
        user_after_prompt=after_path,
        header="Require Import Nat.",
        statement="Nat.gcd 180 168 = 12.",
    )

    assert results == "gcd : nat -> nat -> nat\n"
    assert usage == TokenUsage(1, 2)
    assert names == ["Nat.gcd"]
    model.complete.assert_called_once_with(
        [
            PromptMessage.text("system", "SYSTEM", cache_breakpoint=True),
            PromptMessage(
                role="user",
                parts=(
                    PromptPart("User header=Require Import Nat.\n", cache_breakpoint=True),
                    PromptPart("Statement:\nNat.gcd 180 168 = 12.\n"),
                ),
            ),
        ]
    )
    checker.check_script.assert_called_once()
    script = checker.check_script.call_args.args[0].code
    assert "Require Import Nat." in script
    assert "About Nat.gcd." in script


def test_request_searches_calls_llm_then_coqc(tmp_path) -> None:
    system_path = tmp_path / "system.txt"
    before_path = tmp_path / "before.txt"
    after_path = tmp_path / "after.txt"
    system_path.write_text("SEARCH_SYSTEM", encoding="utf-8")
    before_path.write_text("Search header={header}", encoding="utf-8")
    after_path.write_text("Statement:\n{statement}", encoding="utf-8")

    model = MagicMock()
    model.complete.return_value = CompletionResult(
        text='Search "gcd".\n', usage=TokenUsage(1, 1)
    )
    checker = MagicMock()
    checker.check_script.return_value = CoqcResult(
        success=True, stdout="Nat.gcd\n"
    )

    results, usage, commands = request_searches(
        model,
        checker,
        system_prompt=system_path,
        user_before_prompt=before_path,
        user_after_prompt=after_path,
        header="Require Import Nat.",
        statement="Nat.gcd 180 168 = 12.",
    )

    assert results == (
        '=== Search "gcd". ===\n'
        "Nat.gcd\n"
    )
    assert usage == TokenUsage(1, 1)
    assert commands == ['Search "gcd".']
    model.complete.assert_called_once_with(
        [
            PromptMessage.text("system", "SEARCH_SYSTEM", cache_breakpoint=True),
            PromptMessage(
                role="user",
                parts=(
                    PromptPart("Search header=Require Import Nat.\n", cache_breakpoint=True),
                    PromptPart("Statement:\nNat.gcd 180 168 = 12.\n"),
                ),
            ),
        ]
    )
    checker.check_script.assert_called_once()
    script = checker.check_script.call_args.args[0].code
    assert "Require Import Nat." in script
    assert 'Search "gcd".' in script
    assert "Set Search Output Name Only." in script


def _hit_lines(count: int) -> str:
    return "".join(f"Lemma{i}\n" for i in range(count))


def test_collect_names_from_searches_keeps_all_hits_up_to_cap() -> None:
    checker = MagicMock()
    checker.check_script.side_effect = [
        CoqcResult(success=True, stdout=_hit_lines(SEARCH_SELECTION_THRESHOLD + 2)),
        CoqcResult(success=True, stdout="Nat.gcd\n"),
    ]
    names = collect_names_from_searches(
        checker,
        "Require Import Nat.",
        ['Search "broad".', 'Search "gcd".'],
    )
    assert names == [
        *(f"Lemma{i}" for i in range(SEARCH_SELECTION_THRESHOLD + 2)),
        "Nat.gcd",
    ]
    assert checker.check_script.call_count == 2


def test_collect_names_from_searches_skips_failed_commands() -> None:
    checker = MagicMock()
    checker.check_script.side_effect = [
        CoqcResult(success=False, stderr="Error: bad pattern\n"),
        CoqcResult(success=True, stdout="Nat.gcd\n"),
    ]
    names = collect_names_from_searches(
        checker,
        "Require Import Nat.",
        ["SearchPattern (_ +++ 0).", 'Search "gcd".'],
    )
    assert names == ["Nat.gcd"]


def test_collect_names_from_searches_caps_at_three_commands() -> None:
    checker = MagicMock()
    checker.check_script.return_value = CoqcResult(
        success=True, stdout="Nat.gcd\n"
    )
    collect_names_from_searches(
        checker,
        "Require Import Nat.",
        ['Search "a".', 'Search "b".', 'Search "c".', 'Search "d".'],
    )
    assert checker.check_script.call_count == 3


def test_collect_names_from_searches_caps_total_hits() -> None:
    checker = MagicMock()
    checker.check_script.return_value = CoqcResult(
        success=True, stdout=_hit_lines(MAX_SEARCH_HITS_COLLECT + 50)
    )
    names = collect_names_from_searches(
        checker, "Require Import Nat.", ['Search "huge".']
    )
    assert len(names) == MAX_SEARCH_HITS_COLLECT


def test_request_search_then_abouts_abouts_explicit_and_search_hits(tmp_path) -> None:
    system_path = tmp_path / "search_system.txt"
    before_path = tmp_path / "search_before.txt"
    after_path = tmp_path / "search_after.txt"
    system_path.write_text("SEARCH_SYS", encoding="utf-8")
    before_path.write_text("search header={header}", encoding="utf-8")
    after_path.write_text("search {statement}", encoding="utf-8")

    model = MagicMock()
    model.complete.return_value = CompletionResult(
        text='Search "gcd".\nAbout Nat.mul_comm.\n',
        usage=TokenUsage(1, 1),
    )
    checker = MagicMock()

    def check_script(script, **_kwargs):
        if "Search" in script.code:
            return CoqcResult(success=True, stdout="Nat.gcd\n")
        return CoqcResult(
            success=True,
            stdout="mul_comm : nat -> nat\ngcd : nat -> nat -> nat\n",
        )

    checker.check_script.side_effect = check_script

    lookup = request_search_then_abouts(
        model,
        checker,
        system_prompt=system_path,
        user_before_prompt=before_path,
        user_after_prompt=after_path,
        header="Require Import Nat.",
        statement="Nat.gcd 180 168 = 12.",
    )

    assert lookup.about_results == (
        "mul_comm : nat -> nat\ngcd : nat -> nat -> nat\n"
    )
    assert lookup.usage == TokenUsage(1, 1)
    assert lookup.lemma_names == ["Nat.mul_comm", "Nat.gcd"]
    assert lookup.search_output == 'Search "gcd".\nAbout Nat.mul_comm.\n'
    assert lookup.select_output is None
    prefix = lookup.prefix_messages
    assert prefix[0] == PromptMessage.text("system", "SEARCH_SYS", cache_breakpoint=True)
    assert prefix[1] == PromptMessage(
        role="user",
        parts=(
            PromptPart("search header=Require Import Nat.\n", cache_breakpoint=True),
            PromptPart("search Nat.gcd 180 168 = 12.\n"),
        ),
    )
    assert prefix[1].parts[0].cache_breakpoint is True
    assert prefix[1].parts[1].cache_breakpoint is False
    assert prefix[2].role == "assistant"
    assert prefix[2].parts[0].cache_breakpoint is False
    assert 'Search "gcd".' in prefix[2].joined_text()
    model.complete.assert_called_once()
    scripts = [call.args[0].code for call in checker.check_script.call_args_list]
    assert any('Search "gcd".' in code for code in scripts)
    assert any("Set Search Output Name Only." in code for code in scripts)
    about_script = next(code for code in scripts if "About " in code)
    assert about_script.index("About Nat.mul_comm.") < about_script.index(
        "About Nat.gcd."
    )


def test_request_search_then_abouts_selects_when_too_many_hits(tmp_path) -> None:
    system_path = tmp_path / "search_system.txt"
    before_path = tmp_path / "search_before.txt"
    after_path = tmp_path / "search_after.txt"
    system_path.write_text("SEARCH_SYS", encoding="utf-8")
    before_path.write_text("before {header}", encoding="utf-8")
    after_path.write_text("after {statement}", encoding="utf-8")

    many = [f"Nat.Lemma{i}" for i in range(SEARCH_SELECTION_THRESHOLD)]
    model = MagicMock()
    model.complete.side_effect = [
        CompletionResult(text='Search "Lem".\n', usage=TokenUsage(1, 1)),
        CompletionResult(
            text="\n".join(many[:MAX_SELECTED_SEARCH_HITS]) + "\n",
            usage=TokenUsage(2, 2),
        ),
    ]
    checker = MagicMock()

    def check_script(script, **_kwargs):
        if "Search" in script.code:
            return CoqcResult(success=True, stdout="\n".join(many) + "\n")
        return CoqcResult(success=True, stdout="ok : nat\n")

    checker.check_script.side_effect = check_script

    lookup = request_search_then_abouts(
        model,
        checker,
        system_prompt=system_path,
        user_before_prompt=before_path,
        user_after_prompt=after_path,
        header="Require Import Nat.",
        statement="True.",
    )

    assert model.complete.call_count == 2
    assert lookup.usage == TokenUsage(3, 3)
    assert lookup.lemma_names == many[:MAX_SELECTED_SEARCH_HITS]
    assert lookup.about_results == "ok : nat\n"
    assert lookup.search_output == 'Search "Lem".\n'
    assert lookup.select_output == "\n".join(many[:MAX_SELECTED_SEARCH_HITS]) + "\n"
    select_call = model.complete.call_args_list[1].args[0]
    assert select_call[2].role == "assistant"
    assert select_call[2].parts[0].cache_breakpoint is False
    assert select_call[3].role == "user"
    assert select_call[3].parts[0].cache_breakpoint is False
    assert len(select_call[3].parts) == 1
    assert "Nat.Lemma0" in select_call[3].joined_text()
    # Final prefix still stops after the first assistant reply.
    prefix = lookup.prefix_messages
    assert len(prefix) == 3
    assert prefix[2].joined_text().startswith('Search "Lem".')
    assert prefix[2].parts[0].cache_breakpoint is False
