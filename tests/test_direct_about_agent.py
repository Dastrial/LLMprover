"""Tests for llmprover.prover_agents.direct_about_agent."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from llmprover.domain import (
    CoqcResult,
    Goal,
    LemmaNode,
    Polarity,
    ProofAttempt,
    RocqEnvironment,
    statement_for_polarity,
)
from llmprover.llm.client import CompletionResult, TokenUsage
from llmprover.llm.prompting import (
    PromptMessage,
    PromptPart,
    fill_prompt,
)
from llmprover.prover_agents.direct_about_agent import DirectAboutAgent
from llmprover.prover_agents.prompt_assembly import (
    DIRECT_ABOUT_SPEC,
    cached_system_prompt,
)

PROMPTS_DIR = (
    Path(__file__).resolve().parent.parent / "llmprover" / "prover_agents" / "prompts"
)
GOAL = Goal(
    name="mathd_numbertheory_188",
    statement="Nat.gcd 180 168 = 12.",
    environment=RocqEnvironment(header="Require Import Nat."),
)
SEARCH_SYSTEM = cached_system_prompt(DIRECT_ABOUT_SPEC)
SEARCH_HITS = "Nat.gcd\n"
ABOUT_STDOUT = "gcd : nat -> nat -> nat\n"


def _check_script(script, **_kwargs):
    if "Search" in script.code:
        return CoqcResult(success=True, stdout=SEARCH_HITS)
    return CoqcResult(success=True, stdout=ABOUT_STDOUT)


def test_prove_runs_search_then_about_then_direct_proof() -> None:
    mock_model = MagicMock()
    mock_model.complete.side_effect = [
        CompletionResult(text='Search "gcd".\nAbout Nat.mul_comm.\n', usage=TokenUsage(1, 1)),
        CompletionResult(text="reflexivity.", usage=TokenUsage(3, 4)),
    ]
    agent = DirectAboutAgent(mock_model)
    agent.checker = MagicMock()
    agent.checker.check_script.side_effect = _check_script

    attempt, usage = agent.prove(LemmaNode(goal=GOAL), Polarity.Positive)

    assert attempt == ProofAttempt(
        goal=GOAL,
        polarity=Polarity.Positive,
        script="reflexivity.",
        new_lemmas=[],
        about_lemmas=["Nat.mul_comm", "Nat.gcd"],
        search_about_output='Search "gcd".\nAbout Nat.mul_comm.\n',
    )
    assert usage == TokenUsage(4, 5)
    assert mock_model.complete.call_count == 2

    statement = statement_for_polarity(GOAL.statement, Polarity.Positive)
    search_call = mock_model.complete.call_args_list[0].args[0]
    before = fill_prompt(
            (PROMPTS_DIR / "direct_about_search_user_before.txt")
            .read_text(encoding="utf-8")
            .strip(),
            header="Require Import Nat.",
        )
    if not before.endswith("\n"):
        before = f"{before}\n"
    after = fill_prompt(
            (PROMPTS_DIR / "direct_about_search_user_after.txt")
            .read_text(encoding="utf-8")
            .strip(),
            statement=statement,
        )
    if not after.endswith("\n"):
        after = f"{after}\n"
    assert search_call == [
        PromptMessage.text("system", SEARCH_SYSTEM, cache_breakpoint=True),
        PromptMessage(
            role="user",
            parts=(PromptPart(before, cache_breakpoint=True), PromptPart(after)),
        ),
    ]
    assert search_call[0].parts[-1].cache_breakpoint is True
    assert search_call[1].parts[0].cache_breakpoint is True
    assert search_call[1].parts[1].cache_breakpoint is False
    assert "INTERACTION PROTOCOL" in SEARCH_SYSTEM
    assert "ABOUT BEFORE LIBRARY USE" in SEARCH_SYSTEM
    assert "OUTPUT PROTOCOL — LOOKUP" in SEARCH_SYSTEM

    proof_call = mock_model.complete.call_args_list[1].args[0]
    assert proof_call[0] == PromptMessage.text("system", SEARCH_SYSTEM, cache_breakpoint=True)
    assert proof_call[1] == search_call[1]
    assert proof_call[2].role == "assistant"
    assert proof_call[2].parts[0].cache_breakpoint is False
    assert 'Search "gcd".' in proof_call[2].joined_text()
    final_user = fill_prompt(
            (PROMPTS_DIR / "direct_about_proof_user.txt")
            .read_text(encoding="utf-8")
            .strip(),
            header="Require Import Nat.",
            statement=statement,
            about_results=ABOUT_STDOUT,
        )
    if not final_user.endswith("\n"):
        final_user = f"{final_user}\n"
    assert proof_call[3] == PromptMessage.text("user", final_user)
    assert proof_call[3].parts[0].cache_breakpoint is False
    assert "====== STATEMENT ======" in proof_call[3].joined_text()


def test_describe_includes_model() -> None:
    mock_model = MagicMock()
    mock_model.model = "gpt-5.6-luna"
    mock_model.reasoning_effort = "none"
    assert DirectAboutAgent(mock_model).describe() == (
        "DirectAboutAgent, model=gpt-5.6-luna reasoning=none"
    )
