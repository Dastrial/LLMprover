"""Tests for llmprover.prover_agents.repair_direct_about_agent."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from llmprover.domain import (
    AttemptRecord,
    CoqcResult,
    Goal,
    LemmaNode,
    Polarity,
    ProofAttempt,
    RocqEnvironment,
    statement_for_polarity,
)
from llmprover.history_presenter import DeterministicHistoryPresenter
from llmprover.llm_client import CompletionResult, TokenUsage
from llmprover.prompts import (
    PromptMessage,
    PromptPart,
    fill_prompt,
)
from llmprover.prover_agents.prompt_assembly import (
    REPAIR_DIRECT_ABOUT_SPEC,
    cached_system_prompt,
)
from llmprover.prover_agents.repair_direct_about_agent import RepairDirectAboutAgent

PROMPTS_DIR = (
    Path(__file__).resolve().parent.parent / "llmprover" / "prover_agents" / "prompts"
)
GOAL = Goal(
    name="plus_n0",
    statement="forall n : nat, n + 0 = n.",
    environment=RocqEnvironment(header="Require Import Arith."),
)
SEARCH_SYSTEM = cached_system_prompt(REPAIR_DIRECT_ABOUT_SPEC)
SEARCH_HITS = "Nat.add_0_r\n"
ABOUT_STDOUT = "add_0_r : forall n : nat, n + 0 = n\n"


def _check_script(script, **_kwargs):
    if "Search" in script.code:
        return CoqcResult(success=True, stdout=SEARCH_HITS)
    return CoqcResult(success=True, stdout=ABOUT_STDOUT)


def test_prove_injects_search_then_about_results_and_histories() -> None:
    node = LemmaNode(
        goal=GOAL,
        positive=[
            AttemptRecord(
                attempt=ProofAttempt(goal=GOAL, script="induction n.", new_lemmas=[]),
                rocq_error=CoqcResult(success=False, stderr="Error."),
            )
        ],
    )
    mock_model = MagicMock()
    mock_model.complete.side_effect = [
        CompletionResult(text='Search "add" "0".\n', usage=TokenUsage(1, 1)),
        CompletionResult(text="intros n. apply Nat.add_0_r.", usage=TokenUsage(2, 3)),
    ]
    agent = RepairDirectAboutAgent(mock_model)
    agent.checker = MagicMock()
    agent.checker.check_script.side_effect = _check_script

    attempt, usage = agent.prove(node, Polarity.Positive)

    assert attempt.script == "intros n. apply Nat.add_0_r."
    assert attempt.about_lemmas == ["Nat.add_0_r"]
    assert usage == TokenUsage(3, 4)
    assert mock_model.complete.call_count == 2

    attempt_history, _ = DeterministicHistoryPresenter.repair().present_prompt_block(node)
    statement = statement_for_polarity(GOAL.statement, Polarity.Positive)
    search_call = mock_model.complete.call_args_list[0].args[0]
    before = fill_prompt(
            (PROMPTS_DIR / "repair_direct_about_search_user_before.txt")
            .read_text(encoding="utf-8")
            .strip(),
            header="Require Import Arith.",
        )
    if not before.endswith("\n"):
        before = f"{before}\n"
    after = fill_prompt(
            (PROMPTS_DIR / "repair_direct_about_search_user_after.txt")
            .read_text(encoding="utf-8")
            .strip(),
            statement=statement,
        )
    if not after.endswith("\n"):
        after = f"{after}\n"
    assert search_call[0] == PromptMessage.text("system", SEARCH_SYSTEM, cache_breakpoint=True)
    hist = attempt_history if attempt_history.endswith("\n") else attempt_history + "\n"
    assert search_call[1] == PromptMessage(
        role="user",
        parts=(PromptPart(before + hist, cache_breakpoint=True), PromptPart(after)),
    )
    assert search_call[1].parts[0].cache_breakpoint is True
    assert search_call[1].parts[1].cache_breakpoint is False
    assert "INTERACTION PROTOCOL" in SEARCH_SYSTEM

    proof_call = mock_model.complete.call_args_list[1].args[0]
    assert proof_call[0] == PromptMessage.text("system", SEARCH_SYSTEM, cache_breakpoint=True)
    assert proof_call[1] == search_call[1]
    assert proof_call[2].role == "assistant"
    assert proof_call[2].parts[0].cache_breakpoint is False
    final_user = fill_prompt(
            (PROMPTS_DIR / "repair_direct_about_proof_user.txt")
            .read_text(encoding="utf-8")
            .strip(),
            header="Require Import Arith.",
            statement=statement,
            about_results=ABOUT_STDOUT,
        )
    if not final_user.endswith("\n"):
        final_user = f"{final_user}\n"
    assert proof_call[3] == PromptMessage.text("user", final_user)
    assert proof_call[3].parts[0].cache_breakpoint is False
    assert "About" in proof_call[3].joined_text()
    assert "====== STATEMENT ======" in proof_call[3].joined_text()
