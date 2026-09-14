"""Tests for proof-prompt cache layout (PRE-PROMPT / HISTORY breakpoints)."""

from __future__ import annotations

from unittest.mock import MagicMock

from llmprover.domain import (
    AttemptRecord,
    CoqcResult,
    Goal,
    LemmaNode,
    Polarity,
    ProofAttempt,
)
from llmprover.llm.client import CompletionResult, TokenUsage
from llmprover.prover_agents.decomposition_about_agent import DecompositionAboutAgent
from llmprover.prover_agents.decomposition_agent import DecompositionAgent
from llmprover.prover_agents.direct_about_agent import DirectAboutAgent
from llmprover.prover_agents.direct_agent import DirectAgent
from llmprover.prover_agents.repair_direct_about_agent import RepairDirectAboutAgent
from llmprover.prover_agents.repair_direct_agent import RepairDirectAgent

GOAL = Goal(name="plus_n0", statement="forall n : nat, n + 0 = n.")
SEARCH_REPLY = 'Search "add".\n'
ABOUT_STDOUT = "add_0_r : forall n : nat, n + 0 = n\n"


def _record(script: str = "induction n.") -> AttemptRecord:
    return AttemptRecord(
        attempt=ProofAttempt(goal=GOAL, script=script, new_lemmas=[]),
        rocq_error=CoqcResult(success=False, stderr="Error."),
    )


def _messages(agent, node: LemmaNode, polarity: Polarity = Polarity.Positive):
    mock = MagicMock()
    mock.complete.return_value = CompletionResult(
        text="reflexivity.", usage=TokenUsage()
    )
    agent_obj = agent(mock)
    if isinstance(agent_obj, DecompositionAgent):
        mock.complete.return_value = CompletionResult(
            text="SCRIPT:\nreflexivity.", usage=TokenUsage()
        )
    agent_obj.prove(node, polarity)
    return mock.complete.call_args.args[0]


def _about_calls(
    agent_cls,
    node: LemmaNode,
    polarity: Polarity = Polarity.Positive,
    *,
    proof_text: str = "reflexivity.",
):
    """Run an about agent; return (search_messages, proof_messages)."""
    mock = MagicMock()
    mock.complete.side_effect = [
        CompletionResult(text=SEARCH_REPLY, usage=TokenUsage()),
        CompletionResult(text=proof_text, usage=TokenUsage()),
    ]
    agent = agent_cls(mock)
    agent.checker = MagicMock()

    def _check_script(script, **_kwargs):
        if "Search" in script.code:
            return CoqcResult(success=True, stdout="Nat.add_0_r\n")
        return CoqcResult(success=True, stdout=ABOUT_STDOUT)

    agent.checker.check_script.side_effect = _check_script
    agent.prove(node, polarity)
    calls = mock.complete.call_args_list
    assert len(calls) == 2
    return calls[0].args[0], calls[1].args[0]


def test_direct_caches_header_then_puts_statement_in_suffix() -> None:
    messages = _messages(DirectAgent, LemmaNode(goal=GOAL, positive=[_record()]))
    assert messages[0].role == "system"
    assert messages[0].parts[-1].cache_breakpoint is True
    parts = messages[1].parts
    assert len(parts) == 2
    assert parts[0].cache_breakpoint is True
    assert "====== STATEMENT ======" not in parts[0].text
    assert GOAL.statement not in parts[0].text
    assert parts[1].cache_breakpoint is False
    assert "====== STATEMENT ======" in parts[1].text
    assert GOAL.statement in parts[1].text


def test_repair_without_history_still_breakpoints_after_prefix() -> None:
    messages = _messages(RepairDirectAgent, LemmaNode(goal=GOAL))
    assert messages[0].parts[-1].cache_breakpoint is True
    parts = messages[1].parts
    assert len(parts) == 2  # cached prefix, then dynamic after
    assert parts[0].cache_breakpoint is True
    assert "====== STATEMENT ======" not in parts[0].text
    assert "No previous attempts" in parts[0].text
    assert parts[1].cache_breakpoint is False
    assert "====== STATEMENT ======" in parts[1].text
    assert "Output only tactics" in parts[1].text
    assert "HISTORY:(none)" not in messages[1].joined_text()


def test_repair_with_history_orders_header_history_then_statement() -> None:
    node = LemmaNode(goal=GOAL, positive=[_record()])
    messages = _messages(RepairDirectAgent, node)
    assert messages[0].parts[-1].cache_breakpoint is True
    parts = messages[1].parts
    assert parts[0].cache_breakpoint is True
    assert parts[0].text.startswith("Header (libraries")
    assert "Attempt 1:" in parts[0].text
    assert "====== STATEMENT ======" not in parts[0].text
    assert "Output only tactics" not in parts[0].text
    assert parts[1].cache_breakpoint is False
    assert parts[1].text.startswith("Write the Rocq proof body")
    assert "====== STATEMENT ======" in parts[1].text
    assert GOAL.statement in parts[1].text
    assert "Output only tactics" in parts[1].text


def test_decomposition_history_grows_as_common_prefix() -> None:
    mock = MagicMock()
    mock.complete.return_value = CompletionResult(
        text="SCRIPT:\nok.", usage=TokenUsage()
    )
    agent = DecompositionAgent(mock)
    node = LemmaNode(goal=GOAL, positive=[_record("auto.")])
    agent.prove(node, Polarity.Positive)
    prefix1 = mock.complete.call_args.args[0][1].parts[0].text

    node.positive.append(_record("lia."))
    agent.prove(node, Polarity.Positive)
    prefix2 = mock.complete.call_args.args[0][1].parts[0].text

    assert "Attempt 1:" in prefix1
    assert "auto." in prefix1
    assert "Attempt 1:" in prefix2
    assert "Attempt 2:" in prefix2
    assert "auto." in prefix2
    assert "lia." in prefix2
    assert len(prefix2) > len(prefix1)
    assert mock.complete.call_args.args[0][1].parts[0].cache_breakpoint is True
    assert GOAL.statement in mock.complete.call_args.args[0][1].parts[1].text


def test_pre_prompt_excludes_goal_specific_content() -> None:
    messages = _messages(RepairDirectAgent, LemmaNode(goal=GOAL, positive=[_record()]))
    pre = messages[0].joined_text()
    assert GOAL.statement not in pre
    assert "Attempt 1:" not in pre


def test_polarity_flip_keeps_cached_prefix() -> None:
    mock = MagicMock()
    mock.complete.return_value = CompletionResult(
        text="reflexivity.", usage=TokenUsage()
    )
    agent = RepairDirectAgent(mock)
    node = LemmaNode(goal=GOAL, positive=[_record()])
    agent.prove(node, Polarity.Positive)
    prefix_pos = mock.complete.call_args.args[0][1].parts[0].text
    suffix_pos = mock.complete.call_args.args[0][1].parts[1].text
    agent.prove(node, Polarity.Negative)
    prefix_neg = mock.complete.call_args.args[0][1].parts[0].text
    suffix_neg = mock.complete.call_args.args[0][1].parts[1].text
    assert prefix_pos == prefix_neg
    assert GOAL.statement in suffix_pos
    assert f"~ ({GOAL.statement})" in suffix_neg


def test_direct_about_search_caches_header_then_statement() -> None:
    search, proof = _about_calls(DirectAboutAgent, LemmaNode(goal=GOAL))
    assert search[0].parts[-1].cache_breakpoint is True
    parts = search[1].parts
    assert len(parts) == 2
    assert parts[0].cache_breakpoint is True
    assert "Header (already loaded" in parts[0].text
    assert "====== STATEMENT ======" not in parts[0].text
    assert GOAL.statement not in parts[0].text
    assert parts[1].cache_breakpoint is False
    assert parts[1].text.startswith("====== STATEMENT ======")
    assert GOAL.statement in parts[1].text
    # Final proof turn reuses the search prefix; about results stay uncached.
    assert proof[0] == search[0]
    assert proof[1] == search[1]
    assert proof[-1].parts[0].cache_breakpoint is False
    assert ABOUT_STDOUT.strip() in proof[-1].joined_text()
    assert "====== STATEMENT ======" in proof[-1].joined_text()


def test_repair_about_without_history_still_breakpoints_after_prefix() -> None:
    search, _ = _about_calls(RepairDirectAboutAgent, LemmaNode(goal=GOAL))
    assert search[0].parts[-1].cache_breakpoint is True
    parts = search[1].parts
    assert len(parts) == 2
    assert parts[0].cache_breakpoint is True
    assert "====== STATEMENT ======" not in parts[0].text
    assert "No previous attempts" in parts[0].text
    assert parts[1].cache_breakpoint is False
    assert "====== STATEMENT ======" in parts[1].text


def test_repair_about_with_history_orders_header_history_then_statement() -> None:
    node = LemmaNode(goal=GOAL, positive=[_record()])
    search, proof = _about_calls(RepairDirectAboutAgent, node)
    assert search[0].parts[-1].cache_breakpoint is True
    parts = search[1].parts
    assert parts[0].cache_breakpoint is True
    assert "Header (already loaded" in parts[0].text
    assert "Attempt 1:" in parts[0].text
    assert "====== STATEMENT ======" not in parts[0].text
    assert parts[1].cache_breakpoint is False
    assert parts[1].text.startswith("====== STATEMENT ======")
    assert GOAL.statement in parts[1].text
    assert proof[1] == search[1]
    assert proof[-1].parts[0].cache_breakpoint is False
    assert ABOUT_STDOUT.strip() in proof[-1].joined_text()


def test_decomposition_about_history_grows_as_common_prefix() -> None:
    mock = MagicMock()
    mock.complete.side_effect = [
        CompletionResult(text=SEARCH_REPLY, usage=TokenUsage()),
        CompletionResult(text="SCRIPT:\nok.", usage=TokenUsage()),
        CompletionResult(text=SEARCH_REPLY, usage=TokenUsage()),
        CompletionResult(text="SCRIPT:\nok.", usage=TokenUsage()),
    ]
    agent = DecompositionAboutAgent(mock)
    agent.checker = MagicMock()

    def _check_script(script, **_kwargs):
        if "Search" in script.code:
            return CoqcResult(success=True, stdout="Nat.add_0_r\n")
        return CoqcResult(success=True, stdout=ABOUT_STDOUT)

    agent.checker.check_script.side_effect = _check_script
    node = LemmaNode(goal=GOAL, positive=[_record("auto.")])
    agent.prove(node, Polarity.Positive)
    prefix1 = mock.complete.call_args_list[0].args[0][1].parts[0].text

    node.positive.append(_record("lia."))
    agent.prove(node, Polarity.Positive)
    prefix2 = mock.complete.call_args_list[2].args[0][1].parts[0].text

    assert "Attempt 1:" in prefix1
    assert "auto." in prefix1
    assert "Attempt 1:" in prefix2
    assert "Attempt 2:" in prefix2
    assert "auto." in prefix2
    assert "lia." in prefix2
    assert len(prefix2) > len(prefix1)
    assert mock.complete.call_args_list[2].args[0][1].parts[0].cache_breakpoint is True
    assert GOAL.statement in mock.complete.call_args_list[2].args[0][1].parts[1].text


def test_about_pre_prompt_excludes_goal_specific_content() -> None:
    search, _ = _about_calls(
        RepairDirectAboutAgent, LemmaNode(goal=GOAL, positive=[_record()])
    )
    pre = search[0].joined_text()
    assert GOAL.statement not in pre
    assert "Attempt 1:" not in pre


def test_about_polarity_flip_keeps_cached_search_prefix() -> None:
    mock = MagicMock()
    mock.complete.side_effect = [
        CompletionResult(text=SEARCH_REPLY, usage=TokenUsage()),
        CompletionResult(text="reflexivity.", usage=TokenUsage()),
        CompletionResult(text=SEARCH_REPLY, usage=TokenUsage()),
        CompletionResult(text="reflexivity.", usage=TokenUsage()),
    ]
    agent = RepairDirectAboutAgent(mock)
    agent.checker = MagicMock()

    def _check_script(script, **_kwargs):
        if "Search" in script.code:
            return CoqcResult(success=True, stdout="Nat.add_0_r\n")
        return CoqcResult(success=True, stdout=ABOUT_STDOUT)

    agent.checker.check_script.side_effect = _check_script
    node = LemmaNode(goal=GOAL, positive=[_record()])
    agent.prove(node, Polarity.Positive)
    search_pos = mock.complete.call_args_list[0].args[0]
    agent.prove(node, Polarity.Negative)
    search_neg = mock.complete.call_args_list[2].args[0]
    assert search_pos[1].parts[0].text == search_neg[1].parts[0].text
    assert GOAL.statement in search_pos[1].parts[1].text
    assert f"~ ({GOAL.statement})" in search_neg[1].parts[1].text
