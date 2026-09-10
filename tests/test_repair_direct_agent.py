"""Tests for llmprover.prover_agents.repair_direct_agent."""

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
    REPAIR_DIRECT_SPEC,
    cached_system_prompt,
)
from llmprover.prover_agents.repair_direct_agent import RepairDirectAgent

PROMPTS_DIR = (
    Path(__file__).resolve().parent.parent / "llmprover" / "prover_agents" / "prompts"
)
GOAL = Goal(name="plus_n0", statement="forall n : nat, n + 0 = n.")
EXPECTED_SYSTEM = cached_system_prompt(REPAIR_DIRECT_SPEC)
EMPTY = "No previous attempts.\n"


def attempt_record(
    script: str,
    error: str,
    *,
    polarity: Polarity = Polarity.Positive,
    new_lemmas: list[Goal] | None = None,
) -> AttemptRecord:
    goals = new_lemmas or []
    return AttemptRecord(
        attempt=ProofAttempt(
            goal=GOAL,
            polarity=polarity,
            script=script,
            new_lemmas=goals,
        ),
        rocq_error=CoqcResult(success=False, stderr=error),
        lemmas=[LemmaNode(goal=goal) for goal in goals],
    )


def expected_messages(
    polarity: Polarity,
    node: LemmaNode | None = None,
) -> list[PromptMessage]:
    node = node or LemmaNode(goal=GOAL)
    attempt_history, _ = DeterministicHistoryPresenter.repair().present_prompt_block(
        node
    )
    before = fill_prompt(
        (PROMPTS_DIR / "repair_direct_proof_user_before.txt")
        .read_text(encoding="utf-8")
        .strip(),
        header=GOAL.environment.header,
    )
    if not before.endswith("\n"):
        before = f"{before}\n"
    after = fill_prompt(
        (PROMPTS_DIR / "repair_direct_proof_user_after.txt")
        .read_text(encoding="utf-8")
        .strip(),
        statement=statement_for_polarity(GOAL.statement, polarity),
    )
    if after and not after.endswith("\n"):
        after = f"{after}\n"
    history = (
        attempt_history if attempt_history.endswith("\n") else f"{attempt_history}\n"
    )
    return [
        PromptMessage.text("system", EXPECTED_SYSTEM, cache_breakpoint=True),
        PromptMessage(
            role="user",
            parts=(
                PromptPart(before + history, cache_breakpoint=True),
                PromptPart(after),
            ),
        ),
    ]


def test_present_prompt_block_mixes_polarities() -> None:
    node = LemmaNode(
        goal=GOAL,
        positive=[
            attempt_record("induction n.", "Error on line 1."),
            attempt_record("auto.", "Unable to unify."),
        ],
        negative=[attempt_record("intro H.", "Error B.", polarity=Polarity.Negative)],
    )

    text, _ = DeterministicHistoryPresenter.repair().present_prompt_block(node)

    assert "1 | induction n." in text
    assert "1 | auto." in text
    assert "1 | intro H." in text
    assert text.index("induction n.") < text.index("intro H.")
    assert "Attempt 3:" in text


def test_present_prompt_block_skips_decomposition_attempts() -> None:
    node = LemmaNode(
        goal=GOAL,
        positive=[
            attempt_record(
                "apply helper.",
                "Error.",
                new_lemmas=[Goal(name="helper", statement="True.")],
            ),
            attempt_record("reflexivity.", "Still failing."),
        ],
    )

    text, _ = DeterministicHistoryPresenter.repair().present_prompt_block(node)

    assert "apply helper." not in text
    assert "1 | reflexivity." in text
    assert "Attempt 2:" not in text


def test_present_prompt_block_omits_agent_description() -> None:
    node = LemmaNode(
        goal=GOAL,
        positive=[
            attempt_record(
                "induction n.",
                "Error on line 1.",
            )
        ],
    )
    node.positive[0].attempt.agent = "DirectAgent, model=gpt-4o-mini"

    text, _ = DeterministicHistoryPresenter.repair().present_prompt_block(node)

    assert "Agent:" not in text
    assert "DirectAgent" not in text


def test_present_prompt_block_returns_placeholder_when_empty() -> None:
    text, _ = DeterministicHistoryPresenter.repair().present_prompt_block(
        LemmaNode(goal=GOAL)
    )
    assert text == EMPTY


def test_prove_positive_includes_mixed_history() -> None:
    mock_model = MagicMock()
    mock_model.complete.return_value = CompletionResult(
        text="reflexivity.", usage=TokenUsage(3, 5)
    )
    node = LemmaNode(
        goal=GOAL,
        positive=[attempt_record("induction n.", "Error on line 1.")],
        negative=[attempt_record("intro H.", "Error B.", polarity=Polarity.Negative)],
    )

    attempt, usage = RepairDirectAgent(mock_model).prove(node, Polarity.Positive)

    assert attempt == ProofAttempt(
        goal=GOAL,
        polarity=Polarity.Positive,
        script="reflexivity.",
        new_lemmas=[],
    )
    assert usage == TokenUsage(3, 5)
    mock_model.complete.assert_called_once_with(
        expected_messages(Polarity.Positive, node)
    )


def test_prove_negative_keeps_same_history_order() -> None:
    mock_model = MagicMock()
    mock_model.complete.return_value = CompletionResult(
        text="intro H. contradiction.", usage=TokenUsage(3, 5)
    )
    node = LemmaNode(
        goal=GOAL,
        positive=[attempt_record("induction n.", "Error on line 1.")],
        negative=[attempt_record("intro H.", "Error B.", polarity=Polarity.Negative)],
    )

    attempt, usage = RepairDirectAgent(mock_model).prove(node, Polarity.Negative)

    assert attempt == ProofAttempt(
        goal=GOAL,
        polarity=Polarity.Negative,
        script="intro H. contradiction.",
        new_lemmas=[],
    )
    assert usage == TokenUsage(3, 5)
    mock_model.complete.assert_called_once_with(
        expected_messages(Polarity.Negative, node)
    )
    user = mock_model.complete.call_args.args[0][1].joined_text()
    assert user.index("induction n.") < user.index("intro H.")


def test_prove_strips_markdown_fences_from_llm_output() -> None:
    mock_model = MagicMock()
    mock_model.complete.return_value = CompletionResult(
        text="```\napply H.\n```", usage=TokenUsage(3, 5)
    )

    attempt, usage = RepairDirectAgent(mock_model).prove(
        LemmaNode(goal=GOAL), Polarity.Positive
    )

    assert attempt == ProofAttempt(
        goal=GOAL,
        polarity=Polarity.Positive,
        script="apply H.",
        new_lemmas=[],
    )
    assert usage == TokenUsage(3, 5)


def test_prove_handles_empty_llm_response() -> None:
    mock_model = MagicMock()
    mock_model.complete.return_value = CompletionResult(text="", usage=TokenUsage(3, 5))

    attempt, usage = RepairDirectAgent(mock_model).prove(
        LemmaNode(goal=GOAL), Polarity.Positive
    )

    assert attempt == ProofAttempt(
        goal=GOAL,
        polarity=Polarity.Positive,
        script="",
        new_lemmas=[],
    )
    assert usage == TokenUsage(3, 5)


def test_describe_includes_model_and_reasoning_effort() -> None:
    mock_model = MagicMock()
    mock_model.model = "gpt-5.6-luna"
    mock_model.reasoning_effort = "none"

    description = RepairDirectAgent(mock_model).describe(TokenUsage(3, 5, 4))

    assert description == ("RepairDirectAgent, model=gpt-5.6-luna reasoning=none")


def test_describe_omits_reasoning_effort_when_unset() -> None:
    mock_model = MagicMock()
    mock_model.model = "gpt-4o-mini"
    mock_model.reasoning_effort = None

    assert RepairDirectAgent(mock_model).describe(TokenUsage()) == (
        "RepairDirectAgent, model=gpt-4o-mini"
    )
