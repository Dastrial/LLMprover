"""Tests for llmprover.history_presenter."""

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
from llmprover.history_presenter import (
    FULL_FORMAT_LEGEND,
    DeterministicHistoryPresenter,
    NullHistoryPresenter,
)
from llmprover.llm_client import TokenUsage
from llmprover.utils import EMPTY_ATTEMPTS

GOAL = Goal(name="plus_n0", statement="forall n : nat, n + 0 = n.")


def attempt_record(
    script: str,
    error: str,
    *,
    polarity: Polarity = Polarity.Positive,
    new_lemmas: list[Goal] | None = None,
    agent: str = "",
) -> AttemptRecord:
    goals = new_lemmas or []
    return AttemptRecord(
        attempt=ProofAttempt(
            goal=GOAL,
            polarity=polarity,
            script=script,
            new_lemmas=goals,
            agent=agent,
        ),
        rocq_error=CoqcResult(success=False, stderr=error),
        lemmas=[LemmaNode(goal=goal) for goal in goals],
    )


def test_null_presenter_ignores_records() -> None:
    node = LemmaNode(
        goal=GOAL,
        positive=[attempt_record("auto.", "fail")],
    )
    text, usage = NullHistoryPresenter().present_prompt_block(node)
    assert text == EMPTY_ATTEMPTS
    assert usage == TokenUsage()


def test_repair_presenter_skips_decomposition_attempts() -> None:
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
    text, usage = DeterministicHistoryPresenter.repair().present_prompt_block(node)
    assert "apply helper." not in text
    assert "reflexivity." in text
    assert usage == TokenUsage()


def test_strategy_presenter_includes_agent() -> None:
    node = LemmaNode(
        goal=GOAL,
        positive=[
            attempt_record(
                "auto.",
                "fail",
                agent="DirectAgent, model=gpt-4o-mini",
            )
        ],
    )
    text, _ = DeterministicHistoryPresenter.strategy().present_prompt_block(node)
    assert "Agent: DirectAgent, model=gpt-4o-mini" in text


def test_cache_reuses_text_when_history_unchanged() -> None:
    presenter = DeterministicHistoryPresenter.full()
    presenter.render = MagicMock(  # type: ignore[method-assign]
        wraps=presenter.render
    )
    node = LemmaNode(
        goal=GOAL,
        positive=[attempt_record("auto.", "fail")],
    )

    first, _ = presenter.present_prompt_block(node)
    second, usage = presenter.present_prompt_block(node)

    assert first == second
    assert usage == TokenUsage()
    assert presenter.render.call_count == 1
    assert first.startswith(FULL_FORMAT_LEGEND)


def test_cache_invalidates_when_history_grows() -> None:
    presenter = DeterministicHistoryPresenter.full()
    presenter.render = MagicMock(  # type: ignore[method-assign]
        wraps=presenter.render
    )
    node = LemmaNode(
        goal=GOAL,
        positive=[attempt_record("auto.", "fail")],
    )

    presenter.present_prompt_block(node)
    node.positive.append(attempt_record("induction n.", "fail2"))
    text, _ = presenter.present_prompt_block(node)

    assert presenter.render.call_count == 2
    assert "induction n." in text
    assert "auto." in text
    assert text.startswith(FULL_FORMAT_LEGEND)


def test_incremental_cache_only_renders_new_records() -> None:
    presenter = DeterministicHistoryPresenter.full()
    presenter.render = MagicMock(wraps=presenter.render)  # type: ignore[method-assign]
    node = LemmaNode(
        goal=GOAL,
        positive=[attempt_record("auto.", "fail")],
    )

    presenter.present_prompt_block(node)
    node.positive.append(attempt_record("induction n.", "fail2"))
    presenter.present_prompt_block(node)

    assert presenter.render.call_count == 2
    assert presenter.render.call_args_list[1].kwargs["index"] == 2
    assert "previous_summary" not in presenter.render.call_args_list[1].kwargs


def test_shared_presenter_memory_survives_across_agent_instances() -> None:
    shared = DeterministicHistoryPresenter.repair()
    shared.render = MagicMock(wraps=shared.render)  # type: ignore[method-assign]
    node = LemmaNode(
        goal=GOAL,
        positive=[attempt_record("auto.", "fail")],
    )

    from llmprover.prover_agents.repair_direct_about_agent import RepairDirectAboutAgent
    from llmprover.prover_agents.repair_direct_agent import RepairDirectAgent

    agent_a = RepairDirectAgent(MagicMock(), history_presenter=shared)
    agent_b = RepairDirectAboutAgent(MagicMock(), history_presenter=shared)

    agent_a.history_presenter.present_prompt_block(node)
    agent_b.history_presenter.present_prompt_block(node)

    assert shared.render.call_count == 1
    agent_a.history_presenter.present_prompt_block(node)
    assert shared.render.call_count == 1


def test_mixed_polarities_append_positive_then_negative_on_cold_start() -> None:
    node = LemmaNode(
        goal=GOAL,
        positive=[attempt_record("pos.", "e1")],
        negative=[attempt_record("neg.", "e2", polarity=Polarity.Negative)],
    )
    text, _ = DeterministicHistoryPresenter.full().present_prompt_block(node)
    assert text.index("1 | pos.") < text.index("1 | neg.")
    assert "Attempt 1:" in text
    assert "Attempt 2:" in text


def test_new_negative_appends_after_existing_positive() -> None:
    presenter = DeterministicHistoryPresenter.full()
    node = LemmaNode(
        goal=GOAL,
        positive=[attempt_record("pos.", "e1")],
    )
    first, _ = presenter.present_prompt_block(node)
    node.negative.append(attempt_record("neg.", "e2", polarity=Polarity.Negative))
    second, _ = presenter.present_prompt_block(node)

    assert second.startswith(first)
    assert second.index("1 | pos.") < second.index("1 | neg.")


def test_interleaved_polarities_append_and_previous_block_is_prefix() -> None:
    """Attempts append in arrival order when polarities alternate; old text is a prefix."""
    presenter = DeterministicHistoryPresenter.full()
    node = LemmaNode(goal=GOAL)
    previous = ""

    steps = [
        (Polarity.Positive, "pos1."),
        (Polarity.Negative, "neg1."),
        (Polarity.Positive, "pos2."),
        (Polarity.Negative, "neg2."),
    ]
    for polarity, script in steps:
        record = attempt_record(script, f"err-{script}", polarity=polarity)
        if polarity is Polarity.Positive:
            node.positive.append(record)
        else:
            node.negative.append(record)
        text, _ = presenter.present_prompt_block(node)
        assert text.startswith(previous)
        previous = text

    assert previous.index("1 | pos1.") < previous.index("1 | neg1.")
    assert previous.index("1 | neg1.") < previous.index("1 | pos2.")
    assert previous.index("1 | pos2.") < previous.index("1 | neg2.")
    for n in range(1, 5):
        assert f"Attempt {n}:" in previous


def test_log_render_prints_full_summary_when_verbose(capsys) -> None:
    presenter = DeterministicHistoryPresenter.full()
    presenter.verbose = True
    node = LemmaNode(
        goal=GOAL,
        positive=[attempt_record("auto.", "fail")],
    )

    presenter.present_prompt_block(node)

    output = capsys.readouterr().out
    assert "history summary (full, attempt 1):" in output
    assert "1 | auto." in output


def test_log_render_silent_when_not_verbose(capsys) -> None:
    presenter = DeterministicHistoryPresenter.full()
    node = LemmaNode(
        goal=GOAL,
        positive=[attempt_record("auto.", "fail")],
    )

    presenter.present_prompt_block(node)

    assert capsys.readouterr().out == ""


def test_present_prompt_block_seeds_legend_then_attempts() -> None:
    node = LemmaNode(
        goal=GOAL,
        positive=[attempt_record("auto.", "fail")],
    )
    block, usage = DeterministicHistoryPresenter.full().present_prompt_block(node)
    assert block.startswith(FULL_FORMAT_LEGEND + "\n\n")
    assert "1 | auto." in block
    assert usage == TokenUsage()


def test_empty_node_returns_placeholder_without_legend() -> None:
    block, usage = DeterministicHistoryPresenter.full().present_prompt_block(
        LemmaNode(goal=GOAL)
    )
    assert block == EMPTY_ATTEMPTS
    assert usage == TokenUsage()


def test_presenter_includes_empty_script_attempts() -> None:
    node = LemmaNode(
        goal=GOAL,
        positive=[attempt_record("", "Error: incomplete proof.")],
    )
    block, usage = DeterministicHistoryPresenter.full().present_prompt_block(node)
    assert "Script: (empty)" in block
    assert "Error: incomplete proof." in block
    assert usage == TokenUsage()
