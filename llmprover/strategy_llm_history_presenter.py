"""LLM history presenter: concise failure reasons for the attempt strategy."""

from __future__ import annotations

from pathlib import Path

from llmprover.domain import AttemptRecord
from llmprover.history_presenter import HistoryPresenter
from llmprover.llm_client import LLMClient, TokenUsage
from llmprover.prompts import (
    PromptMessage,
    fill_prompt,
)
from llmprover.utils import format_attempt, normalize_llm_summary

PROMPTS_DIR = Path(__file__).parent / "prompts" / "history"


class StrategyLLMHistoryPresenter(HistoryPresenter):
    """One LLM call per new attempt; brief failure reason for agent selection."""

    def __init__(self, model: LLMClient) -> None:
        super().__init__(policy_id="llm_strategy")
        self.model = model

    def format_legend(self) -> str:
        return (
            "Previous failed attempts (both polarities, in order) are summarized below.\n"
            "Each attempt names the prover agent, then states the concrete\n"
            "problem that caused that attempt to fail."
        )

    def render(
        self,
        record: AttemptRecord,
        *,
        index: int,
    ) -> tuple[str, TokenUsage]:
        raw = format_attempt(record, index, include_agent=True)
        system = (PROMPTS_DIR / "strategy_record_system.txt").read_text(encoding="utf-8").strip()
        user = fill_prompt(
            (PROMPTS_DIR / "strategy_record_user.txt").read_text(encoding="utf-8").strip(),
            attempt_index=str(index),
            attempt=raw,
        )
        completion = self.model.complete(
            [
                PromptMessage.text("system", system, cache_breakpoint=True),
                PromptMessage.text("user", user),
            ]
        )
        explanation = normalize_llm_summary(completion.text, empty="").rstrip()
        parts = [f"Attempt {index}:"]
        agent_type = record.attempt.agent
        if agent_type:
            parts.append(f"Agent: {agent_type}")
        if explanation:
            parts.append(explanation)
        return "\n".join(parts) + "\n\n", completion.usage
