"""LLM history presenter: detailed summaries for repair and decomposition."""

from __future__ import annotations

from pathlib import Path

from llmprover.domain import AttemptRecord
from llmprover.history.presenter import HistoryPresenter
from llmprover.llm.client import LLMClient, TokenUsage
from llmprover.llm.prompting import (
    PromptMessage,
    fill_prompt,
)
from llmprover.utils import format_attempt, normalize_llm_summary

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts" / "history"


class DetailedLLMHistoryPresenter(HistoryPresenter):
    """One LLM call per new attempt; detailed summary of that attempt only."""

    def __init__(self, model: LLMClient) -> None:
        super().__init__(policy_id="llm_detailed")
        self.model = model

    def format_legend(self) -> str:
        return (
            "Previous failed attempts (both polarities, in order) are summarized below.\n"
            "Each entry describes one past attempt: Rocq outcome, any apparent\n"
            "mathematical mistake, and decomposition status when helpers were\n"
            "introduced."
        )

    def render(
        self,
        record: AttemptRecord,
        *,
        index: int,
    ) -> tuple[str, TokenUsage]:
        raw = format_attempt(record, index)
        system = (PROMPTS_DIR / "detailed_record_system.txt").read_text(encoding="utf-8").strip()
        user = fill_prompt(
            (PROMPTS_DIR / "detailed_record_user.txt").read_text(encoding="utf-8").strip(),
            attempt=raw,
        )
        completion = self.model.complete(
            [
                PromptMessage.text("system", system, cache_breakpoint=True),
                PromptMessage.text("user", user),
            ]
        )
        summary = normalize_llm_summary(completion.text, empty="").rstrip()
        parts = [f"Attempt {index}:"]
        if summary:
            parts.append(summary)
        return "\n".join(parts) + "\n\n", completion.usage
