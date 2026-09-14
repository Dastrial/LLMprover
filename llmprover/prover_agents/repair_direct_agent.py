"""LLM-based agent class for repairing a direct proof"""

from __future__ import annotations

from pathlib import Path

from llmprover.domain import (
    LemmaNode,
    Polarity,
    ProofAttempt,
    statement_for_polarity,
)
from llmprover.history.presenter import DeterministicHistoryPresenter, HistoryPresenter
from llmprover.llm.client import LLMClient, TokenUsage
from llmprover.llm.prompting import (
    PromptMessage,
    PromptPart,
    fill_prompt,
)
from llmprover.prover_agents.prompt_assembly import (
    REPAIR_DIRECT_SPEC,
    cached_system_prompt,
)
from llmprover.prover_agents.prover_agent import ProverAgent
from llmprover.utils import parse_proof_script

PROMPTS_DIR = Path(__file__).parent / "prompts"


class RepairDirectAgent(ProverAgent):
    """LLM-based agent for direct Rocq proof generation with history.

    Uses a fixed system prompt and includes both positive and negative attempt
    histories from the lemma node (direct attempts only).
    """

    DEFAULT_SPEC = (
        "Writes a direct proof (no new decomposition) using previous failed "
        "direct attempts on both polarities. Token cost grows with the number "
        "of failed direct attempts included in the prompt; cheaper than "
        "decomposition because history omits decomposition attempts."
    )

    def __init__(
        self,
        model: LLMClient,
        *,
        history_presenter: HistoryPresenter | None = None,
    ) -> None:
        self.model = model
        self.history_presenter = history_presenter or DeterministicHistoryPresenter.repair()

    def prove(
        self, node: LemmaNode, polarity: Polarity
    ) -> tuple[ProofAttempt, TokenUsage]:
        goal = node.goal
        attempt_history, hist_usage = self.history_presenter.present_prompt_block(node)
        system = cached_system_prompt(REPAIR_DIRECT_SPEC)
        before = fill_prompt(
            (PROMPTS_DIR / "repair_direct_proof_user_before.txt").read_text(encoding="utf-8").strip(),
            header=goal.environment.header,
        )
        if not before.endswith("\n"):
            before = f"{before}\n"
        after = fill_prompt(
            (PROMPTS_DIR / "repair_direct_proof_user_after.txt").read_text(encoding="utf-8").strip(),
            statement=statement_for_polarity(goal.statement, polarity),
        )
        if after and not after.endswith("\n"):
            after = f"{after}\n"
        history = attempt_history if attempt_history.endswith("\n") else f"{attempt_history}\n"
        completion = self.model.complete(
            [
                PromptMessage.text("system", system, cache_breakpoint=True),
                PromptMessage(
                    role="user",
                    parts=(PromptPart(before + history, cache_breakpoint=True), PromptPart(after)),
                ),
            ]
        )
        script = parse_proof_script(completion.text)
        return (
            ProofAttempt(
                goal=goal,
                polarity=polarity,
                script=script,
                new_lemmas=[],
            ),
            hist_usage + completion.usage,
        )
