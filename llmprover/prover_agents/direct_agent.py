"""LLM-based agent class for direct proof"""

from __future__ import annotations

from pathlib import Path

from llmprover.domain import LemmaNode, Polarity, ProofAttempt, statement_for_polarity
from llmprover.llm_client import LLMClient, TokenUsage
from llmprover.prompts import (
    PromptMessage,
    PromptPart,
    fill_prompt,
)
from llmprover.prover_agents.prompt_assembly import DIRECT_SPEC, cached_system_prompt
from llmprover.prover_agents.prover_agent import ProverAgent
from llmprover.utils import parse_proof_script

PROMPTS_DIR = Path(__file__).parent / "prompts"


class DirectAgent(ProverAgent):
    """LLM-based agent for direct Rocq proof generation.

    Uses a fixed system prompt and adds the current lemma (at the requested
    polarity) in the user message. Previous attempts are ignored.
    """

    DEFAULT_SPEC = (
        "Writes a direct proof with no decomposition and ignores previous "
        "attempts. Cheapest agent in tokens: fixed short prompt, cost does not "
        "grow with search history."
    )

    def __init__(self, model: LLMClient) -> None:
        self.model = model

    def prove(
        self, node: LemmaNode, polarity: Polarity
    ) -> tuple[ProofAttempt, TokenUsage]:
        goal = node.goal
        system = cached_system_prompt(DIRECT_SPEC)
        before = fill_prompt(
            (PROMPTS_DIR / "direct_proof_user_before.txt").read_text(encoding="utf-8").strip(),
            header=goal.environment.header,
        )
        if not before.endswith("\n"):
            before = f"{before}\n"
        after = fill_prompt(
            (PROMPTS_DIR / "direct_proof_user_after.txt").read_text(encoding="utf-8").strip(),
            statement=statement_for_polarity(goal.statement, polarity),
        )
        if after and not after.endswith("\n"):
            after = f"{after}\n"
        completion = self.model.complete(
            [
                PromptMessage.text("system", system, cache_breakpoint=True),
                PromptMessage(
                    role="user",
                    parts=(PromptPart(before, cache_breakpoint=True), PromptPart(after)),
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
            completion.usage,
        )
