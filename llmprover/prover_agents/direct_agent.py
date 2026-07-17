"""LLM-based agent class for direct proof"""

from __future__ import annotations

from pathlib import Path

from llmprover.domain import LemmaNode, Polarity, ProofAttempt, statement_for_polarity
from llmprover.llm_client import LLMClient
from llmprover.prompts import fill_prompt, load_prompt
from llmprover.prover_agents.prover_agent import ProverAgent
from llmprover.prover_agents.utils import parse_proof_script

PROMPTS_DIR = Path(__file__).parent / "prompts"


class DirectAgent(ProverAgent):
    """LLM-based agent for direct Rocq proof generation.

    Uses a fixed system prompt and adds the current lemma (at the requested
    polarity) in the user message. Previous attempts are ignored.
    """

    def __init__(self, model: LLMClient) -> None:
        self.model = model

    def prove(self, node: LemmaNode, polarity: Polarity) -> ProofAttempt:
        goal = node.goal
        system = load_prompt(PROMPTS_DIR / "direct_proof_system.txt")
        user = fill_prompt(
            load_prompt(PROMPTS_DIR / "direct_proof_user.txt"),
            statement=statement_for_polarity(goal.statement, polarity),
        )
        script = parse_proof_script(
            self.model.complete(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
            )
        )
        return ProofAttempt(
            goal=goal,
            polarity=polarity,
            script=script,
            new_lemmas=[],
        )
