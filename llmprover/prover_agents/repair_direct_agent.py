"""LLM-based agent class for repairing a direct proof"""

from __future__ import annotations

from pathlib import Path

from llmprover.domain import (
    AttemptRecord,
    LemmaNode,
    Polarity,
    ProofAttempt,
    statement_for_polarity,
)
from llmprover.llm_client import LLMClient
from llmprover.prompts import fill_prompt, load_prompt
from llmprover.prover_agents.prover_agent import ProverAgent
from llmprover.prover_agents.utils import format_attempts, parse_proof_script

PROMPTS_DIR = Path(__file__).parent / "prompts"
EMPTY_ATTEMPTS = "No previous attempts.\n"


class RepairDirectAgent(ProverAgent):
    """LLM-based agent for direct Rocq proof generation with history.

    Uses a fixed system prompt and includes both positive and negative attempt
    histories from the lemma node.
    """

    def __init__(self, model: LLMClient) -> None:
        self.model = model

    @staticmethod
    def direct_records(records: list[AttemptRecord]) -> list[AttemptRecord]:
        return [record for record in records if not record.attempt.new_lemmas]

    @classmethod
    def format_histories(cls, node: LemmaNode) -> tuple[str, str]:
        return (
            format_attempts(
                cls.direct_records(node.positive),
                empty=EMPTY_ATTEMPTS,
            ),
            format_attempts(
                cls.direct_records(node.negative),
                empty=EMPTY_ATTEMPTS,
            ),
        )

    def prove(self, node: LemmaNode, polarity: Polarity) -> ProofAttempt:
        goal = node.goal
        this_formula_attempts, opposite_attempts = self.format_histories(node)
        if polarity is Polarity.Negative:
            this_formula_attempts, opposite_attempts = (
                opposite_attempts,
                this_formula_attempts,
            )
        system = load_prompt(PROMPTS_DIR / "direct_proof_system.txt")
        user = fill_prompt(
            load_prompt(PROMPTS_DIR / "repair_direct_proof_user.txt"),
            statement=statement_for_polarity(goal.statement, polarity),
            this_formula_attempts=this_formula_attempts,
            opposite_attempts=opposite_attempts,
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
