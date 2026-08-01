"""LLM-based Proof agent class for decomposition"""

from __future__ import annotations

from pathlib import Path

from llmprover.domain import (
    Goal,
    LemmaNode,
    Polarity,
    ProofAttempt,
    statement_for_polarity,
)
from llmprover.llm_client import LLMClient, TokenUsage
from llmprover.prompts import fill_prompt, load_prompt
from llmprover.prover_agents.prover_agent import ProverAgent
from llmprover.utils import (
    format_histories,
    strip_markdown_fences,
    strip_proof_wrappers,
)

PROMPTS_DIR = Path(__file__).parent / "prompts"


def parse_decomposition_answer(answer: str) -> tuple[list[Goal], str]:
    """Parse LLM output: helper lemmas, blank line, then tactic script."""
    text = strip_markdown_fences(answer.strip())
    if "\n\n" in text:
        lemma_section, script = text.split("\n\n", 1)
    else:
        raise ValueError("No proof script found")

    new_lemmas: list[Goal] = []
    for line in lemma_section.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            raise ValueError("No column found in lemma script")
        name, _, statement = line.partition(":")
        name = name.strip()
        statement = statement.strip()
        if name and statement:
            new_lemmas.append(Goal(statement=statement, name=name))
        else:
            raise ValueError("No statement or name found in lemma script")
    return new_lemmas, strip_proof_wrappers(script)


class DecompositionAgent(ProverAgent):
    """LLM-based agent for decomposing a goal into lemmas.

    Includes the full attempt history (direct and decomposition attempts) for
    both polarities, so the model can avoid repeating failed strategies.
    """

    DEFAULT_SPEC = (
        "Decomposes the goal into helper lemmas using the full attempt history "
        "(direct and decomposition, both polarities), including helper-lemma "
        "statuses. Most token-hungry agent: prompt size grows with every failed "
        "attempt and with the lemmas attached to decompositions."
    )

    def __init__(self, model: LLMClient) -> None:
        self.model = model

    def prove(
        self, node: LemmaNode, polarity: Polarity
    ) -> tuple[ProofAttempt, TokenUsage]:
        goal = node.goal
        this_formula_attempts, opposite_attempts = format_histories(node)
        if polarity is Polarity.Negative:
            this_formula_attempts, opposite_attempts = (
                opposite_attempts,
                this_formula_attempts,
            )
        system = load_prompt(PROMPTS_DIR / "decomposition_system.txt")
        user = fill_prompt(
            load_prompt(PROMPTS_DIR / "decomposition_user.txt"),
            statement=statement_for_polarity(goal.statement, polarity),
            this_formula_attempts=this_formula_attempts,
            opposite_attempts=opposite_attempts,
        )
        completion = self.model.complete(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )
        new_lemmas, script = parse_decomposition_answer(completion.text)
        return (
            ProofAttempt(
                goal=goal,
                polarity=polarity,
                script=script,
                new_lemmas=new_lemmas,
            ),
            completion.usage,
        )
