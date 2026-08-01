"""LLM-based attempt strategy."""

from __future__ import annotations

from pathlib import Path

from llmprover.agent_registry import AgentRegistry
from llmprover.attempt_strategy import AttemptStrategy
from llmprover.domain import LemmaNode, Polarity, statement_for_polarity
from llmprover.llm_client import LLMClient, TokenUsage
from llmprover.model_registry import ModelRegistry
from llmprover.prompts import fill_prompt, load_prompt
from llmprover.prover_agents.prover_agent import ProverAgent
from llmprover.utils import format_histories

PROMPTS_DIR = Path(__file__).parent / "prompts"


def format_specs(specs: dict[str, str]) -> str:
    """Render ``{id: description}`` as one ``id: description`` line per entry."""
    return "\n".join(
        f"{item_id}: {description}" for item_id, description in specs.items()
    )


class LLMAttemptStrategy(AttemptStrategy):
    """LLM-based strategy: pick agent and model from catalogues for a fixed polarity."""

    def __init__(
        self,
        agent_registry: AgentRegistry,
        model_registry: ModelRegistry,
        llm_client: LLMClient,
    ) -> None:
        super().__init__(agent_registry)
        self.model_registry = model_registry
        self.llm_client = llm_client

    def parse_response(self, response: str) -> ProverAgent:
        """Parse two-line LLM output into a ready-to-use prover agent."""
        parts = [p.strip() for p in response.strip().split("\n") if p.strip()]
        if len(parts) != 2:
            raise ValueError(
                "Invalid LLM response: expected 2 lines "
                f"'agent\\nmodel', got {len(parts)} lines: {parts}"
            )
        agent_id, model_id = parts

        llm_client = self.model_registry.get(model_id)
        return self.agent_registry.get(agent_id, llm_client)

    def decide_agent(
        self, node: LemmaNode, polarity: Polarity
    ) -> tuple[ProverAgent, TokenUsage]:
        """Ask the strategy LLM which agent and model to use next at *polarity*.

        Instantiates the chosen ``ProverAgent`` with an ``LLMClient`` for the
        selected model and returns that ready-to-use agent plus token usage.
        """
        this_formula_attempts, opposite_attempts = format_histories(node)
        if polarity is Polarity.Negative:
            this_formula_attempts, opposite_attempts = (
                opposite_attempts,
                this_formula_attempts,
            )

        system_prompt = load_prompt(PROMPTS_DIR / "strategy_system.txt")
        user_prompt = fill_prompt(
            load_prompt(PROMPTS_DIR / "strategy_user.txt"),
            statement=statement_for_polarity(node.goal.statement, polarity),
            agent_specs=format_specs(self.agent_registry.specs()),
            model_specs=format_specs(self.model_registry.specs()),
            this_formula_attempts=this_formula_attempts,
            opposite_attempts=opposite_attempts,
        )
        answer = self.llm_client.complete(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        return self.parse_response(answer.text), answer.usage
