"""LLM-based attempt strategy."""

from __future__ import annotations

from pathlib import Path

from llmprover.prover_agents.agent_registry import AgentRegistry
from llmprover.strategy.attempt_strategy import AttemptStrategy
from llmprover.domain import LemmaNode, Polarity, statement_for_polarity
from llmprover.history.presenter import DeterministicHistoryPresenter, HistoryPresenter
from llmprover.llm.client import LLMClient, TokenUsage
from llmprover.llm.model_registry import ModelRegistry
from llmprover.llm.prompting import (
    PromptMessage,
    PromptPart,
    fill_prompt,
)
from llmprover.prover_agents.prover_agent import ProverAgent
from llmprover.utils import strip_markdown_fences

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# If one prover has this many more calls than every other on the current
# lemma, the next choice is forced onto a less-used agent.  The deliberately
# high value makes automatic swaps practically unreachable in the miniF2F
# runs, leaving agent choice to the LLM strategy selector.
MAX_AGENT_LEAD = 1000


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
        *,
        history_presenter: HistoryPresenter | None = None,
    ) -> None:
        super().__init__(agent_registry)
        self.model_registry = model_registry
        self.llm_client = llm_client
        self.history_presenter = (
            history_presenter or DeterministicHistoryPresenter.strategy()
        )

    def parse_response(self, response: str) -> ProverAgent:
        """Parse LLM output into a ready-to-use prover agent.

        Looks for a known agent id (or class name) and model id in the reply.
        Missing or unknown ids fall back to the first entry of each catalogue
        so a malformed strategy answer still produces an attempt (and history).
        """
        tokens = strip_markdown_fences(response).split()

        agent_ids = list(self.agent_registry.classes)
        model_ids = list(self.model_registry.specs())
        if not agent_ids or not model_ids:
            raise RuntimeError(
                "Cannot select a prover agent: agent or model registry is empty"
            )

        agent_lookup: dict[str, str] = {}
        for registered_id, cls in self.agent_registry.classes.items():
            agent_lookup[registered_id] = registered_id
            agent_lookup[cls.__name__] = registered_id
        model_lookup = set(model_ids)
        overlap = agent_lookup.keys() & model_lookup
        if overlap:
            raise ValueError(
                f"Agent identifiers/class names overlap with model identifiers: "
                f"{sorted(overlap)}"
            )
        agent_id = next(
            (agent_lookup[token] for token in tokens if token in agent_lookup),
            agent_ids[0],
        )
        model_id = next(
            (token for token in tokens if token in model_lookup),
            model_ids[0],
        )
        return self.agent_registry.get(agent_id, self.model_registry.get(model_id))

    def rebalance_agent(self, agent: ProverAgent, node: LemmaNode) -> ProverAgent:
        """Force another prover if *agent* already leads by ``MAX_AGENT_LEAD``.

        Counts come from ``node.agent_call_counts()`` (class names stored on
        each ``ProofAttempt.agent``). The least-used other registered agent
        is chosen; the model wired on *agent* is kept.
        """
        by_name = {
            cls.__name__: agent_id
            for agent_id, cls in self.agent_registry.classes.items()
        }
        if len(by_name) < 2:
            return agent

        counts = {name: 0 for name in by_name}
        for name, n in node.agent_call_counts().items():
            if name in counts:
                counts[name] += n

        chosen = type(agent).__name__
        if chosen not in counts:
            return agent
        others = [name for name in counts if name != chosen]
        if counts[chosen] - max(counts[name] for name in others) < MAX_AGENT_LEAD:
            return agent

        least = min(others, key=lambda name: counts[name])
        model = getattr(agent, "model", None)
        if model is None:
            return agent
        return self.agent_registry.get(by_name[least], model)

    def decide_agent(
        self, node: LemmaNode, polarity: Polarity
    ) -> tuple[ProverAgent, TokenUsage]:
        """LLM-based selection of an agent and model for a fixed polarity.

        The default system prompt is tailored to RepairDirectAboutAgent and
        DecompositionAboutAgent. Using a different agent catalogue requires
        adapting prompts/strategy_system.txt.
        """
        attempt_history, hist_usage = self.history_presenter.present_prompt_block(node)

        system_prompt = (
            (PROMPTS_DIR / "strategy_system.txt").read_text(encoding="utf-8").strip()
        )
        before = fill_prompt(
            (PROMPTS_DIR / "strategy_user_before.txt")
            .read_text(encoding="utf-8")
            .strip(),
            header=node.goal.environment.header,
            agent_specs=format_specs(self.agent_registry.specs()),
            model_specs=format_specs(self.model_registry.specs()),
        )
        if not before.endswith("\n"):
            before = f"{before}\n"
        after = fill_prompt(
            (PROMPTS_DIR / "strategy_user_after.txt")
            .read_text(encoding="utf-8")
            .strip(),
            statement=statement_for_polarity(node.goal.statement, polarity),
        )
        if after and not after.endswith("\n"):
            after = f"{after}\n"
        history = (
            attempt_history
            if attempt_history.endswith("\n")
            else f"{attempt_history}\n"
        )
        answer = self.llm_client.complete(
            [
                PromptMessage.text("system", system_prompt, cache_breakpoint=True),
                PromptMessage(
                    role="user",
                    parts=(
                        PromptPart(before + history, cache_breakpoint=True),
                        PromptPart(after),
                    ),
                ),
            ]
        )
        return self.rebalance_agent(
            self.parse_response(answer.text), node
        ), hist_usage + answer.usage
