"""Direct proof agent with multi-turn Rocq Search → About → proof."""

from __future__ import annotations

from pathlib import Path

from llmprover.domain import LemmaNode, Polarity, ProofAttempt, statement_for_polarity
from llmprover.llm.client import LLMClient, TokenUsage
from llmprover.llm.prompting import PromptMessage, fill_prompt
from llmprover.prover_agents.about_lookup import (
    DIRECT_ABOUT_SEARCH_USER_AFTER,
    DIRECT_ABOUT_SEARCH_USER_BEFORE,
    request_search_then_abouts,
)
from llmprover.prover_agents.prompt_assembly import (
    DIRECT_ABOUT_SPEC,
    cached_system_prompt,
)
from llmprover.prover_agents.prover_agent import ProverAgent
from llmprover.rocq.backend import CoqcBackend
from llmprover.utils import parse_proof_script

PROMPTS_DIR = Path(__file__).parent / "prompts"


class DirectAboutAgent(ProverAgent):
    """Like ``DirectAgent``, with a multi-turn Search/About conversation.

    Turn 1 asks for restricted ``Search`` (name-only) and optional ``About``
    names under a system prompt that explains the full protocol. If Search
    returns too many hits, turn 2 asks the model to keep at most 10 names.
    About is run on explicit names plus selected hits; the final turn reuses
    the conversation through the first reply and requests the direct proof with
    those About results. Previous attempts are ignored.
    """

    DEFAULT_SPEC = (
        "Writes a direct proof with no decomposition after a multi-turn Rocq "
        "Search/About lookup of candidate library lemmas. Ignores previous "
        "attempts. Costs two or three LLM calls plus coqc Search and About "
        "scripts per attempt."
    )

    def __init__(self, model: LLMClient) -> None:
        self.model = model
        self.checker = CoqcBackend()

    def prove(
        self, node: LemmaNode, polarity: Polarity
    ) -> tuple[ProofAttempt, TokenUsage]:
        goal = node.goal
        statement = statement_for_polarity(goal.statement, polarity)
        lookup = request_search_then_abouts(
            self.model,
            self.checker,
            system_prompt=cached_system_prompt(DIRECT_ABOUT_SPEC),
            user_before_prompt=DIRECT_ABOUT_SEARCH_USER_BEFORE,
            user_after_prompt=DIRECT_ABOUT_SEARCH_USER_AFTER,
            header=goal.environment.header,
            statement=statement,
        )
        final_user = fill_prompt(
            (PROMPTS_DIR / "direct_about_proof_user.txt")
            .read_text(encoding="utf-8")
            .strip(),
            header=goal.environment.header,
            statement=statement,
            about_results=lookup.about_results,
        )
        if not final_user.endswith("\n"):
            final_user = f"{final_user}\n"
        completion = self.model.complete(
            [*lookup.prefix_messages, PromptMessage.text("user", final_user)]
        )
        script = parse_proof_script(completion.text)
        return (
            ProofAttempt(
                goal=goal,
                polarity=polarity,
                script=script,
                new_lemmas=[],
                about_lemmas=lookup.lemma_names,
                search_about_output=lookup.search_output,
                select_output=lookup.select_output,
            ),
            lookup.usage + completion.usage,
        )
