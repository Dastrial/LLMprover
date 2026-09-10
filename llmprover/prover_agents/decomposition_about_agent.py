"""Decomposition agent with multi-turn Rocq Search → About → HELPERS/SCRIPT."""

from __future__ import annotations

from pathlib import Path

from llmprover.domain import LemmaNode, Polarity, ProofAttempt, statement_for_polarity
from llmprover.history_presenter import DeterministicHistoryPresenter, HistoryPresenter
from llmprover.llm_client import LLMClient, TokenUsage
from llmprover.prompts import PromptMessage, fill_prompt
from llmprover.prover_agents.about_lookup import (
    DECOMPOSITION_ABOUT_SEARCH_USER_AFTER,
    DECOMPOSITION_ABOUT_SEARCH_USER_BEFORE,
    request_search_then_abouts,
)
from llmprover.prover_agents.decomposition_agent import parse_decomposition_answer
from llmprover.prover_agents.prompt_assembly import (
    DECOMPOSITION_ABOUT_SPEC,
    cached_system_prompt,
)
from llmprover.prover_agents.prover_agent import ProverAgent
from llmprover.rocq import CoqcBackend

PROMPTS_DIR = Path(__file__).parent / "prompts"


class DecompositionAboutAgent(ProverAgent):
    """Like ``DecompositionAgent``, with a multi-turn Search/About conversation.

    Turn 1 asks for restricted ``Search`` (name-only) and optional ``About``
    names under a system prompt that explains the full protocol. If Search
    returns too many hits, turn 2 asks the model to keep at most 10 names.
    About is run on explicit names plus selected hits; the final turn reuses
    the conversation through the first reply and requests HELPERS/SCRIPT with
    those About results.
    """

    DEFAULT_SPEC = (
        "Decomposes the goal into helper lemmas using the full attempt history "
        "(direct and decomposition, both polarities), after a multi-turn Rocq "
        "Search/About lookup of candidate library lemmas. Costs two or three "
        "LLM calls plus coqc Search and About scripts per attempt; prompt size "
        "still grows with failed attempts."
    )

    def __init__(
        self,
        model: LLMClient,
        *,
        history_presenter: HistoryPresenter | None = None,
    ) -> None:
        self.model = model
        self.checker = CoqcBackend()
        self.history_presenter = (
            history_presenter or DeterministicHistoryPresenter.full()
        )

    def prove(
        self, node: LemmaNode, polarity: Polarity
    ) -> tuple[ProofAttempt, TokenUsage]:
        goal = node.goal
        statement = statement_for_polarity(goal.statement, polarity)
        attempt_history, hist_usage = self.history_presenter.present_prompt_block(node)
        lookup = request_search_then_abouts(
            self.model,
            self.checker,
            system_prompt=cached_system_prompt(DECOMPOSITION_ABOUT_SPEC),
            user_before_prompt=DECOMPOSITION_ABOUT_SEARCH_USER_BEFORE,
            user_after_prompt=DECOMPOSITION_ABOUT_SEARCH_USER_AFTER,
            header=goal.environment.header,
            statement=statement,
            attempt_history=attempt_history,
        )
        final_user = fill_prompt(
            (PROMPTS_DIR / "decomposition_about_user.txt")
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
        new_lemmas, script = parse_decomposition_answer(
            completion.text, environment=goal.environment
        )
        return (
            ProofAttempt(
                goal=goal,
                polarity=polarity,
                script=script,
                new_lemmas=new_lemmas,
                about_lemmas=lookup.lemma_names,
                search_about_output=lookup.search_output,
                select_output=lookup.select_output,
            ),
            hist_usage + lookup.usage + completion.usage,
        )
