"""Orchestrator: drive proof search over a ``LemmaNode`` tree."""

from __future__ import annotations

from llmprover.attempt_strategy import AttemptStrategy
from llmprover.domain import AttemptRecord, Goal, LemmaNode, LemmaStatus
from llmprover.lemma_selection_strategy import LemmaSelectionStrategy
from llmprover.rocq import CoqcBackend


class Orchestrator:
    """Run agent selection, proof attempts, and Rocq checks until the goal closes."""

    def __init__(
        self,
        attempt_strategy: AttemptStrategy,
        checker: CoqcBackend,
        lemma_selection_strategy: LemmaSelectionStrategy,
    ) -> None:
        self.attempt_strategy = attempt_strategy
        self.checker = checker
        self.lemma_selection_strategy = lemma_selection_strategy
        self._lemma_name_counter = 0

    def unique_name_lemma(self, lemma: Goal) -> Goal:
        self._lemma_name_counter += 1
        return Goal(
            name=f"{lemma.name}_{self._lemma_name_counter}",
            statement=lemma.statement,
        )

    def prove(
        self,
        goal: Goal,
        max_input_tokens: int = 100000,
        max_output_tokens: int = 10000,
        max_attempts: int = 200,
    ) -> LemmaNode:
        input_token_used = 0
        output_token_used = 0
        attempts_used = 0

        node = LemmaNode(goal=goal)
        while (
            input_token_used < max_input_tokens
            and output_token_used < max_output_tokens
            and attempts_used < max_attempts
            and node.status is LemmaStatus.Open
        ):
            attempts_used += 1
            position, polarity = self.lemma_selection_strategy.select_lemma()
            node_to_attack = node.from_position(position)
            prover_agent, decide_usage = self.attempt_strategy.decide_agent(
                node_to_attack, polarity
            )
            attempt, prove_usage = prover_agent.prove(node_to_attack, polarity)
            usage = decide_usage + prove_usage
            input_token_used += usage.input_tokens
            output_token_used += usage.output_tokens
            coqc_result = self.checker.check_attempt(attempt)
            new_node_lemmas = [
                LemmaNode(goal=self.unique_name_lemma(lemma))
                for lemma in attempt.new_lemmas
            ]
            record = AttemptRecord(
                attempt=attempt,
                rocq_error=coqc_result,
                lemmas=new_node_lemmas,
            )
            node_to_attack.append(record)
            self.lemma_selection_strategy.update(record, position)

        return node
