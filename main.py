"""CLI entry point: run the orchestrator on a Prop-only goal (no Require Import)."""

from __future__ import annotations

from dotenv import load_dotenv

from llmprover.agent_registry import AgentRegistry
from llmprover.domain import AttemptRecord, Goal, LemmaNode
from llmprover.llm_attempt_strategy import LLMAttemptStrategy
from llmprover.llm_client import MistralAIClient
from llmprover.model_registry import ModelRegistry
from llmprover.orchestrator import Orchestrator
from llmprover.prover_agents.decomposition_agent import DecompositionAgent
from llmprover.prover_agents.direct_agent import DirectAgent
from llmprover.prover_agents.repair_direct_agent import RepairDirectAgent
from llmprover.rocq import CoqcBackend
from llmprover.simple_lemma_selection_strategy import SimpleLemmaSelectionStrategy

load_dotenv()  # .env: MISTRAL_API_KEY=...  (or: export MISTRAL_API_KEY=...)

# Intuitionistic Prop distributivity — harder than a plain ``/\`` intro, still
# usable without library imports (Prelude connectives only).
GOAL = Goal(
    name="and_or_distrib",
    statement=(
        "forall A B C : Prop, "
        "(A /\\ (B \\/ C)) <-> ((A /\\ B) \\/ (A /\\ C))"
    ),
)


def build_orchestrator() -> Orchestrator:
    agent_registry = AgentRegistry()
    agent_registry.register(DirectAgent)
    agent_registry.register(RepairDirectAgent)
    agent_registry.register(DecompositionAgent)

    model_registry = ModelRegistry()
    model_registry.register(
        MistralAIClient,
        "mistral-small-latest",
        "Fast/cheap Mistral model; prefer for simple direct proofs.",
    )
    model_registry.register(
        MistralAIClient,
        "mistral-large-latest",
        "Stronger Mistral model; use for hard goals, repair, or decomposition.",
    )

    strategy_llm = MistralAIClient.from_env(model="mistral-small-latest")
    attempt_strategy = LLMAttemptStrategy(
        agent_registry, model_registry, strategy_llm
    )
    return Orchestrator(
        attempt_strategy=attempt_strategy,
        checker=CoqcBackend(),
        lemma_selection_strategy=SimpleLemmaSelectionStrategy(),
    )


def format_record(record: AttemptRecord, indent: str) -> list[str]:
    attempt = record.attempt
    ok = "ok" if record.rocq_error.success else "fail"
    lines = [
        f"{indent}- [{attempt.polarity.value}] {ok} "
        f"new_lemmas={len(attempt.new_lemmas)}",
        f"{indent}  script: {attempt.script!r}",
    ]
    if not record.rocq_error.success and record.rocq_error.stderr.strip():
        err = record.rocq_error.stderr.strip().splitlines()[-1]
        lines.append(f"{indent}  error: {err}")
    for child in record.lemmas:
        lines.extend(format_node(child, indent + "  "))
    return lines


def format_node(node: LemmaNode, indent: str = "") -> list[str]:
    lines = [
        f"{indent}* {node.goal.name}: {node.goal.statement} "
        f"[{node.status.value}]"
    ]
    for record in node.positive + node.negative:
        lines.extend(format_record(record, indent + "  "))
    return lines


if __name__ == "__main__":
    orchestrator = build_orchestrator()
    print(f"Proving: {GOAL.statement}")
    node = orchestrator.prove(GOAL, max_attempts=20)
    print(f"Status: {node.status.value}")
    print("\n".join(format_node(node)))
