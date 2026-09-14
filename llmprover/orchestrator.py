"""Orchestrator: drive proof search over a ``LemmaNode`` tree."""

from __future__ import annotations

from llmprover.strategy.attempt_strategy import AttemptStrategy
from llmprover.rocq.coqc_output import CoqcResult
from llmprover.domain import (
    AttemptRecord,
    Goal,
    LemmaNode,
    LemmaStatus,
    Polarity,
    Position,
    ProofAttempt,
)
from llmprover.rocq.equivalence_merger import CYCLE_ERROR, EquivalenceMerger
from llmprover.rocq.helper_resolution import (
    new_helper_protocol_error,
    script_with_child_names,
)
from llmprover.history.presenter import HistoryPresenter
from llmprover.strategy.lemma_selection_strategy import LemmaSelectionStrategy
from llmprover.llm.client import TokenUsage
from llmprover.rocq.proof_script import ProofScript
from llmprover.prover_agents.prover_agent import ProverAgent
from llmprover.rocq.backend import CoqcBackend


def format_usage(usage: TokenUsage) -> str:
    """Compact token + dollar summary for orchestrator logs."""
    models = ", ".join(name or "?" for name in usage.by_model) or "-"
    return (
        f"tokens in={usage.input_tokens} "
        f"cache_write={usage.cache_write_tokens} "
        f"cache_read={usage.cache_read_tokens} "
        f"out={usage.output_tokens} "
        f"reasoning={usage.reasoning_tokens} "
        f"cost=${usage.cost_usd():.6f} "
        f"models={models}"
    )


def format_lemma_block(name: str, statement: str, script: str) -> str:
    """Render one closed lemma (``Lemma`` / ``Proof`` / ``Qed``)."""
    return f"Lemma {name}: {statement}.\nProof.\n{script}\nQed.\n\n"


def assemble_complete_proof(node: LemmaNode) -> str:
    """Build a compilable ``.v`` script for a ``Proved`` lemma.

    Walks the successful positive frontier depth-first. Each helper is
    emitted under its ``LemmaNode`` name, in front of the lemma that uses it.
    ``{helper}`` placeholders in the parent script are bound to those child
    names. An already emitted node is skipped (the graph is a DAG, not a
    cycle). Failed / superseded attempts are ignored.
    """
    if node.status is not LemmaStatus.Proved:
        raise ValueError(
            f"Cannot assemble a complete proof: status is {node.status.value}"
        )

    emitted_ids: set[int] = set()

    def emit(current: LemmaNode, lemma_name: str) -> str:
        if id(current) in emitted_ids:
            return ""
        record = current.frontier(Polarity.Positive)
        if record is None or not record.fully_succeeded:
            raise ValueError(
                f"Cannot assemble a complete proof for lemma {current.goal.name!r}"
            )
        emitted_ids.add(id(current))
        parts: list[str] = []
        for child in record.lemmas:
            parts.append(emit(child, child.goal.name))
        parts.append(
            format_lemma_block(
                lemma_name,
                record.attempt.target_statement,
                script_with_child_names(record),
            )
        )
        return "".join(parts)

    body = emit(node, node.goal.name)
    header = node.goal.environment.header.strip()
    if header:
        return f"{header}\n\n{body}".rstrip() + "\n"
    return body.rstrip() + "\n"


class Orchestrator:
    """Run agent selection, proof attempts, and Rocq checks until the goal closes."""

    def __init__(
        self,
        attempt_strategy: AttemptStrategy,
        checker: CoqcBackend,
        lemma_selection_strategy: LemmaSelectionStrategy,
        *,
        verbose: bool = False,
        print_attempts: bool = False,
        known_true: bool = False,
    ) -> None:
        self.attempt_strategy = attempt_strategy
        self.checker = checker
        self.lemma_selection_strategy = lemma_selection_strategy
        self.verbose = verbose
        self.print_attempts = print_attempts
        self.known_true = known_true
        self._used_lemma_names: set[str] = set()
        self.last_usage = TokenUsage()
        self.last_attempts_used = 0
        self.equivalence_merger = EquivalenceMerger(
            checker,
            log=self.log,
        )
        self.sync_verbose()

    def sync_verbose(self) -> None:
        """Propagate ``verbose`` to the checker and every ``HistoryPresenter``."""
        self.checker.verbose = self.verbose
        seen: set[int] = set()
        history_presenter = getattr(self.attempt_strategy, "history_presenter", None)
        if isinstance(history_presenter, HistoryPresenter):
            history_presenter.verbose = self.verbose
            seen.add(id(history_presenter))
        agent_registry = getattr(self.attempt_strategy, "agent_registry", None)
        if agent_registry is not None:
            for presenter in agent_registry.presenters.values():
                if id(presenter) not in seen:
                    presenter.verbose = self.verbose
                    seen.add(id(presenter))

    def log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    def print_attempt(self, attempt: ProofAttempt) -> None:
        if not self.verbose and not self.print_attempts:
            return
        print("[orchestrator]   agent output:", flush=True)
        if attempt.search_about_output is not None:
            print("[orchestrator]   search/about (raw LLM):", flush=True)
            text = attempt.search_about_output.strip() or "(empty)"
            for line in text.splitlines():
                print(f"[orchestrator]     {line}", flush=True)
        if attempt.select_output is not None:
            print("[orchestrator]   select (raw LLM):", flush=True)
            text = attempt.select_output.strip() or "(empty)"
            for line in text.splitlines():
                print(f"[orchestrator]     {line}", flush=True)
        if attempt.new_lemmas:
            print("[orchestrator]   new_lemmas:", flush=True)
            for lemma in attempt.new_lemmas:
                print(
                    f"[orchestrator]     {lemma.name}: {lemma.statement}",
                    flush=True,
                )
        script = attempt.script.strip() if attempt.script else "(empty)"
        print("[orchestrator]   script:", flush=True)
        for line in script.splitlines():
            print(f"[orchestrator]     {line}", flush=True)

    def unique_name_lemma(self, lemma: Goal) -> Goal:
        """Return a copy of *lemma* whose name is not already used in this search.

        Keeps the original name when it is free. Otherwise tries ``name_1``,
        ``name_2``, … Names from superseded attempts stay reserved so we do
        not have to rewrite the tree.
        """
        name = lemma.name
        suffix = 1
        while name in self._used_lemma_names:
            name = f"{lemma.name}_{suffix}"
            suffix += 1
        self._used_lemma_names.add(name)
        return Goal(
            name=name,
            statement=lemma.statement,
            environment=lemma.environment,
        )

    def log_complete_proof(self, node: LemmaNode) -> CoqcResult | None:
        """Log the reconstructed Rocq proof of a ``Proved`` node and compile it.

        Returns ``None`` when *node* is not proved. The script is printed in
        verbose mode, then checked with ``coqc``.
        """
        if node.status is not LemmaStatus.Proved:
            return None
        script = assemble_complete_proof(node)
        self.log("[orchestrator] complete proof:")
        if self.verbose:
            print(script, end="" if script.endswith("\n") else "\n", flush=True)
        result = self.checker.check_script(ProofScript(code=script))
        status = "ok" if result.success else "fail"
        self.log(f"[orchestrator] complete proof coqc={status}")
        if not result.success and result.display_stderr.strip():
            self.log("[orchestrator]   error:")
            err = result.display_stderr.rstrip("\n")
            for line in err.splitlines():
                self.log(f"[orchestrator]     {line}")
        return result

    def _select_lemma(
        self, attempts_used: int, attempts_label: str, node: LemmaNode
    ) -> tuple[Position, Polarity, LemmaNode]:
        position, polarity = self.lemma_selection_strategy.select_lemma(
            known_true=self.known_true
        )
        node_to_attack = node.from_position(position)
        self.log(
            f"[orchestrator] attempt {attempts_used}/{attempts_label} "
            f"lemma={node_to_attack.goal.name!r} "
            f"polarity={polarity.value}"
        )
        return position, polarity, node_to_attack

    def _decide_agent(
        self, node_to_attack: LemmaNode, polarity: Polarity
    ) -> tuple[ProverAgent, TokenUsage, str]:
        prover_agent, decide_usage = self.attempt_strategy.decide_agent(
            node_to_attack, polarity
        )
        agent_name = type(prover_agent).__name__
        model = getattr(prover_agent, "model", None)
        model_id = getattr(model, "model", "?")
        effort = getattr(model, "reasoning_effort", None)
        model_label = model_id if effort is None else f"{model_id} reasoning={effort}"
        self.log(
            f"[orchestrator]   decide_agent -> {agent_name} "
            f"model={model_label} "
            f"({format_usage(decide_usage)})"
        )
        self.log(f"[orchestrator]   calling {agent_name}.prove ...")
        return prover_agent, decide_usage, agent_name

    def _make_attempt(
        self,
        prover_agent: ProverAgent,
        node_to_attack: LemmaNode,
        polarity: Polarity,
        agent_name: str,
        total_usage: TokenUsage,
        decide_usage: TokenUsage,
        attempts_used: int,
    ) -> tuple[ProofAttempt, TokenUsage]:
        attempt, prove_usage = prover_agent.prove(node_to_attack, polarity)
        attempt.agent = agent_name
        if attempt.about_lemmas is not None:
            about_label = (
                ", ".join(attempt.about_lemmas) if attempt.about_lemmas else "(none)"
            )
            self.log(f"[orchestrator]   about: {about_label}")
        self.log(
            f"[orchestrator]   prove via {agent_name} "
            f"new_lemmas={len(attempt.new_lemmas)} "
            f"({format_usage(prove_usage)})"
        )
        self.print_attempt(attempt)

        total_usage = total_usage + decide_usage + prove_usage
        self.last_usage = total_usage
        self.last_attempts_used = attempts_used
        return attempt, total_usage

    def _reject_attempt(
        self,
        node_to_attack: LemmaNode,
        position: Position,
        attempt: ProofAttempt,
        error: str,
    ) -> None:
        self.log(f"[orchestrator]   {error}")
        record = AttemptRecord(
            attempt=attempt,
            rocq_error=CoqcResult(success=False, stderr=error),
            lemmas=[],
        )
        node_to_attack.append(record)
        self.lemma_selection_strategy.update(record, position)

    def _build_helper_nodes(
        self, attempt: ProofAttempt, aliases: dict[int, LemmaNode]
    ) -> tuple[list[LemmaNode], list[str]]:
        new_node_lemmas: list[LemmaNode] = []
        for index, lemma in enumerate(attempt.new_lemmas):
            source = aliases.get(index)
            if source is not None:
                self.log(
                    f"[orchestrator]   helper {lemma.name!r} aliased to "
                    f"{source.goal.name!r}"
                )
                new_node_lemmas.append(source)
            else:
                new_node_lemmas.append(LemmaNode(goal=self.unique_name_lemma(lemma)))
        return new_node_lemmas, [child.goal.name for child in new_node_lemmas]

    def _check_attempt(
        self, attempt: ProofAttempt, helper_names: list[str], total_usage: TokenUsage
    ) -> CoqcResult:
        coqc_result = self.checker.check_attempt(attempt, helper_names=helper_names)
        status = "ok" if coqc_result.success else "fail"
        self.log(f"[orchestrator]   coqc={status} total ({format_usage(total_usage)})")
        if not coqc_result.success and coqc_result.display_stderr.strip():
            self.log("[orchestrator]   error:")
            err = coqc_result.display_stderr.rstrip("\n")
            for line in err.splitlines():
                self.log(f"[orchestrator]     {line}")
        return coqc_result

    def prove(
        self,
        goal: Goal,
        max_cost_usd: float = 0.02,
        max_attempts: int | None = 200,
    ) -> LemmaNode:
        total_usage = TokenUsage()
        attempts_used = 0
        attempts_label = "unlimited" if max_attempts is None else str(max_attempts)

        node = LemmaNode(goal=goal)
        self.last_usage = total_usage
        self.last_attempts_used = attempts_used
        self._used_lemma_names = {goal.name}
        self.sync_verbose()
        self.log(
            f"[orchestrator] start goal statement={goal.statement} "
            f"max_cost_usd={max_cost_usd} max_attempts={attempts_label}"
        )
        while (
            total_usage.cost_usd() < max_cost_usd
            and (max_attempts is None or attempts_used < max_attempts)
            and node.status is LemmaStatus.Open
        ):
            attempts_used += 1

            position, polarity, node_to_attack = self._select_lemma(
                attempts_used, attempts_label, node
            )

            prover_agent, decide_usage, agent_name = self._decide_agent(
                node_to_attack, polarity
            )

            attempt, total_usage = self._make_attempt(
                prover_agent,
                node_to_attack,
                polarity,
                agent_name,
                total_usage,
                decide_usage,
                attempts_used,
            )

            protocol_error = new_helper_protocol_error(attempt)
            if protocol_error is not None:
                self._reject_attempt(node_to_attack, position, attempt, protocol_error)
                continue

            cycle, aliases = self.equivalence_merger.analyze(attempt, node, position)
            if cycle is not None:
                helper, cycle_ancestor = cycle
                error = CYCLE_ERROR.format(
                    helper=helper.name,
                    ancestor=cycle_ancestor.goal.name,
                )
                self._reject_attempt(node_to_attack, position, attempt, error)
                continue

            new_node_lemmas, helper_names = self._build_helper_nodes(attempt, aliases)

            coqc_result = self._check_attempt(attempt, helper_names, total_usage)

            record = AttemptRecord(
                attempt=attempt,
                rocq_error=coqc_result,
                lemmas=new_node_lemmas,
            )
            node_to_attack.append(record)
            self.lemma_selection_strategy.update(record, position)
            self.lemma_selection_strategy.sync_with_tree(node)

        self.log(
            f"[orchestrator] done status={node.status.value} "
            f"attempts={attempts_used} "
            f"({format_usage(total_usage)})"
        )
        return node
