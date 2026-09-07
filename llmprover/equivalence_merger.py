"""Detect convertibility-equivalent lemmas in the search tree."""

from __future__ import annotations

from collections.abc import Callable

from llmprover.domain import (
    Goal,
    LemmaNode,
    LemmaStatus,
    Position,
    ProofAttempt,
)
from llmprover.rocq import CoqcBackend

CYCLE_ERROR = (
    "Dropped attempt: helper {helper} is equivalent to ancestor {ancestor} "
    "(cycle up to reflexivity)."
)


class EquivalenceMerger:
    """Drop cyclic helpers and point incomparable duplicates at existing nodes."""

    def __init__(
        self,
        checker: CoqcBackend,
        *,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.checker = checker
        self.log = log if log is not None else (lambda _message: None)

    def analyze(
        self,
        attempt: ProofAttempt,
        root: LemmaNode,
        position: Position,
    ) -> tuple[tuple[Goal, LemmaNode] | None, dict[int, LemmaNode]]:
        """Match ``attempt.new_lemmas`` against every lemma node in *root*.

        Returns ``(cycle, aliases)``. *cycle* is ``(helper, ancestor)`` when a
        helper is equivalent to a node that can reach the node at *position*
        (that node, its tree ancestors, and nodes that became ancestors
        via alias links). Otherwise *aliases* maps helper index to an
        incomparable equivalent node (``Proved`` preferred).
        """
        ancestors = root.nodes_reaching(root.from_position(position))
        environment = attempt.goal.environment
        aliases: dict[int, LemmaNode] = {}
        for index, helper in enumerate(attempt.new_lemmas):
            matches = [
                node
                for node in root.walk()
                if self.checker.statements_equivalent(
                    helper.statement, node.goal.statement, environment
                )
            ]
            cycle_node = next((node for node in matches if id(node) in ancestors), None)
            if cycle_node is not None:
                return (helper, cycle_node), {}
            incomparable = [node for node in matches if id(node) not in ancestors]
            if incomparable:
                incomparable.sort(
                    key=lambda node: 0 if node.status is LemmaStatus.Proved else 1
                )
                aliases[index] = incomparable[0]
        return None, aliases
