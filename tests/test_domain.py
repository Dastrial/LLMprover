"""Tests for llmprover.domain."""

from __future__ import annotations

import pytest

from llmprover.rocq.coqc_output import CoqcResult
from llmprover.domain import (
    AttemptRecord,
    Goal,
    LemmaNode,
    LemmaStatus,
    Polarity,
    ProofAttempt,
    RocqEnvironment,
    statement_for_polarity,
    status_from_frontiers,
)

GOAL = Goal(name="plus_n0", statement="forall n : nat, n + 0 = n.")


def record(
    script: str,
    *,
    success: bool,
    polarity: Polarity = Polarity.Positive,
    error: str = "Error.",
    goal: Goal = GOAL,
) -> AttemptRecord:
    return AttemptRecord(
        attempt=ProofAttempt(
            goal=goal,
            script=script,
            new_lemmas=[],
            polarity=polarity,
        ),
        rocq_error=CoqcResult(
            success=success,
            stderr="" if success else error,
        ),
    )


def decomposition(
    goal: Goal,
    helpers: list[Goal],
    children: list[LemmaNode],
    *,
    polarity: Polarity = Polarity.Positive,
    success: bool = True,
) -> AttemptRecord:
    return AttemptRecord(
        attempt=ProofAttempt(
            goal=goal,
            script="split.",
            new_lemmas=helpers,
            polarity=polarity,
        ),
        rocq_error=CoqcResult(
            success=success,
            stderr="" if success else "Error.",
        ),
        lemmas=children,
    )


def prove(node: LemmaNode) -> None:
    node.append(record("exact I.", success=True, goal=node.goal))


def refute(node: LemmaNode) -> None:
    node.append(
        record(
            "intro H.",
            success=True,
            polarity=Polarity.Negative,
            goal=node.goal,
        )
    )


def three_child_parent(
    polarity: Polarity,
) -> tuple[LemmaNode, list[LemmaNode]]:
    helpers = [Goal(name=f"h{i}", statement="True.") for i in range(3)]
    children = [LemmaNode(goal=goal) for goal in helpers]
    root = LemmaNode(goal=GOAL)
    root.append(decomposition(GOAL, helpers, children, polarity=polarity))
    return root, children


def build_wide_depth2_tree() -> tuple[LemmaNode, dict[str, LemmaNode]]:
    """Wide depth-2 tree with both polarities.

    Root positive frontier has 2 children (``h0``, ``h1``); root negative
    frontier has 3 (``n0``, ``n1``, ``n2``). So index 2 is valid only on the
    negative side. Mid-level nodes also have mixed-polarity grandchildren.
    """
    nodes: dict[str, LemmaNode] = {}

    for i in range(2):
        for j in range(3):
            goal = Goal(name=f"h{i}_{j}", statement="True.")
            nodes[f"h{i}_{j}"] = LemmaNode(goal=goal)
        for j in range(2):
            goal = Goal(name=f"hn{i}_{j}", statement="False.")
            nodes[f"hn{i}_{j}"] = LemmaNode(goal=goal)

    for i in range(3):
        for j in range(3):
            goal = Goal(name=f"n{i}_{j}", statement="True.")
            nodes[f"n{i}_{j}"] = LemmaNode(goal=goal)

    h_goals: list[Goal] = []
    h_nodes: list[LemmaNode] = []
    for i in range(2):
        goal = Goal(name=f"h{i}", statement=f"pos_stmt_{i}.")
        node = LemmaNode(goal=goal)
        nodes[f"h{i}"] = node
        h_goals.append(goal)
        h_nodes.append(node)
        pos_goals = [nodes[f"h{i}_{j}"].goal for j in range(3)]
        pos_nodes = [nodes[f"h{i}_{j}"] for j in range(3)]
        node.append(decomposition(goal, pos_goals, pos_nodes))
        neg_goals = [nodes[f"hn{i}_{j}"].goal for j in range(2)]
        neg_nodes = [nodes[f"hn{i}_{j}"] for j in range(2)]
        node.append(
            decomposition(goal, neg_goals, neg_nodes, polarity=Polarity.Negative)
        )

    n_goals: list[Goal] = []
    n_nodes: list[LemmaNode] = []
    for i in range(3):
        goal = Goal(name=f"n{i}", statement=f"neg_stmt_{i}.")
        node = LemmaNode(goal=goal)
        nodes[f"n{i}"] = node
        n_goals.append(goal)
        n_nodes.append(node)
        child_goals = [nodes[f"n{i}_{j}"].goal for j in range(3)]
        child_nodes = [nodes[f"n{i}_{j}"] for j in range(3)]
        node.append(decomposition(goal, child_goals, child_nodes))

    root = LemmaNode(goal=GOAL)
    nodes["root"] = root
    root.append(decomposition(GOAL, h_goals, h_nodes))
    root.append(decomposition(GOAL, n_goals, n_nodes, polarity=Polarity.Negative))
    return root, nodes


def build_merged_alias_tree() -> tuple[LemmaNode, dict[str, LemmaNode]]:
    base_goal = Goal(name="base", statement="0 + 0 = 0.")
    step_goal = Goal(name="step", statement="True.")
    fin_goal = Goal(name="fin", statement="False.")
    b0_goal = Goal(name="b0", statement="True.")
    b1_goal = Goal(name="b1", statement="True.")
    b2_goal = Goal(name="b2", statement="True.")
    mid2_goal = Goal(name="mid2", statement="True.")
    mid3_goal = Goal(name="mid3", statement="True.")

    base = LemmaNode(goal=base_goal)
    step = LemmaNode(goal=step_goal)
    fin = LemmaNode(goal=fin_goal)
    b0 = LemmaNode(goal=b0_goal)
    b1 = LemmaNode(goal=b1_goal)
    b2 = LemmaNode(goal=b2_goal)
    mid2 = LemmaNode(goal=mid2_goal)
    mid3 = LemmaNode(goal=mid3_goal)

    base.append(decomposition(base_goal, [b0_goal, b1_goal, b2_goal], [b0, b1, b2]))
    prove(b0)
    prove(b1)
    prove(b2)
    prove(base)

    step.append(
        decomposition(step_goal, [base_goal, mid2_goal, mid3_goal], [base, mid2, mid3])
    )
    prove(mid2)

    refute(fin)

    root = LemmaNode(goal=GOAL)
    root.append(
        decomposition(GOAL, [base_goal, step_goal, fin_goal], [base, step, fin])
    )

    nodes = {
        "root": root,
        "base": base,
        "step": step,
        "fin": fin,
        "b0": b0,
        "b1": b1,
        "b2": b2,
        "mid2": mid2,
        "mid3": mid3,
    }
    return root, nodes


# --- statement_for_polarity ---


def test_statement_for_polarity_positive_keeps_statement() -> None:
    assert statement_for_polarity(GOAL.statement, Polarity.Positive) == GOAL.statement


def test_statement_for_polarity_negative_wraps_with_not() -> None:
    assert statement_for_polarity(GOAL.statement, Polarity.Negative) == (
        "~ (forall n : nat, n + 0 = n.)"
    )


# --- RocqEnvironment / Goal ---


def test_goal_defaults_to_empty_environment() -> None:
    assert GOAL.environment == RocqEnvironment()
    assert GOAL.environment.header == ""
    assert GOAL.environment.allowed_axioms == frozenset()


# --- ProofAttempt ---


def test_proof_attempt_defaults_to_positive_polarity() -> None:
    attempt = ProofAttempt(goal=GOAL, script="reflexivity.", new_lemmas=[])
    assert attempt.polarity is Polarity.Positive


def test_proof_attempt_target_statement_follows_polarity() -> None:
    positive = ProofAttempt(
        goal=GOAL,
        script="reflexivity.",
        new_lemmas=[],
        polarity=Polarity.Positive,
    )
    negative = ProofAttempt(
        goal=GOAL,
        script="intro H.",
        new_lemmas=[],
        polarity=Polarity.Negative,
    )
    assert positive.target_statement == GOAL.statement
    assert negative.target_statement == "~ (forall n : nat, n + 0 = n.)"


# --- AttemptRecord ---


def test_attempt_record_can_nest_child_lemma_nodes() -> None:
    helper = Goal(name="helper", statement="True.")
    child = LemmaNode(goal=helper)
    child.append(record("exact I.", success=True, goal=helper))

    parent = AttemptRecord(
        attempt=ProofAttempt(
            goal=GOAL,
            script="apply helper.",
            new_lemmas=[helper],
        ),
        rocq_error=CoqcResult(success=False, stderr="Error."),
        lemmas=[child],
    )

    assert parent.lemmas[0].status is LemmaStatus.Proved
    assert parent.lemmas[0].goal == helper
    assert parent.fully_succeeded is False


# --- status_from_frontiers ---


def test_status_from_frontiers_open_when_both_missing() -> None:
    assert status_from_frontiers(None, None) is LemmaStatus.Open


def test_status_from_frontiers_open_when_both_fail() -> None:
    assert (
        status_from_frontiers(
            record("auto.", success=False),
            record("intro H.", success=False, polarity=Polarity.Negative),
        )
        is LemmaStatus.Open
    )


def test_status_from_frontiers_proved_prefers_positive() -> None:
    assert (
        status_from_frontiers(
            record("reflexivity.", success=True),
            record("intro H.", success=True, polarity=Polarity.Negative),
        )
        is LemmaStatus.Proved
    )


def test_status_from_frontiers_refuted_when_only_negative_succeeds() -> None:
    assert (
        status_from_frontiers(
            record("auto.", success=False),
            record("intro H.", success=True, polarity=Polarity.Negative),
        )
        is LemmaStatus.Refuted
    )


# --- LemmaNode ---


def test_lemma_node_history_returns_empty_positive_list() -> None:
    node = LemmaNode(goal=GOAL)
    assert node.history(Polarity.Positive) == []
    assert node.history(Polarity.Positive) is node.positive


def test_lemma_node_history_returns_empty_negative_list() -> None:
    node = LemmaNode(goal=GOAL)
    assert node.history(Polarity.Negative) == []
    assert node.history(Polarity.Negative) is node.negative


def test_lemma_node_history_returns_positive_attempts() -> None:
    node = LemmaNode(goal=GOAL)
    first = record("auto.", success=False)
    second = record("reflexivity.", success=True)
    node.append(first)
    node.append(second)
    assert node.history(Polarity.Positive) == [first, second]
    assert node.history(Polarity.Negative) == []


def test_lemma_node_history_returns_negative_attempts() -> None:
    node = LemmaNode(goal=GOAL)
    neg = record("intro H.", success=False, polarity=Polarity.Negative)
    node.append(neg)
    assert node.history(Polarity.Negative) == [neg]
    assert node.history(Polarity.Positive) == []


def test_lemma_node_frontier_is_none_when_history_empty() -> None:
    node = LemmaNode(goal=GOAL)
    assert node.frontier(Polarity.Positive) is None
    assert node.frontier(Polarity.Negative) is None


def test_lemma_node_frontier_returns_latest_when_all_fail() -> None:
    node = LemmaNode(goal=GOAL)
    first = record("auto.", success=False)
    second = record("induction n.", success=False)
    node.append(first)
    node.append(second)
    assert node.frontier(Polarity.Positive) is second


def test_lemma_node_frontier_keeps_latest_successful_after_failed_retry() -> None:
    helper = Goal(name="helper", statement="True.")
    child = LemmaNode(goal=helper)
    successful = AttemptRecord(
        attempt=ProofAttempt(
            goal=GOAL,
            script="apply helper.",
            new_lemmas=[helper],
        ),
        rocq_error=CoqcResult(success=True),
        lemmas=[child],
    )
    failed_retry = record("broken.", success=False)

    node = LemmaNode(goal=GOAL)
    node.append(successful)
    node.append(failed_retry)

    assert node.frontier(Polarity.Positive) is successful


def test_lemma_node_frontier_keeps_latest_successful_negative_after_failed_retry() -> (
    None
):
    helper = Goal(name="helper", statement="False.")
    child = LemmaNode(goal=helper)
    successful = AttemptRecord(
        attempt=ProofAttempt(
            goal=GOAL,
            script="apply helper.",
            new_lemmas=[helper],
            polarity=Polarity.Negative,
        ),
        rocq_error=CoqcResult(success=True),
        lemmas=[child],
    )
    failed_retry = record("broken.", success=False, polarity=Polarity.Negative)

    node = LemmaNode(goal=GOAL)
    node.append(successful)
    node.append(failed_retry)

    assert node.frontier(Polarity.Negative) is successful


def test_lemma_node_append_routes_by_polarity() -> None:
    node = LemmaNode(goal=GOAL)
    pos = record("auto.", success=False)
    neg = record("intro H.", success=False, polarity=Polarity.Negative)

    node.append(pos)
    node.append(neg)

    assert node.positive == [pos]
    assert node.negative == [neg]


def test_lemma_node_append_rejects_mismatched_goal() -> None:
    node = LemmaNode(goal=GOAL)
    other = AttemptRecord(
        attempt=ProofAttempt(
            goal=Goal(name="other", statement="True."),
            script="exact I.",
            new_lemmas=[],
        ),
        rocq_error=CoqcResult(success=True),
    )
    with pytest.raises(ValueError) as exc_info:
        node.append(other)
    assert str(exc_info.value) == (
        f"Attempt goal {other.attempt.goal!r} does not match lemma node goal {GOAL!r}"
    )


def test_lemma_node_agent_call_counts_reads_attempt_agent_names() -> None:
    node = LemmaNode(goal=GOAL)
    node.append(
        AttemptRecord(
            attempt=ProofAttempt(
                goal=GOAL,
                script="auto.",
                new_lemmas=[],
                agent="RepairDirectAboutAgent",
            ),
            rocq_error=CoqcResult(success=False, stderr="fail"),
        )
    )
    node.append(
        AttemptRecord(
            attempt=ProofAttempt(
                goal=GOAL,
                script="auto.",
                new_lemmas=[],
                agent="RepairDirectAboutAgent",
            ),
            rocq_error=CoqcResult(success=False, stderr="fail"),
        )
    )
    node.append(
        AttemptRecord(
            attempt=ProofAttempt(
                goal=GOAL,
                polarity=Polarity.Negative,
                script="intro H.",
                new_lemmas=[],
                agent="DecompositionAboutAgent",
            ),
            rocq_error=CoqcResult(success=False, stderr="fail"),
        )
    )
    node.append(
        AttemptRecord(
            attempt=ProofAttempt(
                goal=GOAL,
                script="auto.",
                new_lemmas=[],
            ),
            rocq_error=CoqcResult(success=False, stderr="fail"),
        )
    )

    assert node.agent_call_counts() == {
        "RepairDirectAboutAgent": 2,
        "DecompositionAboutAgent": 1,
    }


def test_lemma_node_status_is_open_with_empty_histories() -> None:
    node = LemmaNode(goal=GOAL)
    assert node.status is LemmaStatus.Open


def test_lemma_node_status_proved_when_latest_positive_succeeds() -> None:
    node = LemmaNode(goal=GOAL)
    node.append(record("auto.", success=False))
    node.append(record("reflexivity.", success=True))
    assert node.status is LemmaStatus.Proved


def test_lemma_node_status_refuted_when_latest_negative_succeeds() -> None:
    node = LemmaNode(goal=GOAL)
    node.append(record("auto.", success=False))
    node.append(
        record("intro H. contradiction.", success=True, polarity=Polarity.Negative)
    )
    assert node.status is LemmaStatus.Refuted


def test_lemma_node_status_refuted_after_failed_positives_and_mixed_negatives() -> None:
    node = LemmaNode(goal=GOAL)
    node.append(record("auto.", success=False))
    node.append(record("induction n.", success=False))
    node.append(record("intro H.", success=False, polarity=Polarity.Negative))
    node.append(
        record("intro H. contradiction.", success=True, polarity=Polarity.Negative)
    )
    assert node.status is LemmaStatus.Refuted


def test_lemma_node_status_open_when_coqc_ok_but_child_still_open() -> None:
    helper = Goal(name="helper", statement="True.")
    child = LemmaNode(goal=helper)
    node = LemmaNode(goal=GOAL)
    node.append(
        AttemptRecord(
            attempt=ProofAttempt(
                goal=GOAL,
                script="apply helper.",
                new_lemmas=[helper],
            ),
            rocq_error=CoqcResult(success=True),
            lemmas=[child],
        )
    )
    assert child.status is LemmaStatus.Open
    assert node.status is LemmaStatus.Open


def test_lemma_node_status_proved_when_coqc_ok_and_children_proved() -> None:
    helper = Goal(name="helper", statement="True.")
    child = LemmaNode(goal=helper)
    child.append(record("exact I.", success=True, goal=helper))
    node = LemmaNode(goal=GOAL)
    node.append(
        AttemptRecord(
            attempt=ProofAttempt(
                goal=GOAL,
                script="apply helper.",
                new_lemmas=[helper],
            ),
            rocq_error=CoqcResult(success=True),
            lemmas=[child],
        )
    )
    assert child.status is LemmaStatus.Proved
    assert node.status is LemmaStatus.Proved


def test_lemma_node_status_open_when_coqc_ok_but_one_child_refuted() -> None:
    proved = Goal(name="ok", statement="True.")
    refuted = Goal(name="bad", statement="False.")
    proved_child = LemmaNode(goal=proved)
    proved_child.append(record("exact I.", success=True, goal=proved))
    refuted_child = LemmaNode(goal=refuted)
    refuted_child.append(
        record("intro H.", success=True, polarity=Polarity.Negative, goal=refuted)
    )
    node = LemmaNode(goal=GOAL)
    node.append(
        AttemptRecord(
            attempt=ProofAttempt(
                goal=GOAL,
                script="apply ok. apply bad.",
                new_lemmas=[proved, refuted],
            ),
            rocq_error=CoqcResult(success=True),
            lemmas=[proved_child, refuted_child],
        )
    )
    assert proved_child.status is LemmaStatus.Proved
    assert refuted_child.status is LemmaStatus.Refuted
    assert node.status is LemmaStatus.Open


def test_lemma_node_status_open_when_coqc_fails_and_one_child_refuted() -> None:
    proved = Goal(name="ok", statement="True.")
    refuted = Goal(name="bad", statement="False.")
    proved_child = LemmaNode(goal=proved)
    proved_child.append(record("exact I.", success=True, goal=proved))
    refuted_child = LemmaNode(goal=refuted)
    refuted_child.append(
        record("intro H.", success=True, polarity=Polarity.Negative, goal=refuted)
    )
    node = LemmaNode(goal=GOAL)
    node.append(
        AttemptRecord(
            attempt=ProofAttempt(
                goal=GOAL,
                script="apply ok. apply bad.",
                new_lemmas=[proved, refuted],
            ),
            rocq_error=CoqcResult(success=False, stderr="Error."),
            lemmas=[proved_child, refuted_child],
        )
    )
    assert proved_child.status is LemmaStatus.Proved
    assert refuted_child.status is LemmaStatus.Refuted
    assert node.status is LemmaStatus.Open


def test_lemma_node_status_open_when_negative_coqc_ok_but_child_open() -> None:
    helper = Goal(name="helper", statement="False.")
    child = LemmaNode(goal=helper)
    node = LemmaNode(goal=GOAL)
    node.append(
        AttemptRecord(
            attempt=ProofAttempt(
                goal=GOAL,
                polarity=Polarity.Negative,
                script="apply helper.",
                new_lemmas=[helper],
            ),
            rocq_error=CoqcResult(success=True),
            lemmas=[child],
        )
    )
    assert node.status is LemmaStatus.Open


def test_lemma_node_status_refuted_when_negative_coqc_ok_and_children_proved() -> None:
    helper = Goal(name="helper", statement="False.")
    child = LemmaNode(goal=helper)
    child.append(record("exact I.", success=True, goal=helper))
    node = LemmaNode(goal=GOAL)
    node.append(
        AttemptRecord(
            attempt=ProofAttempt(
                goal=GOAL,
                polarity=Polarity.Negative,
                script="apply helper.",
                new_lemmas=[helper],
            ),
            rocq_error=CoqcResult(success=True),
            lemmas=[child],
        )
    )
    assert node.status is LemmaStatus.Refuted


def test_lemma_node_status_stays_proved_after_failed_positive_retry() -> None:
    helper = Goal(name="helper", statement="True.")
    child = LemmaNode(goal=helper)
    child.append(record("exact I.", success=True, goal=helper))
    node = LemmaNode(goal=GOAL)
    node.append(
        AttemptRecord(
            attempt=ProofAttempt(
                goal=GOAL,
                script="apply helper.",
                new_lemmas=[helper],
            ),
            rocq_error=CoqcResult(success=True),
            lemmas=[child],
        )
    )
    node.append(record("broken.", success=False))

    assert node.status is LemmaStatus.Proved


def test_lemma_node_status_proved_when_positive_parent_and_three_children_proved() -> (
    None
):
    root, children = three_child_parent(Polarity.Positive)
    for child in children:
        prove(child)
    assert root.status is LemmaStatus.Proved


def test_lemma_node_status_open_when_positive_parent_and_one_child_refuted() -> None:
    root, children = three_child_parent(Polarity.Positive)
    prove(children[0])
    prove(children[1])
    refute(children[2])
    assert root.status is LemmaStatus.Open


def test_lemma_node_status_open_when_positive_parent_and_all_children_refuted() -> None:
    root, children = three_child_parent(Polarity.Positive)
    for child in children:
        refute(child)
    assert root.status is LemmaStatus.Open


def test_lemma_node_status_refuted_when_negative_parent_and_three_children_proved() -> (
    None
):
    root, children = three_child_parent(Polarity.Negative)
    for child in children:
        prove(child)
    assert root.status is LemmaStatus.Refuted


def test_lemma_node_status_open_when_negative_parent_and_one_child_refuted() -> None:
    root, children = three_child_parent(Polarity.Negative)
    prove(children[0])
    prove(children[1])
    refute(children[2])
    assert root.status is LemmaStatus.Open


def test_lemma_node_status_open_when_negative_parent_and_all_children_refuted() -> None:
    root, children = three_child_parent(Polarity.Negative)
    for child in children:
        refute(child)
    assert root.status is LemmaStatus.Open


def test_lemma_node_from_position_returns_root_at_empty_position() -> None:
    node = LemmaNode(goal=GOAL)
    assert node.from_position(()) is node


def test_lemma_node_nodes_along_position_includes_root() -> None:
    node = LemmaNode(goal=GOAL)
    assert node.nodes_along_position(()) == [node]


def test_lemma_node_nodes_along_position_follows_children() -> None:
    helper = Goal(name="helper", statement="True.")
    child = LemmaNode(goal=helper)
    node = LemmaNode(goal=GOAL)
    node.append(
        AttemptRecord(
            attempt=ProofAttempt(
                goal=GOAL,
                script="apply helper.",
                new_lemmas=[helper],
            ),
            rocq_error=CoqcResult(success=True),
            lemmas=[child],
        )
    )
    position = ((Polarity.Positive, 0),)

    assert node.nodes_along_position(position) == [node, child]
    assert node.from_position(position) is child


def test_lemma_node_nodes_along_position_follows_successful_frontier() -> None:
    helper = Goal(name="helper", statement="True.")
    child = LemmaNode(goal=helper)
    successful = AttemptRecord(
        attempt=ProofAttempt(
            goal=GOAL,
            script="apply helper.",
            new_lemmas=[helper],
        ),
        rocq_error=CoqcResult(success=True),
        lemmas=[child],
    )
    failed_retry = AttemptRecord(
        attempt=ProofAttempt(
            goal=GOAL,
            script="apply other.",
            new_lemmas=[Goal(name="other", statement="False.")],
        ),
        rocq_error=CoqcResult(success=False, stderr="Error."),
        lemmas=[LemmaNode(goal=Goal(name="other", statement="False."))],
    )

    node = LemmaNode(goal=GOAL)
    node.append(successful)
    node.append(failed_retry)
    position = ((Polarity.Positive, 0),)

    assert node.nodes_along_position(position) == [node, child]
    assert node.from_position(position) is child


def test_lemma_node_nodes_along_position_on_wide_depth2_tree() -> None:
    root, nodes = build_wide_depth2_tree()

    assert root.nodes_along_position(()) == [root]
    assert root.nodes_along_position(((Polarity.Positive, 0),)) == [
        root,
        nodes["h0"],
    ]
    assert root.nodes_along_position(
        (
            (Polarity.Positive, 1),
            (Polarity.Positive, 0),
        )
    ) == [root, nodes["h1"], nodes["h1_0"]]
    assert root.nodes_along_position(
        (
            (Polarity.Positive, 0),
            (Polarity.Negative, 1),
        )
    ) == [root, nodes["h0"], nodes["hn0_1"]]
    assert root.nodes_along_position(((Polarity.Negative, 2),)) == [
        root,
        nodes["n2"],
    ]
    assert root.nodes_along_position(
        (
            (Polarity.Negative, 1),
            (Polarity.Positive, 2),
        )
    ) == [root, nodes["n1"], nodes["n1_2"]]


def test_lemma_node_from_position_on_wide_depth2_tree() -> None:
    root, nodes = build_wide_depth2_tree()

    assert root.from_position(()) is root
    assert root.from_position(((Polarity.Positive, 1),)) is nodes["h1"]
    assert (
        root.from_position(
            (
                (Polarity.Positive, 0),
                (Polarity.Positive, 2),
            )
        )
        is nodes["h0_2"]
    )
    assert root.from_position(((Polarity.Negative, 0),)) is nodes["n0"]
    assert (
        root.from_position(
            (
                (Polarity.Negative, 2),
                (Polarity.Positive, 1),
            )
        )
        is nodes["n2_1"]
    )
    assert (
        root.from_position(
            (
                (Polarity.Positive, 1),
                (Polarity.Negative, 0),
            )
        )
        is nodes["hn1_0"]
    )


def test_lemma_node_nodes_along_position_rejects_out_of_range_index() -> None:
    root, nodes = build_wide_depth2_tree()

    # Index 2 is valid on the negative frontier (n0, n1, n2)...
    assert root.from_position(((Polarity.Negative, 2),)) is nodes["n2"]
    # ...but out of range on the positive frontier (only h0, h1).
    with pytest.raises(ValueError, match="Helper index 2 out of range for 2 lemmas"):
        root.nodes_along_position(((Polarity.Positive, 2),))


def test_lemma_node_nodes_along_position_rejects_missing_frontier() -> None:
    node = LemmaNode(goal=GOAL)
    node.append(record("auto.", success=False))

    with pytest.raises(ValueError, match="No frontier attempt for polarity 'negative'"):
        node.nodes_along_position(((Polarity.Negative, 0),))


def test_lemma_node_status_on_merged_alias_tree() -> None:
    root, nodes = build_merged_alias_tree()

    assert nodes["base"].status is LemmaStatus.Proved
    assert nodes["step"].status is LemmaStatus.Open
    assert nodes["fin"].status is LemmaStatus.Refuted
    assert nodes["mid2"].status is LemmaStatus.Proved
    assert nodes["mid3"].status is LemmaStatus.Open
    assert root.status is LemmaStatus.Open


def test_lemma_node_nodes_reaching_on_merged_alias_tree() -> None:
    root, nodes = build_merged_alias_tree()

    reaching_base = root.nodes_reaching(nodes["base"])
    assert reaching_base[id(nodes["base"])] is nodes["base"]
    assert reaching_base[id(root)] is root
    assert reaching_base[id(nodes["step"])] is nodes["step"]
    assert id(nodes["b0"]) not in reaching_base

    reaching_b2 = root.nodes_reaching(nodes["b2"])
    assert reaching_b2[id(nodes["b2"])] is nodes["b2"]
    assert reaching_b2[id(nodes["base"])] is nodes["base"]
    assert reaching_b2[id(root)] is root
    assert reaching_b2[id(nodes["step"])] is nodes["step"]

    reaching_mid3 = root.nodes_reaching(nodes["mid3"])
    assert reaching_mid3[id(nodes["mid3"])] is nodes["mid3"]
    assert reaching_mid3[id(nodes["step"])] is nodes["step"]
    assert reaching_mid3[id(root)] is root
    assert id(nodes["base"]) not in reaching_mid3

    reaching_fin = root.nodes_reaching(nodes["fin"])
    assert reaching_fin[id(nodes["fin"])] is nodes["fin"]
    assert reaching_fin[id(root)] is root
    assert id(nodes["step"]) not in reaching_fin


def test_lemma_node_walk_visits_all_distinct_nodes_on_merged_alias_tree() -> None:
    root, nodes = build_merged_alias_tree()

    walked = list(root.walk())
    expected = [
        nodes["root"],
        nodes["base"],
        nodes["b0"],
        nodes["b1"],
        nodes["b2"],
        nodes["step"],
        nodes["mid2"],
        nodes["mid3"],
        nodes["fin"],
    ]

    assert walked == expected
    assert len(walked) == len({id(node) for node in walked})


def test_lemma_node_walk_follows_children() -> None:
    helper = Goal(name="helper", statement="True.")
    child = LemmaNode(goal=helper)
    node = LemmaNode(goal=GOAL)
    node.append(
        AttemptRecord(
            attempt=ProofAttempt(
                goal=GOAL,
                script="apply helper.",
                new_lemmas=[helper],
            ),
            rocq_error=CoqcResult(success=True),
            lemmas=[child],
        )
    )

    assert list(node.walk()) == [node, child]
