"""CLI entry point: run the orchestrator on a small miniF2F-style goal."""

from __future__ import annotations

from dotenv import load_dotenv

from eval_minif2f import build_openai_orchestrator
from llmprover.domain import AttemptRecord, LemmaNode
from llmprover.minif2f.loader import problem_from_row

load_dotenv()  # .env: OPENAI_API_KEY=...  (or: export OPENAI_API_KEY=...)

# Easiest row from ``tests/test_minif2f_live.py``: pure computation.
GOAL = problem_from_row(
    {
        "name": "mathd_numbertheory_188",
        "split": "valid",
        "header": "Require Import Nat.",
        "rocq_statement": ("Theorem mathd_numbertheory_188 :\n  Nat.gcd 180 168 = 12."),
    }
).to_goal()


def format_record(record: AttemptRecord, indent: str) -> list[str]:
    attempt = record.attempt
    ok = "ok" if record.rocq_error.success else "fail"
    lines = [
        f"{indent}- [{attempt.polarity.value}] {ok} ",
        f"new_lemmas={len(attempt.new_lemmas)}",
        f"{indent}  script: {attempt.script!r}",
    ]
    if not record.rocq_error.success and record.rocq_error.display_stderr.strip():
        for line in record.rocq_error.display_stderr.strip().splitlines():
            lines.append(f"{indent}  error: {line}")
    for child in record.lemmas:
        lines.extend(format_node(child, indent + "  "))
    return lines


def format_node(node: LemmaNode, indent: str = "") -> list[str]:
    lines = [f"{indent}* {node.goal.name}: {node.goal.statement} [{node.status.value}]"]
    for record in node.positive + node.negative:
        lines.extend(format_record(record, indent + "  "))
    return lines


if __name__ == "__main__":
    orchestrator = build_openai_orchestrator(verbose=True)
    print(f"Proving: {GOAL.name}")
    print(f"header:\n{GOAL.environment.header}")
    print(f"statement: {GOAL.statement}")
    node = orchestrator.prove(GOAL, max_attempts=10)
    print(f"Status: {node.status.value}")
    print("\n".join(format_node(node)))
