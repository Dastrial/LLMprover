"""Shared helpers for LLMprover (prompt formatting, LLM output parsing, etc.)."""

from __future__ import annotations

from llmprover.domain import AttemptRecord, LemmaNode


def strip_markdown_fences(text: str) -> str:
    """Remove optional ``` fences when the model wraps code anyway."""
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def strip_proof_wrappers(script: str) -> str:
    """Remove leading Proof. and trailing Qed. when the model includes them."""
    lines = script.splitlines()
    if lines and lines[0].strip().lower().removesuffix(".") == "proof":
        lines = lines[1:]
    if lines and lines[-1].strip().lower().removesuffix(".") == "qed":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_proof_script(answer: str) -> str:
    """Normalize a direct proof script from LLM output."""
    return strip_proof_wrappers(strip_markdown_fences(answer.strip()))


def format_attempts(
    records: list[AttemptRecord],
    *,
    empty: str = "",
) -> str:
    """Format attempt records for inclusion in an LLM prompt.

    Each record is rendered with blank lines between top-level fields
    (statement, optional helper lemmas, script, Rocq errors). Helper lemmas
    come from ``record.lemmas`` (nested ``LemmaNode``s) and include
    ``LemmaNode.status``. When *records* is empty, return *empty* unchanged.
    """
    if not records:
        return empty
    result = ""
    for index, record in enumerate(records, start=1):
        attempt = record.attempt
        parts = [
            f"Attempt {index}:",
            f"Statement: {attempt.target_statement}",
        ]
        if record.lemmas:
            lemma_lines = ["Helper lemmas:"]
            for lemma in record.lemmas:
                goal = lemma.goal
                lemma_lines.append(
                    f"  {goal.name} [{lemma.status.value}]: {goal.statement}"
                )
            parts.append("\n".join(lemma_lines))
        parts.append(f"Script: {attempt.script}")
        parts.append(f"Rocq errors: {record.rocq_error.output}")
        result += "\n\n".join(parts) + "\n\n"
    return result


def format_histories(
    node: LemmaNode, empty: str = "No previous attempts.\n"
) -> tuple[str, str]:
    return (
        format_attempts(node.positive, empty=empty),
        format_attempts(node.negative, empty=empty),
    )
