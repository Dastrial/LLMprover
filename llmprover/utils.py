"""Shared helpers for LLMprover (prompt formatting, LLM output parsing, etc.)."""

from __future__ import annotations

from llmprover.domain import AttemptRecord, LemmaNode
from llmprover.rocq.helper_resolution import script_with_child_names

EMPTY_ATTEMPTS = "No previous attempts.\n"


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


def number_script_lines(script: str) -> str:
    """Prefix each line of *script* with a 1-based ``N | `` marker."""
    lines = script.splitlines()
    return "\n".join(f"{index} | {line}" for index, line in enumerate(lines, start=1))


def format_attempt(
    record: AttemptRecord,
    index: int,
    *,
    include_agent: bool = False,
) -> str:
    """Format one attempt record (same layout as one block in ``format_attempts``)."""
    attempt = record.attempt
    parts = [
        f"Attempt {index}:",
        f"Statement: {attempt.target_statement}",
    ]
    if include_agent and attempt.agent:
        parts.append(f"Agent: {attempt.agent}")
    if record.lemmas:
        lemma_lines = ["Helper lemmas:"]
        for lemma in record.lemmas:
            goal = lemma.goal
            lemma_lines.append(
                f"  {goal.name} [{lemma.status.value}]: {goal.statement}"
            )
        parts.append("\n".join(lemma_lines))
    displayed = script_with_child_names(record)
    if displayed.strip():
        parts.append(f"Script:\n{number_script_lines(displayed)}")
    else:
        parts.append("Script: (empty)")
    parts.append(f"Rocq errors: {record.rocq_error.output_without_warnings}")
    return "\n\n".join(parts) + "\n\n"


def format_attempts(
    records: list[AttemptRecord],
    *,
    empty: str = "",
    include_agent: bool = False,
) -> str:
    """Format attempt records for inclusion in an LLM prompt.

    Each record is rendered with blank lines between top-level fields
    (statement, optional agent, optional helper lemmas, script, Rocq errors).
    Helper lemmas come from ``record.lemmas`` (nested ``LemmaNode``s) and
    include ``LemmaNode.status``. When *records* is empty, return *empty*
    unchanged.

    ``include_agent`` is for the strategy prompt only: ProverAgent histories
    omit who produced the attempt.
    """
    if not records:
        return empty
    return "".join(
        format_attempt(record, index, include_agent=include_agent)
        for index, record in enumerate(records, start=1)
    )


def format_histories(
    node: LemmaNode,
    empty: str = EMPTY_ATTEMPTS,
    *,
    include_agent: bool = False,
) -> tuple[str, str]:
    return (
        format_attempts(node.positive, empty=empty, include_agent=include_agent),
        format_attempts(node.negative, empty=empty, include_agent=include_agent),
    )


def normalize_llm_summary(text: str, *, empty: str = EMPTY_ATTEMPTS) -> str:
    """Strip fences and ensure a trailing newline."""
    cleaned = strip_markdown_fences(text.strip())
    if not cleaned:
        return empty
    if not cleaned.endswith("\n"):
        cleaned += "\n"
    return cleaned
