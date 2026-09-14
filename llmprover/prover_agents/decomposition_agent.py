"""LLM-based Proof agent class for decomposition"""

from __future__ import annotations

import re
from pathlib import Path

from llmprover.domain import (
    Goal,
    LemmaNode,
    Polarity,
    ProofAttempt,
    RocqEnvironment,
    statement_for_polarity,
)
from llmprover.history.presenter import DeterministicHistoryPresenter, HistoryPresenter
from llmprover.llm.client import LLMClient, TokenUsage
from llmprover.llm.prompting import (
    PromptMessage,
    PromptPart,
    fill_prompt,
)
from llmprover.prover_agents.prompt_assembly import (
    DECOMPOSITION_SPEC,
    cached_system_prompt,
)
from llmprover.prover_agents.prover_agent import ProverAgent
from llmprover.utils import (
    strip_markdown_fences,
    strip_proof_wrappers,
)

PROMPTS_DIR = Path(__file__).parent / "prompts"
LEMMA_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_']*$")


def _parse_lemma_line(line: str, environment: RocqEnvironment) -> Goal | None:
    """Parse ``name: statement``; return ``None`` when the line is not a helper."""
    stripped = line.strip()
    if ":" not in stripped:
        return None
    name, _, statement = stripped.partition(":")
    name = name.strip()
    statement = statement.strip()
    if not name or not statement or LEMMA_NAME_RE.fullmatch(name) is None:
        return None
    return Goal(statement=statement, name=name, environment=environment)


def _marker_index(lines: list[str], marker: str, start: int = 0) -> int | None:
    for index in range(start, len(lines)):
        if lines[index].strip() == marker:
            return index
    return None


def _parse_helpers_section(
    lines: list[str], environment: RocqEnvironment
) -> list[Goal]:
    """Parse helper declarations, ignoring blank lines between them.

    A wrapped statement (extra newlines inside one helper) is joined onto the
    previous helper. Lines that are neither a new ``name: statement`` nor a
    continuation are skipped.
    """
    lemmas: list[Goal] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        lemma = _parse_lemma_line(stripped, environment)
        if lemma is not None:
            lemmas.append(lemma)
            continue
        if lemmas:
            previous = lemmas[-1]
            lemmas[-1] = Goal(
                statement=f"{previous.statement} {stripped}",
                name=previous.name,
                environment=previous.environment,
            )
    return lemmas


def parse_decomposition_answer(
    answer: str,
    environment: RocqEnvironment | None = None,
) -> tuple[list[Goal], str]:
    """Parse LLM output with ``SCRIPT:`` then ``HELPERS:`` section markers.

    Helper goals inherit *environment* (default empty) from the parent goal.
    Blank lines in the HELPERS section are ignored; a helper statement split
    across lines is concatenated. Malformed helper lines that cannot continue
    a previous helper are skipped.

    A missing ``HELPERS:`` section yields no helpers; a missing ``SCRIPT:``
    section yields an empty script. Without either marker, the whole answer is
    the script. The legacy ``HELPERS:``-then-``SCRIPT:`` order is still
    accepted when both markers are present.
    """
    env = environment if environment is not None else RocqEnvironment()
    lines = strip_markdown_fences(answer.strip()).splitlines()

    script_idx = _marker_index(lines, "SCRIPT:")
    helpers_idx = _marker_index(lines, "HELPERS:")

    if script_idx is None and helpers_idx is None:
        return [], strip_proof_wrappers("\n".join(lines))

    if (
        helpers_idx is not None
        and script_idx is not None
        and helpers_idx < script_idx
    ):
        helper_lines = lines[helpers_idx + 1 : script_idx]
        script = strip_proof_wrappers("\n".join(lines[script_idx + 1 :]))
    elif script_idx is not None:
        helper_end = helpers_idx if helpers_idx is not None else len(lines)
        script = strip_proof_wrappers("\n".join(lines[script_idx + 1 : helper_end]))
        helper_lines = lines[helpers_idx + 1 :] if helpers_idx is not None else []
    else:
        script = ""
        helper_lines = lines[helpers_idx + 1 :]

    return _parse_helpers_section(helper_lines, env), script


class DecompositionAgent(ProverAgent):
    """LLM-based agent for decomposing a goal into lemmas.

    Includes the full attempt history (direct and decomposition attempts) for
    both polarities, so the model can avoid repeating failed strategies.
    """

    DEFAULT_SPEC = (
        "Decomposes the goal into helper lemmas using the full attempt history "
        "(direct and decomposition, both polarities), including helper-lemma "
        "statuses. Most token-hungry agent: prompt size grows with every failed "
        "attempt and with the lemmas attached to decompositions."
    )

    def __init__(
        self,
        model: LLMClient,
        *,
        history_presenter: HistoryPresenter | None = None,
    ) -> None:
        self.model = model
        self.history_presenter = (
            history_presenter or DeterministicHistoryPresenter.full()
        )

    def prove(
        self, node: LemmaNode, polarity: Polarity
    ) -> tuple[ProofAttempt, TokenUsage]:
        goal = node.goal
        attempt_history, hist_usage = self.history_presenter.present_prompt_block(node)
        system = cached_system_prompt(DECOMPOSITION_SPEC)
        before = fill_prompt(
            (PROMPTS_DIR / "decomposition_user_before.txt").read_text(encoding="utf-8").strip(),
            header=goal.environment.header,
        )
        if not before.endswith("\n"):
            before = f"{before}\n"
        after = fill_prompt(
            (PROMPTS_DIR / "decomposition_user_after.txt").read_text(encoding="utf-8").strip(),
            statement=statement_for_polarity(goal.statement, polarity),
        )
        if after and not after.endswith("\n"):
            after = f"{after}\n"
        history = (
            attempt_history
            if attempt_history.endswith("\n")
            else f"{attempt_history}\n"
        )
        completion = self.model.complete(
            [
                PromptMessage.text("system", system, cache_breakpoint=True),
                PromptMessage(
                    role="user",
                    parts=(PromptPart(before + history, cache_breakpoint=True), PromptPart(after)),
                ),
            ]
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
            ),
            hist_usage + completion.usage,
        )
