"""Shared multi-turn Search → (optional select) → About → proof lookup for About agents."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from llmprover.llm.client import LLMClient, TokenUsage
from llmprover.llm.prompting import (
    PromptMessage,
    PromptPart,
    fill_prompt,
)
from llmprover.rocq.proof_script import ProofScript
from llmprover.prover_agents.about_limits import (
    MAX_SEARCH_COMMANDS,
    MAX_SEARCH_HITS_COLLECT,
    MAX_SELECTED_SEARCH_HITS,
    SEARCH_SELECTION_THRESHOLD,
)
from llmprover.rocq.backend import CoqcBackend
from llmprover.utils import strip_markdown_fences

PROMPTS_DIR = Path(__file__).parent / "prompts"
EMPTY_ABOUT = "(no lemmas requested)\n"
EMPTY_SEARCH = "(no search requested)\n"
# Longest names first so ``SearchPattern`` is not parsed as ``Search``.
SEARCH_COMMANDS = ("SearchPattern", "SearchRewrite", "Search")
# With ``Set Search Output Name Only``, Rocq prints one qualified name per line.
SEARCH_HIT_NAME_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_']*(?:\.[A-Za-z_][A-Za-z0-9_']*)*$"
)
SKIP_HIT_NAMES = frozenset({"Warning", "Error", "File"})

DIRECT_ABOUT_SEARCH_USER_BEFORE = PROMPTS_DIR / "direct_about_search_user_before.txt"
DIRECT_ABOUT_SEARCH_USER_AFTER = PROMPTS_DIR / "direct_about_search_user_after.txt"
REPAIR_DIRECT_ABOUT_SEARCH_USER_BEFORE = (
    PROMPTS_DIR / "repair_direct_about_search_user_before.txt"
)
REPAIR_DIRECT_ABOUT_SEARCH_USER_AFTER = (
    PROMPTS_DIR / "repair_direct_about_search_user_after.txt"
)
DECOMPOSITION_ABOUT_SEARCH_USER_BEFORE = (
    PROMPTS_DIR / "decomposition_about_search_user_before.txt"
)
DECOMPOSITION_ABOUT_SEARCH_USER_AFTER = (
    PROMPTS_DIR / "decomposition_about_search_user_after.txt"
)
ABOUT_SELECT_USER = PROMPTS_DIR / "about_select_user.txt"


@dataclass(frozen=True)
class SearchAboutResult:
    """Outcome of the Search → optional select → About lookup phase."""

    about_results: str
    usage: TokenUsage
    lemma_names: list[str]
    prefix_messages: tuple[PromptMessage, ...]
    search_output: str
    select_output: str | None = None


def _prompt_text(prompt: Path | str) -> str:
    """Load prompt text from a path, or return inline string content.

    ``str`` values that name an existing file are read as paths (tests). Long or
    multiline strings are treated as already-rendered prompt text.
    """
    if isinstance(prompt, Path):
        return prompt.read_text(encoding="utf-8").strip()
    if "\n" in prompt or len(prompt) > 512:
        return prompt.strip()
    path = Path(prompt)
    try:
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    except OSError:
        pass
    return prompt.strip()


def lookup_completion(
    model: LLMClient,
    *,
    system_prompt: Path | str,
    user_before_prompt: Path | str,
    user_after_prompt: Path | str,
    **prompt_values: str,
):
    """Run lookup: system, then cached before (+ history), then after (statement)."""
    system = _prompt_text(system_prompt)
    history = prompt_values.get("attempt_history", "")
    fill_values = {
        key: value for key, value in prompt_values.items() if key != "attempt_history"
    }
    before = fill_prompt(_prompt_text(user_before_prompt), **fill_values)
    if not before.endswith("\n"):
        before = f"{before}\n"
    after = fill_prompt(_prompt_text(user_after_prompt), **fill_values)
    if after and not after.endswith("\n"):
        after = f"{after}\n"
    if history and not history.endswith("\n"):
        history = f"{history}\n"
    user = PromptMessage(
        role="user",
        parts=(PromptPart(before + history, cache_breakpoint=True), PromptPart(after)),
    )
    return model.complete(
        [PromptMessage.text("system", system, cache_breakpoint=True), user]
    )


def parse_lemma_names(text: str) -> list[str]:
    """Parse one lemma name per line from an LLM About-lookup response."""
    names: list[str] = []
    seen: set[str] = set()
    for raw in strip_markdown_fences(text).splitlines():
        line = raw.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("about "):
            line = line[6:].strip()
        line = line.rstrip(".").strip()
        if not line or line in seen:
            continue
        seen.add(line)
        names.append(line)
    return names


def first_vernacular_statement(line: str) -> str:
    """Return the first period-terminated statement, respecting strings/parens."""
    depth = 0
    in_string = False
    for i, char in enumerate(line):
        if in_string:
            if char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == "." and depth == 0:
            return line[: i + 1].strip()
    return line.strip()


def normalize_search_command(line: str) -> str | None:
    """Return a canonical ``Search``/``SearchPattern``/``SearchRewrite`` command.

    Rejects other vernacular (``About``, ``Require``, ``SearchHead``, …). Only the
    first statement on the line is kept. Returns ``None`` when the line is not
    an allowed search command.
    """
    stripped = line.strip().strip("`")
    if not stripped:
        return None
    statement = first_vernacular_statement(stripped).rstrip(".").strip()
    if not statement:
        return None
    lower = statement.lower()
    for cmd in SEARCH_COMMANDS:
        prefix = cmd.lower()
        if not lower.startswith(prefix):
            continue
        if len(statement) > len(cmd):
            nxt = statement[len(cmd)]
            if nxt.isalnum() or nxt in "_'":
                continue
        rest = statement[len(cmd) :].strip()
        if rest:
            return f"{cmd} {rest}."
        return f"{cmd}."
    return None


def parse_search_commands(text: str) -> list[str]:
    """Parse one ``Search``/``SearchPattern``/``SearchRewrite`` command per line."""
    commands: list[str] = []
    seen: set[str] = set()
    for raw in strip_markdown_fences(text).splitlines():
        command = normalize_search_command(raw)
        if command is None or command in seen:
            continue
        seen.add(command)
        commands.append(command)
    return commands


def parse_lookup_request(text: str) -> tuple[list[str], list[str]]:
    """Parse mixed ``Search`` commands and ``About`` names from one LLM reply.

    Keeps at most ``MAX_SEARCH_COMMANDS`` search commands. Lemma names are taken
    only from lines that start with ``About``.
    """
    commands: list[str] = []
    names: list[str] = []
    seen_commands: set[str] = set()
    seen_names: set[str] = set()
    for raw in strip_markdown_fences(text).splitlines():
        command = normalize_search_command(raw)
        if command is not None:
            if command in seen_commands or len(commands) >= MAX_SEARCH_COMMANDS:
                continue
            seen_commands.add(command)
            commands.append(command)
            continue
        stripped = raw.strip().strip("`")
        if not stripped.lower().startswith("about "):
            continue
        for name in parse_lemma_names(stripped):
            if name in seen_names:
                continue
            seen_names.add(name)
            names.append(name)
    return commands, names


def parse_search_hit_names(output: str, *, stop_after: int | None = None) -> list[str]:
    """Extract lemma names from a name-only Rocq ``Search`` listing.

    With ``Set Search Output Name Only``, each hit is one qualified name on its
    own line. Indented lines are ignored. When *stop_after* is set, parsing
    stops after that many names plus one extra hit so callers can detect
    overflow.
    """
    names: list[str] = []
    seen: set[str] = set()
    overflow_limit = None if stop_after is None else stop_after + 1
    for line in output.splitlines():
        if not line or line[:1].isspace():
            continue
        name = line.strip()
        if not SEARCH_HIT_NAME_RE.fullmatch(name):
            continue
        if name in SKIP_HIT_NAMES or name in seen:
            continue
        seen.add(name)
        names.append(name)
        if overflow_limit is not None and len(names) >= overflow_limit:
            break
    return names


def build_about_script(header: str, names: list[str]) -> str:
    """Build a Vernacular script: header then ``About name.`` for each name."""
    parts: list[str] = []
    prelude = header.strip()
    if prelude:
        parts.append(prelude)
        parts.append("")
    for name in names:
        parts.append(f"About {name}.")
    parts.append("")
    return "\n".join(parts)


def build_search_script(header: str, command: str, *, name_only: bool = True) -> str:
    """Build a Vernacular script: header then optional name-only flag + search."""
    parts: list[str] = []
    prelude = header.strip()
    if prelude:
        parts.append(prelude)
        parts.append("")
    if name_only:
        parts.append("Set Search Output Name Only.")
    parts.append(command if command.endswith(".") else f"{command}.")
    parts.append("")
    return "\n".join(parts)


def run_abouts(checker: CoqcBackend, header: str, names: list[str]) -> str:
    """Run ``About`` queries through ``coqc`` and return combined output."""
    if not names:
        return EMPTY_ABOUT
    result = checker.check_script(
        ProofScript(code=build_about_script(header, names)), tactic_prelude=False
    )
    output = result.output.strip()
    if output:
        return output + "\n"
    if result.success:
        return "(About produced no output)\n"
    return "(About failed with no output)\n"


def run_searches(checker: CoqcBackend, header: str, commands: list[str]) -> str:
    """Run each search command in its own ``coqc`` file and concatenate output.

    A failed command does not prevent later commands from running. Searches run
    with ``Set Search Output Name Only.``.
    """
    if not commands:
        return EMPTY_SEARCH
    blocks: list[str] = []
    for command in commands:
        result = checker.check_script(
            ProofScript(code=build_search_script(header, command)),
            tactic_prelude=False,
        )
        output = result.output.strip()
        if not output:
            output = (
                "(Search produced no output)"
                if result.success
                else "(Search failed with no output)"
            )
        label = (
            f"=== {command} ===" if result.success else f"=== {command} [failed] ==="
        )
        blocks.append(f"{label}\n{output}")
    return "\n\n".join(blocks) + "\n"


def collect_names_from_searches(
    checker: CoqcBackend, header: str, commands: list[str]
) -> list[str]:
    """Return unique hit names from searches (name-only), capped for safety.

    Failed commands are skipped. Hits are collected across successful searches
    up to ``MAX_SEARCH_HITS_COLLECT`` total unique names.
    """
    names: list[str] = []
    seen: set[str] = set()
    for command in commands[:MAX_SEARCH_COMMANDS]:
        if len(names) >= MAX_SEARCH_HITS_COLLECT:
            break
        result = checker.check_script(
            ProofScript(code=build_search_script(header, command)),
            tactic_prelude=False,
        )
        if not result.success:
            continue
        remaining = MAX_SEARCH_HITS_COLLECT - len(names)
        hits = parse_search_hit_names(result.stdout, stop_after=remaining)
        for name in hits[:remaining]:
            if name in seen:
                continue
            seen.add(name)
            names.append(name)
            if len(names) >= MAX_SEARCH_HITS_COLLECT:
                break
    return names


def unique_names(*groups: list[str]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for name in group:
            if name in seen:
                continue
            seen.add(name)
            names.append(name)
    return names


def request_abouts(
    model: LLMClient,
    checker: CoqcBackend,
    *,
    system_prompt: Path | str,
    user_before_prompt: Path | str,
    user_after_prompt: Path | str,
    **prompt_values: str,
) -> tuple[str, TokenUsage, list[str]]:
    """Ask the LLM for lemma names, run ``About`` via coqc, return results text.

    Returns ``(about_output, usage, lemma_names)``.
    """
    completion = lookup_completion(
        model,
        system_prompt=system_prompt,
        user_before_prompt=user_before_prompt,
        user_after_prompt=user_after_prompt,
        **prompt_values,
    )
    names = parse_lemma_names(completion.text)
    header = prompt_values.get("header", "")
    return run_abouts(checker, header, names), completion.usage, names


def request_searches(
    model: LLMClient,
    checker: CoqcBackend,
    *,
    system_prompt: Path | str,
    user_before_prompt: Path | str,
    user_after_prompt: Path | str,
    **prompt_values: str,
) -> tuple[str, TokenUsage, list[str]]:
    """Ask the LLM for search commands, run each via coqc, return results.

    Returns ``(search_output, usage, commands)``.
    """
    completion = lookup_completion(
        model,
        system_prompt=system_prompt,
        user_before_prompt=user_before_prompt,
        user_after_prompt=user_after_prompt,
        **prompt_values,
    )
    commands = parse_search_commands(completion.text)
    header = prompt_values.get("header", "")
    return run_searches(checker, header, commands), completion.usage, commands


def request_search_then_abouts(
    model: LLMClient,
    checker: CoqcBackend,
    *,
    system_prompt: Path | str,
    user_before_prompt: Path | str,
    user_after_prompt: Path | str,
    **prompt_values: str,
) -> SearchAboutResult:
    """Multi-turn Search → optional select → About for the final proof turn.

    Turn 1 asks for Search/About (system prompt explains the full protocol).
    Searches run with ``Set Search Output Name Only.``. If the number of unique
    hits is >= ``SEARCH_SELECTION_THRESHOLD``, turn 2 asks the model to keep at
    most ``MAX_SELECTED_SEARCH_HITS`` names. About is then run on explicit About
    names plus selected Search hits.

    ``prefix_messages`` is the conversation through the first assistant reply
    (only the system and history parts carry cache breakpoints), for the final
    proof turn.
    """
    system = _prompt_text(system_prompt)
    history = prompt_values.get("attempt_history", "")
    fill_values = {
        key: value for key, value in prompt_values.items() if key != "attempt_history"
    }
    before = fill_prompt(_prompt_text(user_before_prompt), **fill_values)
    if not before.endswith("\n"):
        before = f"{before}\n"
    after = fill_prompt(_prompt_text(user_after_prompt), **fill_values)
    if after and not after.endswith("\n"):
        after = f"{after}\n"
    if history and not history.endswith("\n"):
        history = f"{history}\n"

    # Two breakpoints only: system + after history (statement stays uncached).
    first_user = PromptMessage(
        role="user",
        parts=(PromptPart(before + history, cache_breakpoint=True), PromptPart(after)),
    )
    first_messages = [
        PromptMessage.text("system", system, cache_breakpoint=True),
        first_user,
    ]
    first = model.complete(first_messages)
    usage = first.usage
    select_output: str | None = None

    commands, explicit_names = parse_lookup_request(first.text)
    header = prompt_values.get("header", "")
    search_hits = collect_names_from_searches(checker, header, commands)

    selected_hits = search_hits
    if len(search_hits) >= SEARCH_SELECTION_THRESHOLD:
        hits_text = "\n".join(search_hits) + "\n"
        select_text = fill_prompt(
            ABOUT_SELECT_USER.read_text(encoding="utf-8").strip(),
            search_hits=hits_text,
            max_selected=str(MAX_SELECTED_SEARCH_HITS),
        )
        if not select_text.endswith("\n"):
            select_text = f"{select_text}\n"
        assistant_text = first.text if first.text.endswith("\n") else f"{first.text}\n"
        assistant = PromptMessage.text("assistant", assistant_text)
        select_user = PromptMessage.text("user", select_text)
        select = model.complete([*first_messages, assistant, select_user])
        usage = usage + select.usage
        select_output = select.text
        chosen = parse_lemma_names(select.text)[:MAX_SELECTED_SEARCH_HITS]
        hit_set = set(search_hits)
        selected_hits = [name for name in chosen if name in hit_set]
        if not selected_hits and chosen:
            selected_hits = search_hits[:MAX_SELECTED_SEARCH_HITS]
        elif not selected_hits:
            selected_hits = search_hits[:MAX_SELECTED_SEARCH_HITS]

    names = unique_names(explicit_names, selected_hits)
    about_results = run_abouts(checker, header, names)

    assistant_text = first.text if first.text.endswith("\n") else f"{first.text}\n"
    prefix_messages = (
        first_messages[0],
        first_user,
        PromptMessage.text("assistant", assistant_text),
    )
    return SearchAboutResult(
        about_results=about_results,
        usage=usage,
        lemma_names=names,
        prefix_messages=prefix_messages,
        search_output=first.text,
        select_output=select_output,
    )
