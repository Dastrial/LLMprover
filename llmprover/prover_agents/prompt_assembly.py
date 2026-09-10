"""Declarative composition of modular Rocq prover system prompts."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"
HARD_RULES_DIR = PROMPTS_DIR / "hard_rules"
PROTOCOLS_DIR = PROMPTS_DIR / "protocols"
SOFT_DIR = PROMPTS_DIR / "soft"
FINAL_CHECKS_DIR = PROMPTS_DIR / "final_checks"
INTROS_DIR = PROMPTS_DIR / "intros"

# Capability-grouped HARD RULE stems (filenames without .txt).
COMMON_HARD_RULES: tuple[str, ...] = (
    "obey_output_protocol",
    "no_proof_bypass",
    "rocq_only",
    "mathematical_purpose",
    "account_for_all_goals",
    "fixed_environment",
)
HISTORY_HARD_RULE: str = "history_failures"
ABOUT_HARD_RULE: str = "about_before_library_use"
HELPER_HARD_RULES: tuple[str, ...] = (
    "helper_genuine_progress",
    "helper_reference_protocol",
)

ABOUT_PROTOCOLS: tuple[str, ...] = (
    "about_interaction",
    "about_lookup",
    "about_selection",
)

# Protocol / soft fragments that embed About lookup constants.
_ABOUT_CONSTANT_FRAGMENTS = frozenset(
    {
        "about_interaction",
        "about_lookup",
        "about_selection",
        "about_lookup_guidance",
    }
)

SOFT_INTRO = """\
======================================================================
SOFT RULES — GUIDANCE ONLY
======================================================================

These are heuristics and preferences.
They never override HARD RULES or the current OUTPUT PROTOCOL.
"""


@dataclass(frozen=True)
class ProverPromptSpec:
    """Declarative recipe for one agent system prompt."""

    intro: str
    hard_rules: tuple[str, ...]
    protocols: tuple[str, ...]
    soft_sections: tuple[str, ...]
    final_check: str | None = None


def _read_fragment(directory: Path, stem: str) -> str:
    path = directory / f"{stem}.txt"
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty prompt fragment: {path}")
    return text


def _about_protocol_values() -> dict[str, str]:
    from llmprover.prover_agents.about_limits import (
        MAX_SEARCH_COMMANDS,
        MAX_SELECTED_SEARCH_HITS,
        SEARCH_SELECTION_THRESHOLD,
    )

    return {
        "max_search_commands": str(MAX_SEARCH_COMMANDS),
        "max_selected_search_hits": str(MAX_SELECTED_SEARCH_HITS),
        "search_selection_threshold": str(SEARCH_SELECTION_THRESHOLD),
    }


def _fill_named_placeholders(text: str, values: dict[str, str]) -> str:
    """Replace ``{key}`` for known keys only (safe with Rocq braces)."""
    result = text
    for key, value in values.items():
        result = result.replace(f"{{{key}}}", value)
    return result


def _number_hard_rule(text: str, number: int) -> str:
    if "{N}" not in text:
        raise ValueError(f"HARD RULE fragment missing {{N}} placeholder:\n{text[:80]}")
    return text.replace("{N}", str(number))


def render_prover_system_prompt(spec: ProverPromptSpec) -> str:
    """Assemble a system prompt from *spec* fragments.

    HARD RULE ``{N}`` placeholders are numbered sequentially for this prompt
    only. Other braces (for example ``{helper_name}``) are left untouched.
    """
    sections: list[str] = [_read_fragment(INTROS_DIR, spec.intro)]

    hard_blocks: list[str] = []
    for index, stem in enumerate(spec.hard_rules, start=1):
        hard_blocks.append(
            _number_hard_rule(_read_fragment(HARD_RULES_DIR, stem), index)
        )
    if hard_blocks:
        sections.append(
            "======================================================================\n"
            "HARD RULES\n"
            "======================================================================\n\n"
            + "\n\n".join(hard_blocks)
        )

    protocol_values = _about_protocol_values()
    protocol_blocks: list[str] = []
    for stem in spec.protocols:
        text = _read_fragment(PROTOCOLS_DIR, stem)
        if stem in _ABOUT_CONSTANT_FRAGMENTS:
            text = _fill_named_placeholders(text, protocol_values)
        protocol_blocks.append(text)
    if protocol_blocks:
        sections.append(
            "======================================================================\n"
            "OUTPUT / INTERACTION PROTOCOLS\n"
            "======================================================================\n\n"
            + "\n\n".join(protocol_blocks)
        )

    soft_blocks: list[str] = []
    for stem in spec.soft_sections:
        text = _read_fragment(SOFT_DIR, stem)
        if stem in _ABOUT_CONSTANT_FRAGMENTS:
            text = _fill_named_placeholders(text, protocol_values)
        soft_blocks.append(text)
    if soft_blocks:
        sections.append(SOFT_INTRO + "\n" + "\n\n".join(soft_blocks))

    if spec.final_check is not None:
        sections.append(_read_fragment(FINAL_CHECKS_DIR, spec.final_check))

    rendered = "\n\n".join(sections).strip()
    if "{N}" in rendered:
        raise ValueError("Unresolved {N} placeholder in assembled system prompt")
    return rendered


@lru_cache(maxsize=None)
def cached_system_prompt(spec: ProverPromptSpec) -> str:
    """Render once per process; specs are frozen and hashable."""
    return render_prover_system_prompt(spec)


# ---------------------------------------------------------------------------
# Declarative agent specs
# ---------------------------------------------------------------------------

_COMMON_SOFT: tuple[str, ...] = ("common_proof_guidance", "authorized_automation")
_HISTORY_SOFT: tuple[str, ...] = ("history_guidance",)
_HELPER_SOFT: tuple[str, ...] = ("decomposition_policy",)

DIRECT_SPEC = ProverPromptSpec(
    intro="direct",
    hard_rules=COMMON_HARD_RULES,
    protocols=("direct_final",),
    soft_sections=(*_COMMON_SOFT, "non_about_library_guidance", "direct_policy"),
    final_check="direct",
)

DIRECT_ABOUT_SPEC = ProverPromptSpec(
    intro="direct_about",
    hard_rules=(*COMMON_HARD_RULES, ABOUT_HARD_RULE),
    protocols=(*ABOUT_PROTOCOLS, "direct_final"),
    soft_sections=(*_COMMON_SOFT, "about_lookup_guidance", "direct_policy"),
    final_check="direct_about",
)

REPAIR_DIRECT_SPEC = ProverPromptSpec(
    intro="repair",
    hard_rules=(*COMMON_HARD_RULES, HISTORY_HARD_RULE),
    protocols=("direct_final",),
    soft_sections=(*_COMMON_SOFT, *_HISTORY_SOFT, "non_about_library_guidance"),
    final_check="repair",
)

REPAIR_DIRECT_ABOUT_SPEC = ProverPromptSpec(
    intro="repair_about",
    hard_rules=(*COMMON_HARD_RULES, HISTORY_HARD_RULE, ABOUT_HARD_RULE),
    protocols=(*ABOUT_PROTOCOLS, "direct_final"),
    soft_sections=(*_COMMON_SOFT, *_HISTORY_SOFT, "about_lookup_guidance"),
    final_check="repair_about",
)

DECOMPOSITION_SPEC = ProverPromptSpec(
    intro="decomposition",
    hard_rules=(*COMMON_HARD_RULES, HISTORY_HARD_RULE, *HELPER_HARD_RULES),
    protocols=("decomposition_final",),
    soft_sections=(
        *_COMMON_SOFT,
        *_HISTORY_SOFT,
        *_HELPER_SOFT,
        "non_about_library_guidance",
    ),
    final_check="decomposition",
)

DECOMPOSITION_ABOUT_SPEC = ProverPromptSpec(
    intro="decomposition_about",
    hard_rules=(
        *COMMON_HARD_RULES,
        HISTORY_HARD_RULE,
        ABOUT_HARD_RULE,
        *HELPER_HARD_RULES,
    ),
    protocols=(*ABOUT_PROTOCOLS, "decomposition_final"),
    soft_sections=(
        *_COMMON_SOFT,
        *_HISTORY_SOFT,
        *_HELPER_SOFT,
        "about_lookup_guidance",
    ),
    final_check="decomposition_about",
)

AGENT_SPECS: dict[str, ProverPromptSpec] = {
    "DirectAgent": DIRECT_SPEC,
    "DirectAboutAgent": DIRECT_ABOUT_SPEC,
    "RepairDirectAgent": REPAIR_DIRECT_SPEC,
    "RepairDirectAboutAgent": REPAIR_DIRECT_ABOUT_SPEC,
    "DecompositionAgent": DECOMPOSITION_SPEC,
    "DecompositionAboutAgent": DECOMPOSITION_ABOUT_SPEC,
}
