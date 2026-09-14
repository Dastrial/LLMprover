"""Tests for modular prover system-prompt assembly."""

from __future__ import annotations

import re
from unittest.mock import MagicMock

from llmprover.domain import (
    CoqcResult,
    Goal,
    LemmaNode,
    Polarity,
    RocqEnvironment,
)
from llmprover.llm.client import CompletionResult, TokenUsage
from llmprover.prover_agents.about_limits import (
    MAX_SEARCH_COMMANDS,
    MAX_SELECTED_SEARCH_HITS,
    SEARCH_SELECTION_THRESHOLD,
)
from llmprover.prover_agents.direct_about_agent import DirectAboutAgent
from llmprover.prover_agents.prompt_assembly import (
    ABOUT_HARD_RULE,
    AGENT_SPECS,
    COMMON_HARD_RULES,
    DECOMPOSITION_ABOUT_SPEC,
    DECOMPOSITION_SPEC,
    DIRECT_ABOUT_SPEC,
    DIRECT_SPEC,
    HARD_RULES_DIR,
    HELPER_HARD_RULES,
    HISTORY_HARD_RULE,
    REPAIR_DIRECT_ABOUT_SPEC,
    REPAIR_DIRECT_SPEC,
    ProverPromptSpec,
    cached_system_prompt,
    render_prover_system_prompt,
)

HARD_RULE_TITLE_RE = re.compile(r"^HARD RULE (\d+) — (.+)$", re.MULTILINE)


def _titles(prompt: str) -> list[tuple[int, str]]:
    return [(int(m.group(1)), m.group(2)) for m in HARD_RULE_TITLE_RE.finditer(prompt)]


def test_hard_rule_numbering_is_sequential_and_local() -> None:
    direct = render_prover_system_prompt(DIRECT_SPEC)
    repair = render_prover_system_prompt(REPAIR_DIRECT_SPEC)
    assert "{N}" not in direct
    assert "{N}" not in repair
    direct_nums = [n for n, _ in _titles(direct)]
    repair_nums = [n for n, _ in _titles(repair)]
    assert direct_nums == list(range(1, len(COMMON_HARD_RULES) + 1))
    assert repair_nums == list(range(1, len(COMMON_HARD_RULES) + 2))
    # History rule is present only on repair and is locally numbered.
    assert HISTORY_HARD_RULE not in DIRECT_SPEC.hard_rules
    assert "PREVIOUS ROCQ FAILURES ARE CONSTRAINTS" in repair
    assert "PREVIOUS ROCQ FAILURES ARE CONSTRAINTS" not in direct
    history_num = next(
        n
        for n, title in _titles(repair)
        if title == "PREVIOUS ROCQ FAILURES ARE CONSTRAINTS"
    )
    assert history_num == len(COMMON_HARD_RULES) + 1


def test_numbering_does_not_alter_concrete_helper_name_braces() -> None:
    prompt = render_prover_system_prompt(DECOMPOSITION_SPEC)
    assert "{foo}" in prompt
    assert "{N}" not in prompt


def test_capability_specific_hard_rule_inclusion() -> None:
    expected = {
        "DirectAgent": (False, False, False),
        "DirectAboutAgent": (False, True, False),
        "RepairDirectAgent": (True, False, False),
        "RepairDirectAboutAgent": (True, True, False),
        "DecompositionAgent": (True, False, True),
        "DecompositionAboutAgent": (True, True, True),
    }
    for name, (want_hist, want_about, want_helper) in expected.items():
        stems = set(AGENT_SPECS[name].hard_rules)
        assert (HISTORY_HARD_RULE in stems) is want_hist, name
        assert (ABOUT_HARD_RULE in stems) is want_about, name
        assert (set(HELPER_HARD_RULES) <= stems) is want_helper, name
        assert set(COMMON_HARD_RULES) <= stems, name


def test_protocol_inclusion_by_agent_family() -> None:
    direct = render_prover_system_prompt(DIRECT_SPEC)
    decomp = render_prover_system_prompt(DECOMPOSITION_SPEC)
    about = render_prover_system_prompt(DIRECT_ABOUT_SPEC)
    decomp_about = render_prover_system_prompt(DECOMPOSITION_ABOUT_SPEC)

    assert "OUTPUT PROTOCOL — FINAL DIRECT / REPAIR ANSWER" in direct
    assert "HELPERS:" not in direct.split("SOFT RULES")[0]
    assert "OUTPUT PROTOCOL — FINAL ANSWER" in decomp
    assert "HELPERS:" in decomp
    assert "SCRIPT:" in decomp

    for prompt in (about, decomp_about):
        assert "INTERACTION PROTOCOL — ABOUT AGENTS" in prompt
        assert "OUTPUT PROTOCOL — LOOKUP" in prompt
        assert "OUTPUT PROTOCOL — SELECTION" in prompt
        assert str(MAX_SEARCH_COMMANDS) in prompt
        assert str(SEARCH_SELECTION_THRESHOLD) in prompt
        assert str(MAX_SELECTED_SEARCH_HITS) in prompt

    assert "OUTPUT PROTOCOL — FINAL DIRECT / REPAIR ANSWER" in about
    assert "OUTPUT PROTOCOL — FINAL ANSWER" in decomp_about
    assert "HELPERS:" in decomp_about.split("SOFT RULES")[0]


def test_capability_section_titles_in_rendered_prompts() -> None:
    direct = render_prover_system_prompt(DIRECT_SPEC)
    about = render_prover_system_prompt(DIRECT_ABOUT_SPEC)
    repair = render_prover_system_prompt(REPAIR_DIRECT_SPEC)
    decomp = render_prover_system_prompt(DECOMPOSITION_SPEC)

    assert "ABOUT BEFORE LIBRARY USE" in about
    assert "ABOUT BEFORE LIBRARY USE" not in direct

    assert "HISTORY GUIDANCE" in repair
    assert "HISTORY GUIDANCE" not in direct

    assert "DECOMPOSITION POLICY" in decomp
    assert "DECOMPOSITION POLICY" not in direct
    assert "HELPER GENUINE PROGRESS" in decomp
    assert "CURRENT-RESPONSE HELPER REFERENCE PROTOCOL" in decomp


def test_mode_intros_are_agent_specific() -> None:
    assert DIRECT_SPEC.intro == "direct"
    assert DIRECT_ABOUT_SPEC.intro == "direct_about"
    assert REPAIR_DIRECT_SPEC.intro == "repair"
    assert REPAIR_DIRECT_ABOUT_SPEC.intro == "repair_about"
    assert DECOMPOSITION_SPEC.intro == "decomposition"
    assert DECOMPOSITION_ABOUT_SPEC.intro == "decomposition_about"
    assert len({spec.intro for spec in AGENT_SPECS.values()}) == 6


def test_soft_composition_by_capability() -> None:
    soft = {name: set(spec.soft_sections) for name, spec in AGENT_SPECS.items()}
    for name in ("DirectAgent", "DirectAboutAgent"):
        assert "history_guidance" not in soft[name]
        assert "decomposition_policy" not in soft[name]
    for name in (
        "RepairDirectAgent",
        "RepairDirectAboutAgent",
        "DecompositionAgent",
        "DecompositionAboutAgent",
    ):
        assert "history_guidance" in soft[name]
    for name in ("DecompositionAgent", "DecompositionAboutAgent"):
        assert "decomposition_policy" in soft[name]
    for name in ("RepairDirectAgent", "RepairDirectAboutAgent"):
        assert "decomposition_policy" not in soft[name]


def test_soft_rules_marked_as_guidance_only() -> None:
    prompt = render_prover_system_prompt(DIRECT_SPEC)
    assert "SOFT RULES — GUIDANCE ONLY" in prompt


def test_every_hard_rule_file_has_n_placeholder() -> None:
    for path in sorted(HARD_RULES_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        assert "{N}" in text, path.name
        assert text.count("{N}") == 1, path.name


def test_about_system_prompt_shared_across_lookup_and_final() -> None:
    goal = Goal(
        name="t",
        statement="True.",
        environment=RocqEnvironment(header="Require Import Nat."),
    )
    mock = MagicMock()
    mock.complete.side_effect = [
        CompletionResult(text="", usage=TokenUsage(1, 1)),
        CompletionResult(text="exact I.", usage=TokenUsage(2, 2)),
    ]
    agent = DirectAboutAgent(mock)
    agent.checker = MagicMock()
    agent.checker.check_script.return_value = CoqcResult(success=True, stdout="")
    agent.prove(LemmaNode(goal=goal), Polarity.Positive)

    system = cached_system_prompt(DIRECT_ABOUT_SPEC)
    lookup = mock.complete.call_args_list[0].args[0]
    final = mock.complete.call_args_list[1].args[0]
    assert lookup[0].joined_text() == system
    assert final[0].joined_text() == system
    assert lookup[0].parts[-1].cache_breakpoint is True
    assert final[3].parts[0].cache_breakpoint is False


def test_cached_system_prompt_matches_render() -> None:
    assert cached_system_prompt(DIRECT_SPEC) == render_prover_system_prompt(DIRECT_SPEC)


def test_agent_specs_cover_all_six_agents() -> None:
    assert set(AGENT_SPECS) == {
        "DirectAgent",
        "DirectAboutAgent",
        "RepairDirectAgent",
        "RepairDirectAboutAgent",
        "DecompositionAgent",
        "DecompositionAboutAgent",
    }
    assert AGENT_SPECS["DirectAgent"] is DIRECT_SPEC
    assert AGENT_SPECS["DirectAboutAgent"] is DIRECT_ABOUT_SPEC
    assert AGENT_SPECS["RepairDirectAgent"] is REPAIR_DIRECT_SPEC
    assert AGENT_SPECS["RepairDirectAboutAgent"] is REPAIR_DIRECT_ABOUT_SPEC
    assert AGENT_SPECS["DecompositionAgent"] is DECOMPOSITION_SPEC
    assert AGENT_SPECS["DecompositionAboutAgent"] is DECOMPOSITION_ABOUT_SPEC


def test_custom_spec_numbering_starts_at_one() -> None:
    spec = ProverPromptSpec(
        intro="direct",
        hard_rules=(HISTORY_HARD_RULE, ABOUT_HARD_RULE),
        protocols=("direct_final",),
        soft_sections=(),
        final_check=None,
    )
    prompt = render_prover_system_prompt(spec)
    nums = [n for n, _ in _titles(prompt)]
    assert nums == [1, 2]
