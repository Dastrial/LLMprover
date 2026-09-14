"""Utilities for filling prompts and representing cache breakpoints.

Cache layout used by proof agents (provider-specific translation in
``llm_client``):

    PRE-PROMPT -> breakpoint
    HEADER (+ specs) + HISTORY -> breakpoint
    STATEMENT (+ About / dynamic suffix) -> not explicitly cached by us
"""

from __future__ import annotations

from dataclasses import dataclass


def fill_prompt(template: str, **values: str) -> str:
    """Replace ``{key}`` placeholders in *template*."""
    result = template
    for key, value in values.items():
        result = result.replace(f"{{{key}}}", value)
    return result


@dataclass(frozen=True)
class PromptPart:
    """One text segment, optionally ending an explicit cacheable prefix."""

    text: str
    cache_breakpoint: bool = False


@dataclass(frozen=True)
class PromptMessage:
    """One chat message as ordered text parts (roles stay provider-facing)."""

    role: str
    parts: tuple[PromptPart, ...]

    @classmethod
    def text(
        cls, role: str, text: str, *, cache_breakpoint: bool = False
    ) -> PromptMessage:
        """Build a single-part message."""
        return cls(
            role=role, parts=(PromptPart(text=text, cache_breakpoint=cache_breakpoint),)
        )

    def joined_text(self) -> str:
        """Concatenate part texts (ignores breakpoint markers)."""
        return "".join(part.text for part in self.parts)
