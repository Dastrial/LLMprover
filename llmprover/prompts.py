"""Utilities for loading and filling prompt templates."""

from __future__ import annotations

from pathlib import Path


def load_prompt(path: str | Path) -> str:
    """Read a prompt file."""
    return Path(path).read_text(encoding="utf-8").strip()


def fill_prompt(template: str, **values: str) -> str:
    """Replace ``{key}`` placeholders in *template*."""
    result = template
    for key, value in values.items():
        result = result.replace(f"{{{key}}}", value)
    return result
