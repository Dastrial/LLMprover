"""Source-level proof script manipulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProofScript:
    """A Rocq ``.v`` source string, optionally marking the LLM tactic region.

    ``tactic_start_line`` / ``tactic_end_line`` are 1-based inclusive line
    numbers of the tactic body (between ``Proof.`` and ``Qed.``) inside
    ``code``. Used to remap coqc error loci back to the LLM script.
    """

    code: str
    tactic_start_line: int | None = None
    tactic_end_line: int | None = None

    @classmethod
    def from_file(cls, path: str) -> ProofScript:
        """Read a ``.v`` file and return its contents as a ``ProofScript``."""
        file_path = Path(path)
        if file_path.suffix != ".v":
            raise ValueError(f"Expected a .v file, got: {file_path}")
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
        return cls(code=file_path.read_text(encoding="utf-8"))


def tactic_line_span(prefix: str, script: str) -> tuple[int, int]:
    """Inclusive 1-based line span of *script* after *prefix* in assembled source.

    Matches ``f"{prefix}{script}\\n…"`` as used by ``make_script``.
    """
    start = prefix.count("\n") + 1
    # Trailing ``\\n`` after the tactic body always contributes at least one line
    # (empty script → blank line between ``Proof.`` and ``Qed.``).
    n_lines = f"{script}".count("\n")
    return start, start + n_lines


def shift_tactic_region(
    source: ProofScript, prefix: str
) -> tuple[int | None, int | None]:
    """Return tactic line span after prepending *prefix* to ``source.code``."""
    if source.tactic_start_line is None or source.tactic_end_line is None:
        return None, None
    if not prefix:
        return source.tactic_start_line, source.tactic_end_line
    shift = prefix.count("\n")
    return source.tactic_start_line + shift, source.tactic_end_line + shift
