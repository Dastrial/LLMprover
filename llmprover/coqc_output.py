from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

# coqc: File "path", line N, characters A-B:
COQC_LOCUS_RE = re.compile(
    r'^File "(?P<path>[^"]+)", line (?P<line>\d+), '
    r"characters (?P<start>\d+)-(?P<end>\d+):",
    re.MULTILINE,
)

ASSUMPTION_NAME_RE = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_']*(?:\.[A-Za-z_][A-Za-z0-9_']*)*)\s*:"
)


@dataclass(frozen=True)
class CoqcLocation:
    """One ``File …, line N, characters A-B:`` locus from coqc stderr.

    ``line`` / ``start_char`` / ``end_char`` are the coordinates coqc reported
    in the compiled ``.v`` file (1-based line, 0-based character range).

    When the compiled file wraps an LLM tactic script, ``script_line`` is the
    1-based line inside that tactic body (``None`` if the locus falls outside).
    """

    line: int
    start_char: int
    end_char: int
    script_line: int | None = None


# Locus labels that may precede a coqc Warning/Error message (raw or remapped).
_COQC_FILE_LOCUS_RE = re.compile(r'^File "[^"]+", line \d+, characters \d+-\d+:$')
_COQC_SCRIPT_LOCUS_RE = re.compile(r"^Script line \d+, characters \d+-\d+:$")
OUTSIDE_TACTIC_LOCUS = "Generated Rocq code outside the tactic script:"


def _is_coqc_locus_line(line: str) -> bool:
    stripped = line.strip()
    return (
        bool(_COQC_FILE_LOCUS_RE.match(stripped))
        or bool(_COQC_SCRIPT_LOCUS_RE.match(stripped))
        or stripped == OUTSIDE_TACTIC_LOCUS
    )


def _is_coqc_warning_line(line: str) -> bool:
    return line.lstrip().startswith("Warning:")


def _is_coqc_error_line(line: str) -> bool:
    return line.lstrip().startswith("Error:")


def _skip_coqc_warning_block(lines: list[str], start: int) -> int:
    """Index just past a Warning block that begins at *start* (locus or Warning:)."""
    i = start
    if _is_coqc_locus_line(lines[i]):
        i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1
    if i < len(lines) and _is_coqc_warning_line(lines[i]):
        i += 1
    while i < len(lines):
        if (
            _is_coqc_locus_line(lines[i])
            or _is_coqc_warning_line(lines[i])
            or _is_coqc_error_line(lines[i])
        ):
            break
        i += 1
    return i


def strip_coqc_warnings(text: str) -> str:
    """Remove coqc ``Warning:`` blocks (and their preceding locus lines) from *text*."""
    if not text:
        return text
    lines = text.splitlines()
    kept: list[str] = []
    i = 0
    while i < len(lines):
        if _is_coqc_locus_line(lines[i]):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and _is_coqc_warning_line(lines[j]):
                i = _skip_coqc_warning_block(lines, i)
                continue
        elif _is_coqc_warning_line(lines[i]):
            i = _skip_coqc_warning_block(lines, i)
            continue
        kept.append(lines[i])
        i += 1
    return "\n".join(kept).strip()


def parse_coqc_locations(
    stderr: str,
    *,
    tactic_start_line: int | None = None,
    tactic_end_line: int | None = None,
) -> list[CoqcLocation]:
    """Parse ``File …, line N, characters A-B:`` loci from coqc stderr.

    When *tactic_start_line* / *tactic_end_line* mark the LLM tactic body in the
    compiled file, each locus inside that span gets a 1-based ``script_line``.
    """
    locations: list[CoqcLocation] = []
    for match in COQC_LOCUS_RE.finditer(stderr):
        line = int(match.group("line"))
        start_char = int(match.group("start"))
        end_char = int(match.group("end"))
        script_line: int | None = None
        if tactic_start_line is not None and tactic_end_line is not None:
            if tactic_start_line <= line <= tactic_end_line:
                script_line = line - tactic_start_line + 1
        locations.append(
            CoqcLocation(
                line=line,
                start_char=start_char,
                end_char=end_char,
                script_line=script_line,
            )
        )
    return locations


def remap_coqc_stderr(stderr: str, locations: Sequence[CoqcLocation]) -> str:
    """Rewrite coqc File/line loci to script-relative labels.

    Loci with ``script_line`` become ``Script line N, characters A-B:``.
    Loci outside the tactic body drop file coordinates (the LLM does not see
    the generated ``.v``) and become ``OUTSIDE_TACTIC_LOCUS``. Other stderr
    lines are left unchanged.
    """
    if not locations:
        return stderr
    loc_iter = iter(locations)

    def replace(match: re.Match[str]) -> str:
        loc = next(loc_iter, None)
        if loc is None:
            return match.group(0)
        if loc.script_line is not None:
            return (
                f"Script line {loc.script_line}, "
                f"characters {loc.start_char}-{loc.end_char}:"
            )
        return OUTSIDE_TACTIC_LOCUS

    return COQC_LOCUS_RE.sub(replace, stderr)


@dataclass
class CoqcResult:
    """Outcome of running ``coqc`` on a script.

    ``stderr`` is the raw coqc stream. ``locations`` holds parsed loci, with
    optional ``script_line`` remapping into the LLM tactic script.
    ``script_stderr`` is stderr with loci rewritten to script coordinates when
    remapping was applied; ``output`` prefers it for LLM / verbose display.
    """

    success: bool
    stdout: str = ""
    stderr: str = ""
    locations: list[CoqcLocation] = field(default_factory=list)
    script_stderr: str | None = None

    @property
    def display_stderr(self) -> str:
        """Stderr for humans / LLMs (script-relative loci when available)."""
        return self.stderr if self.script_stderr is None else self.script_stderr

    @property
    def output(self) -> str:
        """Full coqc output (stdout + display stderr) — pass back to the LLM on failure."""
        return "\n".join(
            part for part in (self.stdout, self.display_stderr) if part
        ).strip()

    @property
    def output_without_warnings(self) -> str:
        """Like ``output``, but with coqc Warning blocks removed."""
        return strip_coqc_warnings(self.output)

    def __str__(self) -> str:
        return (
            f"CoqcResult(success={self.success}, stdout={self.stdout}, "
            f"stderr={self.stderr}, locations={self.locations!r})"
        )


def parse_print_assumptions(output: str) -> set[str]:
    """Extract fully qualified axiom / admit names from ``Print Assumptions`` output.

    Returns an empty set when Rocq reports ``Closed under the global context``.
    """
    if "Closed under the global context" in output:
        return set()

    names: set[str] = set()
    in_axioms = False
    for line in output.splitlines():
        stripped = line.strip()
        if stripped == "Axioms:":
            in_axioms = True
            continue
        if not in_axioms:
            continue
        match = ASSUMPTION_NAME_RE.match(stripped)
        if match:
            name = match.group(1)
            if name in {"Warning", "Error", "File"}:
                continue
            names.add(name)
    return names
