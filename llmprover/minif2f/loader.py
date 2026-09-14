"""Load miniF2F-rocq from HuggingFace into LLMprover ``Goal``s."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from llmprover.domain import Goal, RocqEnvironment

DEFAULT_DATASET = "LLM4Rocq/miniF2F-rocq"

THEOREM_HEAD_RE = re.compile(
    r"^\s*(?:Theorem|Lemma|Corollary|Proposition)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_']*)\s*"
    r"(?P<rest>.*)\s*$",
    re.DOTALL,
)
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_']*")


@dataclass(frozen=True)
class MiniF2FProblem:
    """One miniF2F-rocq row, ready to turn into a ``Goal``."""

    name: str
    split: str
    statement: str
    header: str
    informal_statement: str = ""
    informal_proof: str = ""

    def format_statement(self) -> str:
        """Return ``name : statement`` (display / decomposition form)."""
        return f"{self.name}: {self.statement}"

    def to_goal(self) -> Goal:
        """Build a ``Goal`` for proof search.

        ``allowed_axioms`` is left empty: ``check_attempt`` validates
        assumptions against ``header`` at check time.
        """
        return Goal(
            name=self.name,
            statement=self.statement,
            environment=RocqEnvironment(header=self.header),
        )


def split_top_level_colon(text: str) -> tuple[str, str] | None:
    """Split *text* at the first ``:`` outside parentheses / brackets."""
    depth = 0
    for i, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == ":" and depth == 0:
            return text[:i].strip(), text[i + 1 :].strip()
    return None


def parse_binder_tokens(binders_text: str) -> list[str]:
    """Tokenize binders: ``(x : T)`` groups or bare identifiers (``a``).

    miniF2F only uses these two forms (no ``{x}`` / ``[x]`` observed).
    """
    tokens: list[str] = []
    i = 0
    n = len(binders_text)
    while i < n:
        if binders_text[i].isspace():
            i += 1
            continue
        if binders_text[i] == "(":
            depth = 0
            start = i
            while i < n:
                if binders_text[i] == "(":
                    depth += 1
                elif binders_text[i] == ")":
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                i += 1
            else:
                raise ValueError(f"Unbalanced binder parentheses: {binders_text!r}")
            tokens.append(binders_text[start:i])
            continue
        match = IDENT_RE.match(binders_text, i)
        if match:
            tokens.append(match.group(0))
            i = match.end()
            continue
        raise ValueError(
            "Unsupported binder syntax in miniF2F statement "
            f"(expected '(…)' or identifier): {binders_text!r}"
        )
    return tokens


def parse_rocq_statement(rocq_statement: str) -> str:
    """Extract a Gallina formula from HF ``rocq_statement``.

    Accepts ``Theorem|Lemma|… name [binders] : body.`` and returns *body*,
    rewriting binders before ``:`` as ``forall``:

    - ``Theorem foo (n : nat) : n + 0 = n.``
      → ``forall (n : nat), n + 0 = n``
    - ``Theorem foo (n : nat) (h : n > 0) : P.``
      → ``forall (n : nat) (h : n > 0), P``
    - ``Theorem foo a : P.`` (inferred type)
      → ``forall a, P``

    The theorem name is ignored (callers use the HF ``name`` field). Trailing
    ``.`` is stripped.

    Raises:
        ValueError: if the text is not a parseable theorem declaration
            (corrupt HF rows are skipped by ``load_minif2f``).
    """
    text = (rocq_statement or "").strip()
    match = THEOREM_HEAD_RE.match(text)
    if not match:
        raise ValueError(f"Not a Theorem/Lemma declaration: {text[:80]!r}")

    rest = match.group("rest")
    split = split_top_level_colon(rest)
    if split is None:
        raise ValueError(f"Missing ':' in theorem declaration: {text[:80]!r}")

    binders_text, body = split
    if body.endswith("."):
        body = body[:-1].rstrip()
    if not body:
        raise ValueError(f"Empty theorem body: {text[:80]!r}")

    binders_text = binders_text.strip()
    if not binders_text:
        return body

    tokens = parse_binder_tokens(binders_text)
    return f"forall {' '.join(tokens)}, {body}"


def problem_from_row(row: Mapping[str, Any]) -> MiniF2FProblem:
    """Parse one HF dataset row into a ``MiniF2FProblem``.

    Raises:
        ValueError: if ``rocq_statement`` cannot be parsed.
    """
    name = str(row.get("name") or "unnamed")
    header = str(row.get("header") or "")
    statement = parse_rocq_statement(str(row.get("rocq_statement") or ""))
    return MiniF2FProblem(
        name=name,
        split=str(row.get("split") or ""),
        statement=statement,
        header=header,
        informal_statement=str(row.get("informal_statement") or ""),
        informal_proof=str(row.get("informal_proof") or ""),
    )


def load_minif2f(
    *,
    dataset_id: str = DEFAULT_DATASET,
    splits: Sequence[str] | None = None,
) -> dict[str, list[MiniF2FProblem]]:
    """Download / load miniF2F-rocq from HuggingFace.

    Returns ``split_name -> problems``. Rows whose ``rocq_statement`` cannot
    be parsed (e.g. ``amc12_2001_p5``) are skipped.
    """
    try:
        import datasets
    except ImportError as exc:  # pragma: no cover - env dependent
        raise ImportError(
            "Loading miniF2F requires the 'datasets' package. "
            "Install with: pip install 'llmprover[bench]' "
            "or: pip install datasets"
        ) from exc

    raw = datasets.load_dataset(dataset_id)
    selected = splits if splits is not None else list(raw.keys())
    result: dict[str, list[MiniF2FProblem]] = {}
    for split in selected:
        problems: list[MiniF2FProblem] = []
        for row in raw[split]:
            try:
                problems.append(problem_from_row(row))
            except ValueError:
                continue
        result[split] = problems
    return result


def load_minif2f_goals(
    *,
    dataset_id: str = DEFAULT_DATASET,
    splits: Sequence[str] | None = None,
) -> dict[str, list[Goal]]:
    """Load miniF2F and convert every problem to a ``Goal``."""
    problems = load_minif2f(dataset_id=dataset_id, splits=splits)
    return {split: [p.to_goal() for p in items] for split, items in problems.items()}


def iter_problems(
    problems_by_split: Mapping[str, Sequence[MiniF2FProblem]],
) -> Iterable[MiniF2FProblem]:
    """Flatten split → problem lists."""
    for problems in problems_by_split.values():
        yield from problems
