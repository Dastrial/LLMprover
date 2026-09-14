from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from llmprover.domain import AttemptRecord, Goal, ProofAttempt

_IDENT_CHAR = r"A-Za-z0-9_'"


def substitute_helper_placeholders(script: str, mapping: Mapping[str, str]) -> str:
    """Replace ``{helper_name}`` placeholders with the mapped lemma names.

    Only known helper names are rewritten, longest name first, so ``{helper}``
    does not clobber ``{helper_1}``. Bare Rocq braces that are not a helper
    placeholder are left unchanged.
    """
    if not mapping:
        return script
    pattern = re.compile(
        r"\{("
        + "|".join(re.escape(name) for name in sorted(mapping, key=len, reverse=True))
        + r")\}"
    )
    return pattern.sub(lambda match: mapping[match.group(1)], script)


def bare_new_helper_names(script: str, helper_names: Sequence[str]) -> list[str]:
    """Return newly proposed helper names used as bare identifiers in *script*.

    ``{name}`` placeholders are ignored. Identifier boundaries follow the
    helper-name grammar ``[A-Za-z_][A-Za-z0-9_']*``, so ``helper_2`` is not a
    reference to ``helper``. Unrelated Rocq braces are left alone.
    """
    if not helper_names:
        return []
    names = list(dict.fromkeys(helper_names))
    placeholders = re.compile(
        r"\{("
        + "|".join(re.escape(name) for name in sorted(names, key=len, reverse=True))
        + r")\}"
    )
    remainder = placeholders.sub("", script)
    bare: list[str] = []
    for name in names:
        ident = re.compile(rf"(?<![{_IDENT_CHAR}]){re.escape(name)}(?![{_IDENT_CHAR}])")
        if ident.search(remainder):
            bare.append(name)
    return bare


def format_bare_helper_error(names: Sequence[str]) -> str:
    """Actionable protocol error for bare references to newly proposed helpers."""
    if len(names) == 1:
        name = names[0]
        return (
            f"Invalid helper reference: newly proposed helper '{name}' "
            f"must be referenced as '{{{name}}}' in SCRIPT, not as bare '{name}'."
        )
    quoted = ", ".join(f"'{name}'" for name in names)
    braced = ", ".join(f"'{{{name}}}'" for name in names)
    return (
        f"Invalid helper reference: newly proposed helpers {quoted} "
        f"must be referenced as {braced} in SCRIPT, not as bare identifiers."
    )


def new_helper_protocol_error(attempt: ProofAttempt) -> str | None:
    """Return a protocol error when a helper introduced by this attempt
    is referenced without braces in its script.

    Identifiers not listed in ``attempt.new_lemmas`` are ignored.
    """
    bare = bare_new_helper_names(
        attempt.script, [lemma.name for lemma in attempt.new_lemmas]
    )
    if not bare:
        return None
    return format_bare_helper_error(bare)


def resolve_attempt_helpers(
    attempt: ProofAttempt,
    helper_names: Sequence[str] | None = None,
) -> ProofAttempt:
    """Return an attempt whose helper placeholders and proposed helper names
    are replaced by ``helper_names``.

    If several proposed helpers map to the same name, keep only the first
    corresponding ``Goal`` in ``new_lemmas``.
    """
    names = (
        list(helper_names)
        if helper_names is not None
        else [lemma.name for lemma in attempt.new_lemmas]
    )
    if len(names) != len(attempt.new_lemmas):
        raise ValueError(
            "helper_names must have one entry per new lemma, "
            f"got {len(names)} for {len(attempt.new_lemmas)} lemmas"
        )
    mapping = {
        lemma.name: name for lemma, name in zip(attempt.new_lemmas, names, strict=True)
    }
    script = substitute_helper_placeholders(attempt.script, mapping)
    resolved_lemmas: list[Goal] = []
    seen: set[str] = set()
    for lemma, name in zip(attempt.new_lemmas, names, strict=True):
        if name in seen:
            continue
        seen.add(name)
        if name == lemma.name:
            resolved_lemmas.append(lemma)
        else:
            resolved_lemmas.append(
                Goal(
                    name=name,
                    statement=lemma.statement,
                    environment=lemma.environment,
                )
            )
    if script == attempt.script and resolved_lemmas == attempt.new_lemmas:
        return attempt
    return ProofAttempt(
        goal=attempt.goal,
        polarity=attempt.polarity,
        script=script,
        new_lemmas=resolved_lemmas,
        agent=attempt.agent,
        about_lemmas=attempt.about_lemmas,
        search_about_output=attempt.search_about_output,
        select_output=attempt.select_output,
    )


def script_with_child_names(record: AttemptRecord) -> str:
    """Return the attempt script with helper placeholders replaced by child
    goal names, falling back to the proposed helper names when no complete
    child mapping is available.
    """
    mapping = {lemma.name: lemma.name for lemma in record.attempt.new_lemmas}
    if len(record.lemmas) == len(record.attempt.new_lemmas):
        mapping.update(
            {
                helper.name: child.goal.name
                for helper, child in zip(
                    record.attempt.new_lemmas, record.lemmas, strict=True
                )
            }
        )
    return substitute_helper_placeholders(record.attempt.script, mapping)
