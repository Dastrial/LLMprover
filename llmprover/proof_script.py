"""Source-level proof script manipulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProofScript:
    code: str

    @classmethod
    def from_file(cls, path: str) -> ProofScript:
        """Read a ``.v`` file and return its contents as a ``ProofScript``."""
        file_path = Path(path)
        if file_path.suffix != ".v":
            raise ValueError(f"Expected a .v file, got: {file_path}")
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
        return cls(code=file_path.read_text(encoding="utf-8"))
