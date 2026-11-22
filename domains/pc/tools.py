"""domains/pc/tools.py — Tools of the PC domain.

READ tools are executed inside act() (confined to the root). MUTATING tools
(move/copy/delete/write/mkdir) are NEVER executed there: the agent requests them
with TOOL:, they are recorded as proposals, and the PCAdapter runs them through
the 4 locks. See domains/pc/actions.py::MUTATING_PC_TOOLS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable


def build_pc_tools(root: str) -> dict[str, Callable]:
    base = Path(root).resolve()

    def _safe(path: str) -> Path:
        p = (base / str(path)).resolve()
        if not (p == base or base in p.parents):
            raise PermissionError(f"Ruta fuera de la raíz del dominio: {path}")
        return p

    def list_dir(path: str = ".") -> list[str]:
        p = _safe(path)
        if not p.is_dir():
            return [f"ERROR: no es directorio {path}"]
        return sorted(c.name + ("/" if c.is_dir() else "") for c in p.iterdir())

    def read_file(path: str) -> str:
        p = _safe(path)
        if not p.is_file():
            return f"ERROR: no existe {path}"
        return p.read_text(encoding="utf-8", errors="replace")[:4000]

    def stat(path: str) -> dict:
        p = _safe(path)
        if not p.exists():
            return {"exists": False}
        return {"exists": True, "dir": p.is_dir(), "bytes": p.stat().st_size if p.is_file() else None}

    # Mutating tools: registered so that the agent CAN propose them (the prompt
    # lists them), but the agent loop never executes them.
    def _proposal_only(**kwargs):  # pragma: no cover - must never be called
        raise RuntimeError("Herramienta mutante: la aplica el PCAdapter tras los candados")

    return {
        "list_dir": list_dir,
        "read_file": read_file,
        "stat": stat,
        "write_file": _proposal_only,
        "move_file": _proposal_only,
        "copy_file": _proposal_only,
        "delete_file": _proposal_only,
        "mkdir": _proposal_only,
    }
