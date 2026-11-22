"""domains/code/tools.py — Tools that the code agents can invoke.

Each tool is a function. The agent can only use the ones assigned to it in
config/swarm.yaml (principle of least privilege, right from Phase 1).

The tools are built with `build_tools(project_root)`: they are BOUND to the
project root and reject any path that leaves it (no `..` going outside, no
absolute paths pointing elsewhere). In code this is hygiene; in PC (Phase 4)
this same confinement is one of the locks.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable


def build_tools(project_root: str) -> dict[str, Callable]:
    """Returns the COMPLETE catalog of code tools, bound to the root.

    The swarm then hands each agent only the subset its config allows.
    """
    root = Path(project_root).resolve()

    def _safe(path: str) -> Path:
        p = (root / path).resolve()
        if not (p == root or root in p.parents):
            raise PermissionError(f"Ruta fuera del proyecto: {path}")
        return p

    def read_file(path: str) -> str:
        p = _safe(path)
        if not p.is_file():
            return f"ERROR: no existe {path}"
        return p.read_text(encoding="utf-8", errors="replace")

    def write_file(path: str, content: str) -> None:
        p = _safe(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    def apply_patch(path: str, content: str) -> None:
        # Phase 1: "patch" = full-content replacement (robust with small
        # models). A unified-diff version can come later.
        write_file(path, content)

    def grep(pattern: str, path: str = ".") -> list[str]:
        base = _safe(path)
        hits: list[str] = []
        files = [base] if base.is_file() else base.rglob("*")
        for f in files:
            if not f.is_file() or _is_binary(f):
                continue
            try:
                for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if pattern in line:
                        hits.append(f"{f.relative_to(root)}:{i}: {line.strip()}")
            except OSError:
                continue
        return hits[:200]  # cap so as not to flood the agent's context

    def list_dir(path: str = ".") -> list[str]:
        p = _safe(path)
        if not p.is_dir():
            return [f"ERROR: no es directorio {path}"]
        return sorted(
            (c.name + ("/" if c.is_dir() else "")) for c in p.iterdir()
        )

    def run_tests() -> dict:
        return _run([_py(), "-m", "pytest", "-q"], root)

    def run_linter() -> dict:
        # Tries ruff; if it is missing, falls back to a compile check with py_compile.
        res = _run(["ruff", "check", "."], root)
        if res.get("not_found"):
            res = _run([_py(), "-m", "compileall", "-q", "."], root)
        return res

    def diff() -> str:
        res = _run(["git", "diff"], root)
        return res.get("stdout", "")

    catalog = {
        "read_file": read_file,
        "write_file": write_file,
        "apply_patch": apply_patch,
        "grep": grep,
        "list_dir": list_dir,
        "run_tests": run_tests,
        "run_linter": run_linter,
        "diff": diff,
    }
    return catalog


# --- helpers ---------------------------------------------------------------

import sys


def _py() -> str:
    return sys.executable or "python3"


def _run(cmd: list[str], cwd: Path, timeout: int = 300) -> dict:
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
    except FileNotFoundError:
        return {"ok": False, "not_found": True, "cmd": cmd[0],
                "stdout": "", "stderr": f"{cmd[0]} no instalado"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "timeout": True, "stdout": "", "stderr": "timeout"}
    except OSError as e:
        # The SYSTEM could not launch the process (e.g. WinError 1455 from memory
        # pressure). This says NOTHING about the evaluated code: it is flagged
        # separately so the fitness can refuse to score with a broken environment.
        return {"ok": False, "infra_error": True, "stdout": "",
                "stderr": f"fallo al lanzar {cmd[0]}: {e}"}
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }


def _is_binary(path: Path) -> bool:
    try:
        chunk = path.read_bytes()[:1024]
    except OSError:
        return True
    return b"\x00" in chunk


# Tool contract (documentation only). The executable catalog is provided by
# build_tools(project_root); each function is bound to that root:
#   read_file(path) -> str
#   write_file(path, content) -> None        # mutation: applied by the adapter
#   apply_patch(path, content) -> None       # mutation: applied by the adapter
#   grep(pattern, path=".") -> list[str]
#   list_dir(path=".") -> list[str]
#   run_tests() -> dict                       # {ok, returncode, stdout, stderr}
#   run_linter() -> dict
#   diff() -> str
