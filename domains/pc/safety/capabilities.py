"""Lock 2 — Capability-based permissions: each agent gets ONLY what it needs.

Not 'access to the PC': a capability = (allowed actions, allowed paths).
DEFAULT-DENY policy: whatever is not explicitly granted is denied.
Every denial is reported with its reason (auditable).
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import PurePosixPath


@dataclass(frozen=True)
class Capability:
    actions: frozenset[str]     # e.g. {"move", "mkdir"}
    paths: tuple[str, ...]      # globs RELATIVE to the domain root: "descargas/**"

    @staticmethod
    def make(actions, paths) -> "Capability":
        return Capability(frozenset(actions), tuple(paths))


class TraversalError(ValueError):
    """The path tries to leave the domain root with '..'."""


def _norm(path: str) -> str:
    """Normalizes to relative posix, COLLAPSING '..' — without this, a scope like
    'descargas/**' would match 'descargas/../secretos/x' (bypassing the lock).

    If after collapsing the path leaves the root (it starts with '..'), it is a
    traversal: it is rejected loudly; a string that could match a glob is NOT
    returned.
    """
    raw = str(path).replace("\\", "/")
    parts: list[str] = []
    for seg in PurePosixPath(raw).parts:
        if seg in ("", ".", "/"):
            continue
        if seg == "..":
            if not parts:
                raise TraversalError(f"la ruta se sale de la raíz del dominio: {path}")
            parts.pop()
        else:
            parts.append(seg)
    return "/".join(parts)


def _match(path_norm: str, pattern: str) -> bool:
    # "dir/**" must also cover "dir" and everything inside it.
    if pattern.endswith("/**"):
        base = pattern[:-3]
        return path_norm == base or path_norm.startswith(base + "/")
    return fnmatch(path_norm, pattern)


class CapabilityGuard:
    def __init__(self, grants: dict[str, list[Capability]] | None = None):
        self.grants = grants or {}

    def authorize(self, agent_id: str, action) -> tuple[bool, str]:
        """Can this agent execute this action? (action = PCAction).

        ALL the paths the action touches must be covered by some capability of
        the agent that includes that kind of action. Paths are normalized by
        COLLAPSING '..' before matching: a traversal attempt is denied (a glob
        is not given the benefit of the doubt).
        """
        caps = self.grants.get(agent_id, [])
        if not caps:
            return False, f"'{agent_id}' no tiene ninguna capacidad concedida (default-deny)"

        relevant = [c for c in caps if action.kind in c.actions]
        if not relevant:
            return False, f"'{agent_id}' no tiene la capacidad '{action.kind}'"

        for path in action.touched_paths():
            try:
                norm = _norm(path)
            except TraversalError as e:
                return False, f"'{agent_id}': {e}"
            if not any(_match(norm, pat) for c in relevant for pat in c.paths):
                return False, (f"'{agent_id}' no puede '{action.kind}' sobre "
                               f"'{norm}' (fuera de sus rutas concedidas)")
        return True, "ok"
