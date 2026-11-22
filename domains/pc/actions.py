"""domains/pc/actions.py — The action model of the PC domain.

Agents do NOT execute mutations: they request them with the usual TOOL: syntax
(core/agent.py records them in tool_calls without executing them) and the
PCAdapter turns them into PCAction objects to run them through the 4 locks.

Types supported in Phase 4 (low risk, file system only):
    write  {path, content}   mkdir  {path}
    move   {src, dst}        copy   {src, dst}       delete {path}
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


MUTATING_PC_TOOLS = {"write_file", "move_file", "copy_file", "delete_file", "mkdir"}

_TOOL_TO_KIND = {
    "write_file": "write", "move_file": "move", "copy_file": "copy",
    "delete_file": "delete", "mkdir": "mkdir",
}


@dataclass
class PCAction:
    kind: str
    args: dict

    def touched_paths(self) -> list[str]:
        keys = ("path", "src", "dst")
        return [str(self.args[k]) for k in keys if k in self.args]

    def __str__(self):
        return f"{self.kind} {self.args}"


def extract_actions(result) -> list[PCAction]:
    """Converts what the agent proposed (mutating tool_calls + file blocks)
    into the ordered list of actions to validate."""
    actions: list[PCAction] = []
    for call in getattr(result, "tool_calls", []):
        name = call.get("name")
        if name in _TOOL_TO_KIND:
            actions.append(PCAction(_TOOL_TO_KIND[name], dict(call.get("args", {}))))
    for change in getattr(result, "proposed_changes", []):
        actions.append(PCAction("write", {"path": change.path, "content": change.content}))
    return actions


def apply_action(action: PCAction, root: Path, kill=None) -> dict:
    """Executes ONE action inside root (jail included). If `kill` (KillSwitch)
    is passed, it backs up and records in the journal so the action can be undone.

    This is the ONLY executor: the sandbox and live mode use exactly the same
    code; only the root they act on changes.
    """
    import shutil

    def safe(p: str) -> Path:
        q = (root / str(p)).resolve()
        if not (q == root.resolve() or root.resolve() in q.parents):
            raise PermissionError(f"Ruta fuera de la raíz del dominio: {p}")
        return q

    a = action.args
    if action.kind == "write":
        dst = safe(a["path"])
        undo = None
        if kill is not None:
            backup = kill.backup_file(dst)
            undo = ({"kind": "restore_file", "path": str(dst), "backup": backup}
                    if backup else {"kind": "remove_file", "path": str(dst)})
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(a.get("content", ""), encoding="utf-8")
        if kill is not None:
            kill.record(action.kind, a, undo)
        return {"ok": True}

    if action.kind == "mkdir":
        dst = safe(a["path"])
        existed = dst.exists()
        dst.mkdir(parents=True, exist_ok=True)
        if kill is not None and not existed:
            kill.record(action.kind, a, {"kind": "remove_dir", "path": str(dst)})
        return {"ok": True}

    if action.kind == "move":
        src, dst = safe(a["src"]), safe(a["dst"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        if kill is not None:
            kill.record(action.kind, a,
                        {"kind": "move_back", "src": str(dst), "dst": str(src)})
        return {"ok": True}

    if action.kind == "copy":
        src, dst = safe(a["src"]), safe(a["dst"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
        if kill is not None:
            kill.record(action.kind, a, {"kind": "remove_file", "path": str(dst)})
        return {"ok": True}

    if action.kind == "delete":
        target = safe(a["path"])
        undo = None
        if kill is not None:
            backup = kill.backup_file(target)
            undo = {"kind": "restore_file", "path": str(target), "backup": backup}
        target.unlink()
        if kill is not None:
            kill.record(action.kind, a, undo)
        return {"ok": True}

    return {"ok": False, "error": f"acción desconocida: {action.kind}"}
