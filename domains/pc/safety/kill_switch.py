"""Lock 4 — Kill switch + rollback: kill the swarm and undo what it did.

Kill switch: a flag file (.swarm/KILL). Anything — a human, a monitor, another
process — can create it; the adapter checks it BEFORE each action and aborts
with KillSwitchActive. It is activated with `KillSwitch.activate()` or by hand:
    echo stop > .swarm/KILL

Rollback: every REAL action goes into an append-only journal (.swarm/pc_journal.jsonl)
with its undo information (previous content backed up in .swarm/pc_backups/).
`rollback()` walks the journal in reverse order and restores the previous state.

This is what turns 'software that acts on your PC' into 'bounded, reversible
expansion'.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path


class KillSwitchActive(RuntimeError):
    """The swarm is stopped by the emergency stop."""


class KillSwitch:
    def __init__(self, state_dir: str | Path = ".swarm"):
        self.state_dir = Path(state_dir)
        self.flag = self.state_dir / "KILL"
        self.journal = self.state_dir / "pc_journal.jsonl"
        self.backups = self.state_dir / "pc_backups"

    # ---- emergency stop ---------------------------------------------------

    def activate(self, reason: str = "activado manualmente") -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.flag.write_text(f"{time.time()} {reason}\n", encoding="utf-8")

    def clear(self) -> None:
        self.flag.unlink(missing_ok=True)

    def is_active(self) -> bool:
        return self.flag.exists()

    def check(self) -> None:
        """Call BEFORE each real action. Aborts if a stop is active."""
        if self.is_active():
            raise KillSwitchActive(
                f"Kill switch activo ({self.flag}). Nada se ejecuta hasta "
                f"limpiarlo explícitamente con clear().")

    # ---- journal + rollback --------------------------------------------------

    def backup_file(self, path: Path) -> str | None:
        """Backs up a file before touching it. Returns the path of the backup
        (or None if the file did not exist)."""
        if not path.exists():
            return None
        self.backups.mkdir(parents=True, exist_ok=True)
        dest = self.backups / f"{uuid.uuid4().hex}_{path.name}"
        shutil.copy2(path, dest)
        return str(dest)

    def record(self, kind: str, args: dict, undo: dict) -> None:
        """Records an APPLIED action and how to undo it."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        row = {"ts": time.time(), "kind": kind, "args": args, "undo": undo}
        with self.journal.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def rollback(self) -> list[str]:
        """Undoes EVERYTHING recorded, in reverse order. Returns the details.
        When done, it archives the journal (.rolled_back is kept for auditing)."""
        if not self.journal.exists():
            return []
        rows = [json.loads(l) for l in
                self.journal.read_text(encoding="utf-8").splitlines() if l.strip()]
        undone: list[str] = []
        for row in reversed(rows):
            undo = row["undo"]
            kind = undo.get("kind")
            try:
                if kind == "restore_file":       # write/delete on an existing file
                    target = Path(undo["path"])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(undo["backup"], target)
                elif kind == "remove_file":      # write that created a new file
                    Path(undo["path"]).unlink(missing_ok=True)
                elif kind == "move_back":        # move: put it back where it was
                    src, dst = Path(undo["src"]), Path(undo["dst"])
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if src.exists():
                        shutil.move(str(src), str(dst))
                elif kind == "remove_dir":       # mkdir that created the directory
                    p = Path(undo["path"])
                    if p.is_dir() and not any(p.iterdir()):
                        p.rmdir()
                undone.append(f"deshecho {row['kind']} {row['args']}")
            except OSError as e:
                undone.append(f"FALLO al deshacer {row['kind']} {row['args']}: {e}")
        self.journal.rename(self.journal.with_suffix(".jsonl.rolled_back"))
        return undone
