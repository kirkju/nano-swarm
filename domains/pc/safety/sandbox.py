"""Lock 1 — Sandbox: act in isolation before touching the real thing.

The real tree is copied to a staging directory and ALL the actions are executed
there, with the SAME executor that live mode would use. Promotion is allowed
only if the rehearsal comes out clean. The sandbox also verifies postconditions
(what was moved is at its destination, what was deleted is gone).
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from domains.pc.actions import apply_action


@dataclass
class SandboxReport:
    ok: bool
    results: list = field(default_factory=list)
    staging: str = ""
    error: str = ""


class Sandbox:
    IGNORE = (".swarm", ".git", "__pycache__")

    def run(self, actions, root: str | Path) -> SandboxReport:
        root = Path(root).resolve()
        staging = Path(tempfile.mkdtemp(prefix="pc_sandbox_"))
        try:
            # Replica of the real world (without the swarm's internal state).
            shutil.copytree(root, staging, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns(*self.IGNORE))
            results = []
            for action in actions:
                try:
                    out = apply_action(action, staging)   # no journal: rehearsal only
                except Exception as e:
                    return SandboxReport(
                        ok=False, results=results, staging=str(staging),
                        error=f"'{action}' reventó en el sandbox: {e}")
                results.append({"action": str(action), **out})
                if not out.get("ok"):
                    return SandboxReport(ok=False, results=results,
                                         staging=str(staging),
                                         error=f"'{action}' falló en el sandbox")

            bad = self._verify(actions, staging)
            if bad:
                return SandboxReport(ok=False, results=results,
                                     staging=str(staging), error=bad)
            return SandboxReport(ok=True, results=results, staging=str(staging))
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def _verify(self, actions, staging: Path) -> str:
        """Postconditions of the expected FINAL STATE of the whole plan.

        It works out what should exist at the end (later actions can undo the
        effects of earlier ones: moving and then deleting is valid) and compares
        that against the actual staging tree.
        """
        expected: dict[str, bool] = {}   # relative path -> must exist at the end
        for action in actions:
            a = action.args
            if action.kind in ("write", "mkdir"):
                expected[a["path"]] = True
            elif action.kind == "move":
                expected[a["src"]] = False
                expected[a["dst"]] = True
            elif action.kind == "copy":
                expected[a["dst"]] = True
            elif action.kind == "delete":
                expected[a["path"]] = False

        for rel, should_exist in expected.items():
            exists = (staging / rel).exists()
            if exists != should_exist:
                return (f"postcondición rota: {rel} "
                        f"{'no existe' if should_exist else 'sigue existiendo'} "
                        f"al final del plan")
        return ""
