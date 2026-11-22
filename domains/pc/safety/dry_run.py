"""Lock 3 — Dry-run: propose and validate in a dry run before executing.

`simulate()` is PURE: it does not touch the disk. It validates each action
against the current state (and the state the earlier actions of the same plan
would leave) and returns the plan with the reason for each rejection. If ANY step
is invalid, the whole plan is considered invalid: nothing is executed halfway.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath


# Paths that no plan may touch, whatever capabilities it has.
PROTECTED = (".swarm", ".git")


@dataclass
class PlannedStep:
    action: object            # PCAction
    ok: bool
    reason: str = "ok"
    effect: str = ""          # human-readable description of what would happen


@dataclass
class Plan:
    steps: list[PlannedStep] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.steps) and all(s.ok for s in self.steps)

    def summary(self) -> str:
        return "\n".join(
            f"  [{'ok' if s.ok else 'X'}] {s.action} -> {s.effect or s.reason}"
            for s in self.steps)


class DryRun:
    def simulate(self, actions, root: str | Path) -> Plan:
        root = Path(root).resolve()
        plan = Plan()
        # Simulated state: what would exist after the plan's earlier steps.
        created: set[str] = set()
        removed: set[str] = set()

        def exists(p: Path) -> bool:
            key = str(p)
            if key in created:
                return True
            if key in removed:
                return False
            return p.exists()

        def check_path(raw: str) -> tuple[Path | None, str]:
            q = (root / str(raw)).resolve()
            if not (q == root or root in q.parents):
                return None, f"fuera de la raíz del dominio: {raw}"
            rel = PurePosixPath(str(q.relative_to(root)).replace("\\", "/"))
            if rel.parts and rel.parts[0] in PROTECTED:
                return None, f"ruta protegida ({rel.parts[0]}/): {raw}"
            return q, "ok"

        for action in actions:
            a = action.args
            step = PlannedStep(action=action, ok=True)

            paths = {}
            for key in ("path", "src", "dst"):
                if key in a:
                    q, why = check_path(a[key])
                    if q is None:
                        step.ok, step.reason = False, why
                        break
                    paths[key] = q
            if not step.ok:
                plan.steps.append(step)
                continue

            if action.kind == "write":
                target = paths["path"]
                step.effect = (f"sobrescribiría {a['path']}" if exists(target)
                               else f"crearía {a['path']}")
                created.add(str(target))
            elif action.kind == "mkdir":
                step.effect = f"crearía el directorio {a['path']}"
                created.add(str(paths["path"]))
            elif action.kind in ("move", "copy"):
                src, dst = paths["src"], paths["dst"]
                if not exists(src):
                    step.ok, step.reason = False, f"el origen no existe: {a['src']}"
                elif exists(dst):
                    step.ok, step.reason = False, f"el destino ya existe (no se pisa): {a['dst']}"
                else:
                    step.effect = f"{action.kind} {a['src']} -> {a['dst']}"
                    created.add(str(dst))
                    if action.kind == "move":
                        removed.add(str(src))
            elif action.kind == "delete":
                target = paths["path"]
                if not exists(target):
                    step.ok, step.reason = False, f"no existe: {a['path']}"
                else:
                    step.effect = f"borraría {a['path']} (con respaldo)"
                    removed.add(str(target))
            else:
                step.ok, step.reason = False, f"acción desconocida: {action.kind}"

            plan.steps.append(step)

        if not plan.steps:
            plan.steps.append(PlannedStep(action=None, ok=False,
                                          reason="el agente no propuso ninguna acción"))
        return plan
