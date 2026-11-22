"""domains/pc/adapter.py — The swarm's body in the PC domain (Phase 4).

SAME core as code; what changes is that here EVERY mutation goes through the
4 locks, IN THIS ORDER and without exceptions:

  1. kill switch   — did someone stop the swarm? -> abort
  2. capabilities  — is this agent allowed to do EACH action? -> default-deny
  3. dry-run       — is the plan valid in a dry run? -> if one step fails, nothing runs
  4. sandbox       — real rehearsal on a replica; promoted only if it comes out clean

And even with all 4 green: by default the adapter is in DRY MODE (live=False)
and does NOT touch the real world — it returns the validated plan. Promoting to
real requires an explicit `live=True` at construction. Every real action is
recorded in the kill switch journal and is reversible with rollback().
"""

from __future__ import annotations

from pathlib import Path

from domains.pc.actions import apply_action, extract_actions
from domains.pc.safety.capabilities import CapabilityGuard
from domains.pc.safety.dry_run import DryRun
from domains.pc.safety.kill_switch import KillSwitch
from domains.pc.safety.sandbox import Sandbox


class PCAdapter:
    def __init__(self, root: str | Path, caps: CapabilityGuard,
                 live: bool = False, kill: KillSwitch | None = None):
        self.root = Path(root).resolve()
        self.caps = caps
        self.live = live                      # False = ALWAYS dry-run (default)
        self.kill = kill or KillSwitch(self.root / ".swarm")
        self.dry_run = DryRun()
        self.sandbox = Sandbox()
        self.applied: list[str] = []
        self.last_report: dict | None = None   # read by PCFitness

    def execute(self, result) -> dict:
        self.last_report = self._execute(result)
        return self.last_report

    def _execute(self, result) -> dict:
        """Returns a report with the verdict of each lock. It only mutates the
        real world if live=True AND all 4 locks are green."""
        report = {"mode": "live" if self.live else "dry-run", "applied": [],
                  "locks": {}}

        # Lock 4 (kill) is checked FIRST: if it is active, do not even plan.
        self.kill.check()
        report["locks"]["kill_switch"] = "ok (sin parada activa)"

        actions = extract_actions(result)
        report["actions"] = [str(a) for a in actions]
        if not actions:
            report["locks"]["capabilities"] = report["locks"]["dry_run"] = \
                report["locks"]["sandbox"] = "n/a (sin acciones propuestas)"
            report["ok"] = True
            return report

        # Lock 2: capabilities (default-deny; ONE denial aborts everything).
        agent_id = getattr(result, "agent_id", "")
        for action in actions:
            allowed, why = self.caps.authorize(agent_id, action)
            if not allowed:
                report["locks"]["capabilities"] = f"DENEGADO: {why}"
                report["ok"] = False
                return report
        report["locks"]["capabilities"] = f"ok ({len(actions)} acciones autorizadas)"

        # Lock 3: dry-run (pure, without touching the disk).
        plan = self.dry_run.simulate(actions, self.root)
        report["plan"] = plan.summary()
        if not plan.ok:
            report["locks"]["dry_run"] = "INVÁLIDO:\n" + plan.summary()
            report["ok"] = False
            return report
        report["locks"]["dry_run"] = "ok (plan válido)"

        # Lock 1: sandbox (rehearsal on a replica with the SAME executor).
        sb = self.sandbox.run(actions, self.root)
        if not sb.ok:
            report["locks"]["sandbox"] = f"FALLÓ: {sb.error}"
            report["ok"] = False
            return report
        report["locks"]["sandbox"] = "ok (ensayo limpio en réplica)"

        if not self.live:
            report["ok"] = True
            report["note"] = ("modo seco: plan validado por los 4 candados, "
                              "NADA se aplicó al mundo real")
            return report

        # Promotion to real: re-check the stop BEFORE EACH action (not only
        # before the batch) and record each one in the journal. This way the kill
        # switch stops a long plan midway, not only between plans; what was
        # already applied stays in the journal and is reversible with rollback().
        for action in actions:
            self.kill.check()
            apply_action(action, self.root, kill=self.kill)
            self.applied.append(str(action))
            report["applied"].append(str(action))
        report["ok"] = True
        return report

    def rollback(self) -> list[str]:
        """Undoes all the real actions of this session (lock 4)."""
        return self.kill.rollback()
