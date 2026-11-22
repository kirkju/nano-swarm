"""domains/pc/fitness.py — Fitness of the PC domain: sparse, with safety as a
strong negative score.

Objective signals available without a human:
  - did the plan pass the 4 locks?            (half of the score)
  - did the task produce actions (it did not come up empty)?
  - zero capability denials / zero violations?  -> if there were any, a flat 0.0

Human feedback (thumbs up/down) can be added on top; because the signal is so
sparse, NO aggressive evolution is done in this domain (see docs/architecture.md).
"""

from __future__ import annotations

from core.fitness.base import FitnessFunction

WEIGHTS = {"locks": 0.5, "acted": 0.3, "tiebreak": 0.2}
COST_REF_TOKENS = 2000.0


class PCFitness(FitnessFunction):
    def score(self, task: str, result, adapter) -> float:
        report = getattr(adapter, "last_report", None) or {}

        # Safety violation = 0. No nuance: it is the rule of the domain.
        locks = report.get("locks", {})
        if any(str(v).startswith(("DENEGADO", "INVÁLIDO", "FALLÓ")) for v in locks.values()):
            return 0.0

        score = 0.0
        if report.get("ok") and report.get("actions"):
            score += WEIGHTS["locks"]          # plan validated by the 4 locks
        if report.get("actions"):
            score += WEIGHTS["acted"]          # it proposed something executable
        cost = getattr(result, "cost_tokens", 0)
        score += WEIGHTS["tiebreak"] * (1.0 - min(cost / COST_REF_TOKENS, 1.0))
        return round(min(score, 1.0), 4)
