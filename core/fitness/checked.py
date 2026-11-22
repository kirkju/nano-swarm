"""core/fitness/checked.py — Composite fitness: domain + objective per-task check.

The domain fitness (e.g. CodeFitness) measures overall health: tests, lint,
compiles, cost. But to EVALUATE and SELECT (Phases 2-3) we also need to know
whether the agent solved THIS specific task: did it name the right file? did it
use the tool it was supposed to? does the expected file exist?

CheckedFitness composes both signals and is still a FitnessFunction: everything
accepted in Phase 2-3 goes through core/fitness/, as docs/architecture.md requires.

The `check` is a callable with the SAME signature as score(task, result, adapter)
that returns 0..1. It must be objective (file existence, recorded tool_calls,
unambiguous substrings), never a subjective judgment.
"""

from __future__ import annotations

from typing import Callable

from core.fitness.base import FitnessFunction


class CheckedFitness(FitnessFunction):
    def __init__(
        self,
        base: FitnessFunction,
        check: Callable[[str, object, object], float] | None = None,
        check_weight: float = 0.5,
    ):
        if not 0.0 <= check_weight <= 1.0:
            raise ValueError(f"check_weight debe estar en [0,1], no {check_weight}")
        self.base = base
        self.check = check
        self.check_weight = check_weight

    def score(self, task: str, result, adapter) -> float:
        base_score = self.base.score(task, result, adapter)
        if self.check is None:
            return base_score
        try:
            check_score = max(0.0, min(1.0, float(self.check(task, result, adapter))))
        except Exception:
            # A check that blows up counts as a task NOT solved, not as a swarm
            # crash: the signal must be robust enough to run without a human.
            check_score = 0.0
        w = self.check_weight
        return round((1.0 - w) * base_score + w * check_score, 4)
