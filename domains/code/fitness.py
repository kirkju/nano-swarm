"""domains/code/fitness.py — FREE fitness of the code domain.

The project itself tells you whether the agent got it right: tests, lint,
compiles, cost. This is what lets evolution really work: it runs 24/7 without
a human.

Score 0..1 = weighted sum of objective signals:
  tests_pass   0.60   (what matters most: does the suite pass?)
  lint_clean   0.20
  compiles     0.15   (the proposed files are valid Python)
  tie-break    0.05   (fewer tokens and less latency = a touch better)

The weights live here and are explicit on purpose: when Phase 3 arrives,
evolution optimizes EXACTLY this function, so it has to reflect what you really
consider "well done".
"""

from __future__ import annotations

import py_compile
import tempfile
from pathlib import Path

from core.fitness.base import FitnessFunction


WEIGHTS = {"tests": 0.60, "lint": 0.20, "compiles": 0.15, "tiebreak": 0.05}

# Reference scales to normalize the tie-break to [0,1].
COST_REF_TOKENS = 2000.0
LATENCY_REF_S = 10.0


class CodeFitness(FitnessFunction):
    def __init__(self, weights: dict | None = None, require_tests: bool = True):
        self.weights = weights or WEIGHTS
        # If the target project has no tests yet, do not penalize it for that.
        self.require_tests = require_tests

    def score(self, task: str, result, adapter) -> float:
        w = self.weights
        score = 0.0

        # 1) Tests of the target project (main signal, free and automatic).
        #    If the project has no suite of its own (require_tests=False), this
        #    weight is worth 0 and is NOT redistributed: the base ceiling drops to
        #    0.40, so that in read-only tasks (tester/reviewer/navigator) the
        #    discriminative CHECK dominates. Giving away 0.60 let agents that did
        #    nothing through the gate.
        if self.require_tests:
            tests = adapter.run_tests()
            _refuse_if_broken_env(tests)
            if tests.get("ok"):
                score += w["tests"]

        # 2) Linter happy.
        lint = adapter.run_linter()
        _refuse_if_broken_env(lint)
        if lint.get("ok"):
            score += w["lint"]

        # 3) The proposed changes are syntactically valid Python.
        if _all_compile(getattr(result, "proposed_changes", [])):
            score += w["compiles"]

        # 4) Tie-break: cheap and fast is better (it only fine-tunes, never dominates).
        cost_term = 1.0 - min(getattr(result, "cost_tokens", 0) / COST_REF_TOKENS, 1.0)
        lat_term = 1.0 - min(getattr(result, "latency_s", 0.0) / LATENCY_REF_S, 1.0)
        score += w["tiebreak"] * (0.5 * cost_term + 0.5 * lat_term)

        return round(min(score, 1.0), 4)


def _refuse_if_broken_env(result: dict) -> None:
    """If the tool could not even be LAUNCHED (memory pressure, etc.), measuring
    would be lying: better to fail loudly than to score low in silence."""
    if result.get("infra_error"):
        raise RuntimeError(
            f"Entorno de medición roto, me niego a puntuar: {result.get('stderr')}")


def _all_compile(changes) -> bool:
    """True if ALL the proposed changes compile as Python. No changes
    (e.g. the navigator) counts as valid."""
    if not changes:
        return True
    ok = True
    for ch in changes:
        if not str(ch.path).endswith(".py"):
            continue  # we only validate Python syntax
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(ch.content)
            tmp = f.name
        try:
            py_compile.compile(tmp, doraise=True)
        except py_compile.PyCompileError:
            ok = False
        finally:
            Path(tmp).unlink(missing_ok=True)
    return ok
