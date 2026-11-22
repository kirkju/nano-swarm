"""evolution/selection.py — Selection of the best variants (Phase 3).

Each variant is scored by running ITS agent's tasks in the evaluation suite,
where the score is issued by core/fitness/ (CheckedFitness on top of
CodeFitness). No variant is accepted any other way: this is the rule of the
repo and here it is structural, not optional.

Everything is recorded in an append-only JSONL lineage: for each variant it
stores genome, mutation, parent, generation and score. Reconstructing a
winner's history means reading its chain of parent_id values.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from evolution.variants import Variant, genome_to_overrides


def make_suite_scorer(agent_id: str, base_prompt: str, backend=None,
                      tasks=None) -> Callable[[Variant], dict]:
    """Default scorer: runs the agent's tasks in eval/suite.py.

    The fitness doing the scoring is the one in core/fitness/ (the suite builds
    CheckedFitness(CodeFitness, check) per task). Returns the full report.
    """
    from eval.suite import TASKS, run_suite

    tasks = tasks or [t for t in TASKS if t.expect_agent == agent_id]

    def scorer(variant: Variant) -> dict:
        overrides = {agent_id: genome_to_overrides(variant.genome, base_prompt)}
        return run_suite(backend=backend, tasks=tasks,
                         agent_overrides=overrides, verbose=False)

    return scorer


def select(variants: list[Variant], scorer: Callable[[Variant], dict],
           keep: int = 1, lineage_path: str | Path | None = None,
           deadline: float | None = None) -> list[Variant]:
    """Scores each variant with the scorer (which MUST rely on core/fitness/),
    records the lineage and returns the `keep` best ones.

    deadline: time.time() limit; variants that cannot be evaluated in time are
    recorded as skipped (score None) — the time cap takes precedence.
    """
    for v in variants:
        if v.score is not None:      # already evaluated (e.g. the reigning champion)
            continue
        if deadline and time.time() > deadline:
            _log(lineage_path, v, skipped=True)
            continue
        report = scorer(v)
        v.score = report["mean"]
        v.detail = {r["name"]: r["score"] for r in report["results"]}
        _log(lineage_path, v)

    scored = [v for v in variants if v.score is not None]
    # Reproducible tie-break: on equal score, the lower generation wins
    # (simpler), then by id.
    scored.sort(key=lambda v: (-v.score, v.generation, v.variant_id))
    return scored[:keep]


def _log(lineage_path, v: Variant, skipped: bool = False) -> None:
    if lineage_path is None:
        return
    path = Path(lineage_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": time.time(),
        "variant_id": v.variant_id,
        "agent_id": v.agent_id,
        "generation": v.generation,
        "parent_id": v.parent_id,
        "mutation": v.mutation,
        "genome": v.genome,
        "score": v.score,
        "detail": v.detail,
    }
    if skipped:
        row["skipped"] = "tope de tiempo"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
