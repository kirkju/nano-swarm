"""evolution/run.py — Evolution driver (Phase 3), with hard CAPS.

Loop per agent:
  gen 0: the champion is the current config (empty genome = production as is),
         scored against the suite.
  gen k: mutate the champion -> population, score EVERYTHING with core/fitness/
         (through eval/suite.py), keep the best. If nobody beats the champion,
         it counts as a dry generation; two dry ones in a row = stop (converged).

Caps: --generations (default 3) and --max-minutes per agent (default 25).
Lineage: evolution/lineage/<agent>.jsonl (append-only, every evaluated variant).
Winners: evolution/winners/<agent>.json — ONLY if they beat the current config.

Usage:
    python -m evolution.run --agent reviewer
    python -m evolution.run --all
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from pathlib import Path

import yaml

from core.swarm import _system_prompt
from evolution.selection import make_suite_scorer, select
from evolution.variants import Variant, generate_variants, _genome_key

ROOT = Path(__file__).resolve().parent.parent

# Which model keys each agent may try when mutating. It deliberately does NOT
# include `fallback` (7B: expensive; escalating is already the router's job,
# not the genome's).
MODEL_POOLS = {
    "navigator": ["small", "tiny"],
    "editor": ["mid", "small"],
    "tester": ["small", "tiny"],
    "reviewer": ["small", "tiny"],
}


def _base_prompt(agent_id: str) -> str:
    cfg = yaml.safe_load((ROOT / "config" / "swarm.yaml").read_text(encoding="utf-8"))
    spec = next(a for a in cfg["agents"] if a["id"] == agent_id)
    return _system_prompt(agent_id, spec.get("purpose", ""), spec.get("tools", []),
                          spec.get("instructions", ""))


def _model_pool(agent_id: str) -> list[str]:
    """Static pool + any LoRA of this agent accepted in Phase 2 (hf keys
    declared in config/models.yaml)."""
    models = yaml.safe_load((ROOT / "config" / "models.yaml").read_text(encoding="utf-8"))["models"]
    pool = [m for m in MODEL_POOLS.get(agent_id, []) if m in models]
    pool += [k for k, spec in models.items()
             if isinstance(spec, dict) and spec.get("backend") == "hf"
             and agent_id in k]
    return pool


def _effective(agent_id: str) -> dict:
    """The agent's EFFECTIVE values today: the model from swarm.yaml and the
    backend defaults. They are passed to the mutator so that it does not
    generate no-op variants."""
    cfg = yaml.safe_load((ROOT / "config" / "swarm.yaml").read_text(encoding="utf-8"))
    spec = next(a for a in cfg["agents"] if a["id"] == agent_id)
    return {"model": spec["model"], "temperature": 0.2, "num_predict": 1024}


def evolve_agent(agent_id: str, generations: int, population: int,
                 max_minutes: float, seed: int, backend=None) -> Variant:
    rng = random.Random(seed)
    deadline = time.time() + max_minutes * 60
    base_prompt = _base_prompt(agent_id)
    effective = _effective(agent_id)
    lineage = ROOT / "evolution" / "lineage" / f"{agent_id}.jsonl"
    scorer = make_suite_scorer(agent_id, base_prompt, backend=backend)
    # State that persists ACROSS generations: genomes already evaluated (do not
    # re-score losers) and globally unique ids (lineage without collisions).
    seen: set[str] = set()
    id_counter = itertools.count()

    champion = Variant(variant_id=f"{agent_id}-g0-actual", agent_id=agent_id,
                       generation=0, parent_id=None, genome={}, mutation="config actual")
    seen.add(_genome_key(champion.genome))
    print(f"== evolución {agent_id}: puntuando al campeón actual...")
    select([champion], scorer, keep=1, lineage_path=lineage, deadline=deadline)
    if champion.score is None:
        raise SystemExit("Tope de tiempo antes de puntuar la config actual; nada que hacer.")
    print(f"   campeón g0: {champion.score:.4f}")

    dry = 0
    for gen in range(1, generations + 1):
        if time.time() > deadline:
            print(f"   [tope de tiempo] paro en la generación {gen}")
            break
        if dry >= 2:
            print("   dos generaciones sin mejora: converge, paro.")
            break

        children = generate_variants(champion, n=population, rng=rng,
                                     model_pool=_model_pool(agent_id),
                                     effective=effective, seen=seen,
                                     id_counter=id_counter)
        if not children:
            print(f"   g{gen}: no quedan genomas nuevos que probar; paro.")
            break
        best = select(children + [champion], scorer, keep=1,
                      lineage_path=lineage, deadline=deadline)[0]
        for c in children:
            mark = " <- MEJOR" if c is best else ""
            print(f"   g{gen} {c.variant_id}: {c.score} ({c.mutation}){mark}")
        if best is champion:
            dry += 1
            print(f"   g{gen}: nadie supera al campeón ({champion.score:.4f}); seca {dry}/2")
        else:
            dry = 0
            champion = best
            print(f"   g{gen}: nuevo campeón {champion.variant_id} = {champion.score:.4f}")

    return champion


def main():
    ap = argparse.ArgumentParser(description="Evolución de variantes (con topes)")
    ap.add_argument("--agent", choices=list(MODEL_POOLS))
    ap.add_argument("--all", action="store_true", help="evolucionar los 4 agentes")
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--population", type=int, default=4)
    ap.add_argument("--max-minutes", type=float, default=25.0,
                    help="tope de tiempo POR AGENTE")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    agents = list(MODEL_POOLS) if args.all else [args.agent]
    if not agents or agents == [None]:
        ap.error("usa --agent <id> o --all")

    winners_dir = ROOT / "evolution" / "winners"
    winners_dir.mkdir(parents=True, exist_ok=True)

    # ONE backend shared by the whole evolution (avoids reloading HF models per
    # task and the memory pressure that breaks subprocesses on Windows).
    from core.backends import DispatchBackend
    models_cfg = yaml.safe_load((ROOT / "config" / "models.yaml").read_text(encoding="utf-8"))
    backend = DispatchBackend(models_cfg["models"],
                              host=models_cfg.get("host", "http://localhost:11434"),
                              root=str(ROOT))

    for agent_id in agents:
        champ = evolve_agent(agent_id, args.generations, args.population,
                             args.max_minutes, args.seed, backend=backend)
        if champ.generation == 0:
            print(f"== {agent_id}: la config actual sigue siendo la mejor "
                  f"({champ.score:.4f}); no se escribe ganador.\n")
            continue
        out = winners_dir / f"{agent_id}.json"
        out.write_text(json.dumps({
            "variant_id": champ.variant_id,
            "agent_id": agent_id,
            "score": champ.score,
            "genome": champ.genome,
            "mutation": champ.mutation,
            "parent_id": champ.parent_id,
            "generation": champ.generation,
            "detail": champ.detail,
            "lineage_file": f"evolution/lineage/{agent_id}.jsonl",
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"== {agent_id}: GANADORA {champ.variant_id} ({champ.score:.4f}) -> {out}\n")


if __name__ == "__main__":
    main()
