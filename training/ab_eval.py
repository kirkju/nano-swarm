"""training/ab_eval.py — Acceptance A/B of a LoRA (Phase 2).

Rule of the repo: a LoRA is ACCEPTED only if it measures BETTER than the
context-only version against core/fitness/ on the evaluation suite. This script
runs both arms and issues the verdict.

Arms (same agent tasks, same fitness, greedy decoding):
  A) context:   the agent served by the SAME HF base, without the adapter
  B) lora:      the same HF base + the trained adapter

Comparing HF against HF isolates the effect of the LoRA (comparing against Ollama
would mix in quantization and engine). The Phase 1 baseline is also reported as
a reference.

Usage:
    python -m training.ab_eval --agent reviewer
    python -m training.ab_eval --agent editor --base Qwen/Qwen2.5-Coder-1.5B-Instruct

Output: eval/results/ab_<agent>.json  (+ verdict on stdout and exit code 0/1)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.suite import TASKS, run_suite

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = "Qwen/Qwen2.5-1.5B-Instruct"


def arm_overrides(agent: str, base: str, adapter: str | None) -> dict:
    """Overrides that serve `agent` with the HF base (and optional adapter).

    Instead of touching config/models.yaml on disk, we inject a dynamic model
    key: the DispatchBackend resolves through the models dict it receives, so the
    key is registered on the fly in the backend of the swarm being built.
    """
    key = f"_ab_{agent}_{'lora' if adapter else 'ctx'}"
    spec = {"backend": "hf", "base": base}
    if adapter:
        spec["adapter"] = adapter
    return {agent: {"model": key}}, key, spec


def make_backend(agent: str, base: str, adapter: str | None):
    """SHARED backend for both arms of the A/B.

    One backend per arm leaves the previous arm's model resident (nobody evicts
    it) and on this machine that breaks subprocess spawning for the fitness
    (WinError 1455): the second arm scores ~0.19 on the base fitness and the A/B
    ends up biased AGAINST the second arm. With a single DispatchBackend,
    HFLocalBackend evicts the previous model before loading the next one.
    """
    import yaml
    from core.backends import DispatchBackend

    models_cfg = yaml.safe_load((ROOT / "config" / "models.yaml").read_text(encoding="utf-8"))
    models = models_cfg["models"]
    for arm_adapter in (None, adapter):
        _, key, spec = arm_overrides(agent, base, arm_adapter)
        models[key] = spec
    return DispatchBackend(models, host=models_cfg.get("host", "http://localhost:11434"),
                           root=str(ROOT))


def run_arm(backend, agent: str, tasks, base: str, adapter: str | None) -> dict:
    overrides, _, _ = arm_overrides(agent, base, adapter)
    return run_suite(backend=backend, tasks=tasks, agent_overrides=overrides,
                     config_dir=str(ROOT / "config"))


def main():
    ap = argparse.ArgumentParser(description="A/B de aceptación de un LoRA")
    ap.add_argument("--agent", required=True,
                    choices=["navigator", "editor", "tester", "reviewer"])
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--adapter", default=None,
                    help="ruta del adaptador (default models/lora/<agent>)")
    args = ap.parse_args()

    adapter = args.adapter or f"models/lora/{args.agent}"
    tasks = [t for t in TASKS if t.expect_agent == args.agent]
    print(f"== A/B {args.agent}: {len(tasks)} tareas | base {args.base} | adaptador {adapter}\n")

    backend = make_backend(args.agent, args.base, adapter)

    print("-- brazo A: solo contexto (base HF sin adaptador)")
    ctx = run_arm(backend, args.agent, tasks, args.base, None)
    print(f"   media contexto: {ctx['mean']:.4f}\n")

    print("-- brazo B: base HF + LoRA")
    lora = run_arm(backend, args.agent, tasks, args.base, adapter)
    print(f"   media LoRA:     {lora['mean']:.4f}\n")

    baseline_path = ROOT / "eval" / "results" / "phase1_baseline.json"
    phase1 = None
    if baseline_path.exists():
        phase1 = json.loads(baseline_path.read_text(encoding="utf-8"))["per_agent"].get(args.agent)

    accepted = lora["mean"] > ctx["mean"]
    report = {
        "agent": args.agent, "base": args.base, "adapter": adapter,
        "context_mean": ctx["mean"], "lora_mean": lora["mean"],
        "phase1_ollama_mean": phase1,
        "accepted": accepted,
        "context": ctx, "lora": lora,
    }
    out = ROOT / "eval" / "results" / f"ab_{args.agent}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Fase 1 (Ollama, referencia): {phase1}")
    print(f"VEREDICTO: {'ACEPTADO' if accepted else 'RECHAZADO'} "
          f"(LoRA {lora['mean']:.4f} vs contexto {ctx['mean']:.4f})")
    print(f"Detalle: {out}")
    sys.exit(0 if accepted else 1)


if __name__ == "__main__":
    main()
