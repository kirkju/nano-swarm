"""eval/run.py — Quality gate of the swarm.

Runs the suite in eval/suite.py (18 tasks across the four agents) and compares the mean
fitness against the gate. This is what decides whether a phase is "ready".

Usage:
    python -m eval.run                    # against Ollama (production)
    python -m eval.run --fake             # offline: validates the wiring, not the intelligence
    python -m eval.run --json out.json    # also dump the results to JSON

Exits with a code != 0 if the mean fitness falls below the gate.
"""

from __future__ import annotations

import argparse
import json
import sys

from eval.suite import TASKS, make_fake_backend, run_suite

GATE = 0.6   # minimum mean fitness for the phase to count as "green"


def main():
    parser = argparse.ArgumentParser(description="Evaluación del enjambre")
    parser.add_argument("--fake", action="store_true",
                        help="usar backend simulado (offline/CI; valida cableado)")
    parser.add_argument("--json", default=None, metavar="PATH",
                        help="guardar resultados detallados en un JSON")
    parser.add_argument("--only", default=None,
                        help="filtrar tareas por subcadena del nombre (p.ej. 'editor/')")
    args = parser.parse_args()

    tasks = [t for t in TASKS if args.only in t.name] if args.only else TASKS
    if not tasks:
        raise SystemExit(f"Ningún task matchea --only {args.only!r}")

    if args.fake:
        backend = make_fake_backend(tasks)
    else:
        # ONE backend shared by the whole suite: otherwise each task reloads the
        # HF models from scratch (slow) and multiplies the memory pressure.
        import yaml
        from core.backends import DispatchBackend
        models_cfg = yaml.safe_load(open("config/models.yaml", encoding="utf-8"))
        backend = DispatchBackend(models_cfg["models"],
                                  host=models_cfg.get("host", "http://localhost:11434"))
    print(f"Corriendo suite ({len(tasks)} tareas) con backend "
          f"{'FAKE' if args.fake else 'real (ollama+hf)'}...\n")
    report = run_suite(backend=backend, tasks=tasks)

    print(f"\nAptitud media: {report['mean']:.3f}   (puerta: {GATE})")
    print("Por agente: " + "  ".join(f"{a}={s:.3f}" for a, s in sorted(report["per_agent"].items())))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Resultados guardados en {args.json}")

    # ASCII markers on purpose: the Windows console (cp1252) chokes on ✓/✗ and
    # we do not want the quality gate to die because of an emoji.
    if report["mean"] < GATE:
        print("[X] Por debajo de la puerta: la fase NO esta lista.")
        sys.exit(1)
    print("[OK] Fase verde segun la suite actual.")


if __name__ == "__main__":
    main()
