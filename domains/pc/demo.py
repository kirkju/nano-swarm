"""domains/pc/demo.py — LOW-RISK task of the PC domain, DRY by default.

A real agent (tiny model through Ollama) is given a messy directory and
proposes how to organize it. Its proposals go through the 4 locks; in dry mode
(default) NOTHING is applied: the validated plan and the fitness are printed.

Usage:
    python -m domains.pc.demo                 # dry-run (default, recommended)
    python -m domains.pc.demo --live          # applies FOR REAL (with journal/rollback)
    python -m domains.pc.demo --live --rollback   # applies, then undoes (full demo)

The target directory is ALWAYS a scratch folder created by the demo itself
(.swarm/pc_demo/), never your real folders.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

from core.agent import NanoAgent
from core.backends import DispatchBackend
from domains.pc.actions import MUTATING_PC_TOOLS
from domains.pc.adapter import PCAdapter
from domains.pc.fitness import PCFitness
from domains.pc.safety.capabilities import Capability, CapabilityGuard
from domains.pc.tools import build_pc_tools

ROOT = Path(__file__).resolve().parent.parent.parent

SYSTEM = (
    "Eres 'organizador', un nano-agente que ordena archivos dentro de UNA carpeta.\n"
    "Protocolo (síguelo al pie de la letra):\n"
    "1. Tu PRIMERA respuesta es SOLO: TOOL: list_dir {}\n"
    "2. Cuando veas el listado, propone las mutaciones con líneas TOOL:, una por acción:\n"
    '   TOOL: mkdir {"path": "texto"}\n'
    '   TOOL: move_file {"src": "notas.txt", "dst": "texto/notas.txt"}\n'
    "   Reglas: archivos .txt y .md van a texto/, .png y .jpg a imagenes/, "
    ".csv a datos/. No toques nada más.\n"
    "3. Termina con una línea: CONFIDENCE: <0..1>\n"
    "Las mutaciones NO se ejecutan: son propuestas que validan los candados."
)

MESSY = {
    "notas.txt": "apuntes\n", "resumen.md": "# resumen\n",
    "grafico.png": "PNG-fake\n", "vacaciones.jpg": "JPG-fake\n",
    "ventas.csv": "a,b\n1,2\n",
}


def main():
    ap = argparse.ArgumentParser(description="Demo PC de bajo riesgo (4 candados)")
    ap.add_argument("--live", action="store_true",
                    help="aplicar de verdad (default: seco). Sigue pasando por los 4 candados.")
    ap.add_argument("--rollback", action="store_true",
                    help="tras aplicar en live, deshacer todo (demuestra el candado 4)")
    args = ap.parse_args()

    # The demo's own scratch folder: never the user's real folders.
    demo_root = ROOT / ".swarm" / "pc_demo"
    if demo_root.exists():
        shutil.rmtree(demo_root)
    demo_root.mkdir(parents=True)
    for name, content in MESSY.items():
        (demo_root / name).write_text(content, encoding="utf-8")

    models_cfg = yaml.safe_load((ROOT / "config" / "models.yaml").read_text(encoding="utf-8"))
    backend = DispatchBackend(models_cfg["models"],
                              host=models_cfg.get("host", "http://localhost:11434"),
                              root=str(ROOT))

    agent = NanoAgent(
        agent_id="organizador",
        model="small",
        system_prompt=SYSTEM,
        tools=build_pc_tools(str(demo_root)),
        backend=backend,
        mutating_tools=MUTATING_PC_TOOLS,
        file_block_changes=False,     # in PC, writes go through TOOL: write_file
    )

    # Lock 2: ONLY moving and creating folders, ONLY inside the scratch folder.
    caps = CapabilityGuard({"organizador": [
        Capability.make({"move", "mkdir"}, ("**",)),
    ]})
    adapter = PCAdapter(demo_root, caps, live=args.live)

    task = "organiza los archivos de esta carpeta según las reglas"
    print(f"[demo] raíz: {demo_root}  |  modo: {'LIVE' if args.live else 'SECO'}")
    result = agent.act(task)
    print(f"[agente] confianza {result.confidence:.2f}, {result.cost_tokens} tokens")

    report = adapter.execute(result)
    print("\n[candados]")
    for lock, verdict in report["locks"].items():
        print(f"  {lock}: {verdict}")
    if report.get("plan"):
        print("\n[plan]")
        print(report["plan"])
    if report["applied"]:
        print("\n[aplicado de verdad]")
        for a in report["applied"]:
            print(f"  {a}")

    score = PCFitness().score(task, result, adapter)
    print(f"\n[fitness PC] {score}")

    if args.live and args.rollback:
        print("\n[rollback] deshaciendo todo...")
        for line in adapter.rollback():
            print(f"  {line}")
        restantes = sorted(p.name for p in demo_root.iterdir() if p.is_file())
        print(f"[rollback] archivos de vuelta en la raíz: {restantes}")

    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
