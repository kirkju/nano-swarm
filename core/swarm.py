"""core/swarm.py — The swarm coordinator.

Ties all the pieces together: loads agents from config, passes them through the
router, runs the domain tool, measures with fitness and stores in memory.

This is the main loop. It is domain-agnostic: the same Swarm works for code and
for PC; what changes is the `adapter` injected into it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import yaml

from core.agent import NanoAgent
from core.fitness.base import FitnessFunction
from core.memory.project_index import ProjectIndex
from core.memory.store import MemoryStore
from core.router import Router


@dataclass
class SwarmOutcome:
    """Result of handling a request: what it did and how it scored."""
    task: str
    route_reason: str
    output: str
    score: float
    touched_files: list
    confidence: float
    cost_tokens: int


class Swarm:
    def __init__(self, router: Router, memory: MemoryStore,
                 index: ProjectIndex, fitness: FitnessFunction, adapter):
        self.router = router
        self.memory = memory
        self.index = index
        self.fitness = fitness
        self.adapter = adapter   # domain adapter (code or pc)

    def handle(self, task: str) -> SwarmOutcome:
        """Main loop for one request (the 8 steps of the design)."""
        # 1. Context = episodic memory + project structure (RAG).
        context = "\n\n".join(p for p in (self.memory.recall(task), self.index.relevant(task)) if p)

        # 2. The router picks the agent (Phase 1 rules).
        route = self.router.route(task)

        # 3. The agent acts (reads with tools, proposes changes).
        result = route.agent.act(task, context)

        # 4. Explicit escalation if confidence is low.
        result = self.router.maybe_escalate(result, task, context)

        # 5. The adapter applies the proposed changes (the only point that mutates).
        touched = self.adapter.execute(result)

        # 6. Fitness measures the result (free in code: tests/lint/compiles).
        score = self.fitness.score(task, result, self.adapter)

        # 7. Memory remembers it (this is how the swarm improves with use).
        self.memory.remember(task, result, score)

        # 8. Return what happened.
        return SwarmOutcome(
            task=task,
            route_reason=route.reason,
            output=result.output,
            score=score,
            touched_files=touched,
            confidence=result.confidence,
            cost_tokens=result.cost_tokens,
        )


# --------------------------------------------------------------------------
# Building the swarm from config (config/swarm.yaml + models.yaml)
# --------------------------------------------------------------------------

def _system_prompt(agent_id: str, purpose: str, tool_names: list[str],
                   instructions: str = "") -> str:
    """Short, directive prompt (tiny models perform better this way).
    `instructions` is the agent's working protocol (config/swarm.yaml)."""
    base = (
        f"Eres '{agent_id}', un nano-agente especializado dentro de un enjambre.\n"
        f"Tu función: {purpose}\n"
        f"Herramientas que puedes usar: {', '.join(tool_names) or 'ninguna'}.\n"
        "Usa SOLO esas herramientas. Sé conciso y específico. Si necesitas leer algo,"
        ' pídelo con: TOOL: <nombre> {"arg": "valor"}.\n'
        "Si propones cambios de código, devuélvelos como bloques ```file path=...```.\n"
        "No te salgas de tu función: si la tarea es de otro agente, dilo brevemente."
    )
    if instructions:
        base += f"\n\nProtocolo de trabajo (síguelo al pie de la letra):\n{instructions.strip()}"
    return base


def build_swarm(project_root: str, config_dir: str = "config", backend=None,
                fitness=None, memory_path: str = ".swarm/memory.db",
                agent_overrides: dict | None = None,
                with_fallback: bool = True) -> Swarm:
    """Wires a complete CODE-domain Swarm from the config YAML files.

    backend: if None, uses OllamaBackend (production). Tests inject a
    FakeBackend to run the loop without models.
    fitness: if None, CodeFitness(). The eval injects a CheckedFitness per task.
    memory_path: sqlite file for episodic memory (":memory:" for isolated evals).
    agent_overrides: {agent_id: {system_prompt|model|gen_options|lora_adapter}} —
        this way the Phase 2/3 variants are tested without touching the config on disk.
    with_fallback: if False, the escape-hatch agent (7B) is NOT wired. The A/B and
        evolution disable it to measure the agent IN ISOLATION: if an arm escalates
        to the 7B, its score would come from another model and the comparison
        would be biased.
    """
    from domains.code.tools import build_tools
    from domains.code.fitness import CodeFitness
    from domains.code.adapter import CodeAdapter

    cfg_dir = Path(config_dir)
    swarm_cfg = yaml.safe_load((cfg_dir / "swarm.yaml").read_text(encoding="utf-8"))
    models_cfg = yaml.safe_load((cfg_dir / "models.yaml").read_text(encoding="utf-8"))
    models = models_cfg["models"]

    if backend is None:  # pragma: no cover - requires Ollama running
        from core.backends import DispatchBackend
        backend = DispatchBackend(models, host=models_cfg.get("host", "http://localhost:11434"))

    catalog = build_tools(project_root)
    overrides = agent_overrides or {}

    # One NanoAgent per config entry; specialization through prompt + tools.
    agents: dict[str, NanoAgent] = {}
    for spec in swarm_cfg["agents"]:
        tool_names = spec.get("tools", [])
        # The prompt ONLY advertises tools that are actually wired: a tool in the
        # config that does not exist in the catalog (e.g. a typo) would make the
        # serving prompt promise something the runtime always denies — and it would
        # diverge from the training prompt. It is filtered here, with a warning.
        wired = [n for n in tool_names if n in catalog]
        missing = [n for n in tool_names if n not in catalog]
        if missing:
            import warnings
            warnings.warn(
                f"agente '{spec['id']}': herramientas en config sin catálogo, "
                f"ignoradas: {missing}", RuntimeWarning)
        agent_tools = {n: catalog[n] for n in wired}
        ov = overrides.get(spec["id"], {})
        agents[spec["id"]] = NanoAgent(
            agent_id=spec["id"],
            model=ov.get("model", spec["model"]),   # key in models.yaml (small/mid/...)
            system_prompt=ov.get(
                "system_prompt",
                _system_prompt(spec["id"], spec.get("purpose", ""), wired,
                               spec.get("instructions", "")),
            ),
            tools=agent_tools,
            backend=backend,
            lora_adapter=ov.get("lora_adapter"),
            gen_options=ov.get("gen_options"),
        )

    # Escape-hatch agent (fallback): more capable, with read + write access.
    fallback = None
    if with_fallback and "fallback" in models:
        fallback = NanoAgent(
            agent_id="fallback",
            model="fallback",
            system_prompt=_system_prompt(
                "fallback",
                "Resolver tareas difíciles cuando un nano-agente no tuvo confianza.",
                list(catalog),
            ),
            tools=catalog,
            backend=backend,
        )

    escalate_below = (
        swarm_cfg.get("router", {})
        .get("escalate_when", {})
        .get("confidence_below", 0.6)
    )
    router = Router(agents, escalate_below=escalate_below, fallback=fallback)

    index = ProjectIndex(project_root)
    index.build()

    return Swarm(
        router=router,
        memory=MemoryStore(memory_path),
        index=index,
        fitness=fitness or CodeFitness(),
        adapter=CodeAdapter(project_root, tools=catalog),
    )


def main():  # pragma: no cover - entry point of `make run`
    parser = argparse.ArgumentParser(description="nano-swarm — enjambre de nano-agentes")
    parser.add_argument("--domain", default="code", choices=["code"],
                        help="dominio activo (Fase 1: solo 'code')")
    parser.add_argument("--project", default=".", help="raíz del proyecto objetivo")
    parser.add_argument("--task", default=None, help="tarea a resolver (si se omite, modo interactivo)")
    args = parser.parse_args()

    if args.domain != "code":
        raise SystemExit("Fase 1 solo soporta el dominio 'code' (ver docs/architecture.md).")

    swarm = build_swarm(args.project)

    def run_one(task: str):
        out = swarm.handle(task)
        print(f"\n[router] {out.route_reason}")
        print(f"[fitness] {out.score:.3f}   [confianza] {out.confidence:.2f}   [tokens] {out.cost_tokens}")
        if out.touched_files:
            print(f"[archivos tocados] {', '.join(out.touched_files)}")
        print(f"\n{out.output}\n")

    if args.task:
        run_one(args.task)
    else:
        print("nano-swarm listo (dominio código). Escribe una tarea, o Ctrl-D para salir.")
        try:
            while True:
                task = input("swarm> ").strip()
                if task:
                    run_one(task)
        except (EOFError, KeyboardInterrupt):
            print("\nadiós 🐝")


if __name__ == "__main__":
    main()
