"""evolution/variants.py — Generation of agent variants (Phase 3).

A variant is a GENOME of an agent's serving configuration:
    {"directives": [str], "gen_options": {...}|None, "model": str|None}

  - directives:   protocol lines that are APPENDED to the base system prompt
                  (cheap mutation; the first thing docs/architecture.md asks for).
  - gen_options:  decoding (temperature, num_predict; seed stays fixed).
  - model:        key in config/models.yaml (lets Ollama compete against
                  HF base + LoRA accepted in Phase 2).

⚠️ This module only GENERATES variants. Nothing is accepted without going through
evolution/selection.py, which scores with core/fitness/ (rule of the repo).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class Variant:
    variant_id: str
    agent_id: str
    generation: int
    parent_id: str | None
    genome: dict
    mutation: str = "origen"        # what changed vs. the parent (for the lineage)
    score: float | None = None
    detail: dict = field(default_factory=dict)


# Per-agent directive libraries: prompt mutations that make sense, not random
# character noise. Evolution picks which subset helps.
DIRECTIVES: dict[str, list[str]] = {
    "navigator": [
        "Responde en UNA sola frase.",
        "Nombra SIEMPRE la ruta con extensión (p. ej. modulo.py).",
        "Si un grep no encuentra nada, prueba list_dir antes de rendirte.",
        "No expliques tu proceso: da la respuesta y ya.",
    ],
    "editor": [
        "No escribas NADA fuera del bloque ```file path=...```.",
        "Incluye un docstring de una línea en cada función.",
        "Prefiere la solución más corta que pase las pruebas.",
        "No toques archivos que empiecen por test_.",
    ],
    "tester": [
        "Primera línea: veredicto. Segunda línea: detalle.",
        "Cita textualmente la línea más informativa de la salida de pytest.",
        "Si todo pasa, di cuántos tests pasaron.",
        "Nunca sugieras arreglos: tu trabajo es diagnosticar, no editar.",
    ],
    "reviewer": [
        "Empieza tu veredicto con la palabra ACEPTA o RECHAZA.",
        "Máximo tres frases.",
        "Enumera el riesgo más grave primero.",
        "Ante la duda, RECHAZA y explica qué falta para aceptar.",
    ],
}

TEMPERATURES = [0.0, 0.2, 0.4]
NUM_PREDICT = [512, 1024]


def mutate(parent: Variant, rng: random.Random, model_pool: list[str],
           child_id: str, effective: dict | None = None) -> Variant:
    """Applies ONE mutation to the parent's genome (small, attributable changes:
    if you mutate three things at once you never know which one helped).

    `effective` holds the agent's EFFECTIVE values (model from swarm.yaml and
    backend defaults) when the genome does not set them. They are excluded when
    mutating so as not to generate no-ops: e.g. swap_model to the model it
    already uses, or temperature to the backend default — different genomes but
    identical behavior that would 'win' through fitness noise.
    """
    effective = effective or {}
    g = {
        "directives": list(parent.genome.get("directives", [])),
        "gen_options": dict(parent.genome.get("gen_options") or {}),
        "model": parent.genome.get("model"),
    }
    # Current effective value of each axis (what the agent ACTUALLY uses today).
    eff_model = g["model"] if g["model"] is not None else effective.get("model")
    eff_temp = g["gen_options"].get("temperature", effective.get("temperature"))
    eff_np = g["gen_options"].get("num_predict", effective.get("num_predict"))

    pool = DIRECTIVES.get(parent.agent_id, [])
    ops = ["add_directive", "temperature", "num_predict"]
    if g["directives"]:
        ops.append("drop_directive")
    if len([m for m in model_pool if m != eff_model]) >= 1 and model_pool:
        ops.append("swap_model")

    op = rng.choice(ops)
    if op == "add_directive":
        candidates = [d for d in pool if d not in g["directives"]]
        if candidates:
            d = rng.choice(candidates)
            g["directives"].append(d)
            mutation = f"añade directiva: {d!r}"
        else:
            op = "temperature"
    if op == "drop_directive":
        d = rng.choice(g["directives"])
        g["directives"].remove(d)
        mutation = f"quita directiva: {d!r}"
    if op == "temperature":
        choices = [t for t in TEMPERATURES if t != eff_temp]
        t = rng.choice(choices) if choices else eff_temp
        g["gen_options"]["temperature"] = t
        mutation = f"temperature -> {t}"
    if op == "num_predict":
        choices = [n for n in NUM_PREDICT if n != eff_np]
        n = rng.choice(choices) if choices else eff_np
        g["gen_options"]["num_predict"] = n
        mutation = f"num_predict -> {n}"
    if op == "swap_model":
        m = rng.choice([m for m in model_pool if m != eff_model])
        g["model"] = m
        mutation = f"model -> {m}"

    return Variant(
        variant_id=child_id,
        agent_id=parent.agent_id,
        generation=parent.generation + 1,
        parent_id=parent.variant_id,
        genome=g,
        mutation=mutation,
    )


def generate_variants(parent: Variant, n: int = 4, rng: random.Random | None = None,
                      model_pool: list[str] | None = None,
                      effective: dict | None = None, seen: set | None = None,
                      id_counter: "itertools.count | None" = None) -> list[Variant]:
    """Produces n children of the parent, each with a different genome.

    `seen`: genome keys ALREADY evaluated (it persists ACROSS generations so
    known losers are not re-scored). `id_counter`: global id counter, so that two
    children never share a variant_id even if the parent's generation does not
    change after a dry generation (otherwise the lineage becomes ambiguous)."""
    import itertools
    rng = rng or random.Random(42)
    model_pool = model_pool or []
    id_counter = id_counter if id_counter is not None else itertools.count()
    # Shared `seen` persists across calls; if not passed, it starts with the parent.
    if seen is None:
        seen = {_genome_key(parent.genome)}
    children: list[Variant] = []
    attempts = 0
    while len(children) < n and attempts < n * 12:
        attempts += 1
        idx = next(id_counter)
        child = mutate(parent, rng, model_pool, effective=effective,
                       child_id=f"{parent.agent_id}-v{idx}")
        key = _genome_key(child.genome)
        if key in seen:
            continue
        seen.add(key)
        children.append(child)
    return children


def genome_to_overrides(genome: dict, base_prompt: str) -> dict:
    """Converts a genome into the override that build_swarm() understands."""
    ov: dict = {}
    directives = genome.get("directives") or []
    if directives:
        ov["system_prompt"] = base_prompt + "\n\nAjustes de esta variante:\n" + \
            "\n".join(f"- {d}" for d in directives)
    if genome.get("gen_options"):
        ov["gen_options"] = dict(genome["gen_options"])
    if genome.get("model"):
        ov["model"] = genome["model"]
    return ov


def _genome_key(g: dict) -> str:
    import json
    return json.dumps(g, sort_keys=True, ensure_ascii=False)
