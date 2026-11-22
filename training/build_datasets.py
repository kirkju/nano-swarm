"""training/build_datasets.py — Builds the per-agent LoRA datasets (Phase 2).

Provenance and licenses (details in DATASET.md):
  - editor:    MBPP "sanitized" (google-research-datasets/mbpp, CC-BY-4.0),
               splits train/validation/prompt. The test split is NOT touched.
  - tester:    in-house synthetic — modules with planted bugs are generated and
               pytest is run FOR REAL to capture the tool's actual output.
  - reviewer:  in-house synthetic — clean files and files with planted flaws
               (syntax, division by zero, mutable defaults, wrong values).
  - navigator: in-house synthetic — random mini-projects; locating symbols.

Anti-contamination: the file/function names from eval/suite.py are on a
blacklist and cannot appear in any training example.

Output format (one JSONL per agent in training/data/):
  {"system": <the agent's real system prompt>, "prompt": <real prompt from act()>,
   "response": <target response following the swarm protocol>}

The prompts are built with the SAME functions production uses
(core.swarm._system_prompt and NanoAgent._build_prompt): training with a format
different from the serving one is the classic way to throw the effort away.

Usage:  python -m training.build_datasets
"""

from __future__ import annotations

import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from core.agent import NanoAgent
from core.swarm import _system_prompt

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "training" / "data"
RNG = random.Random(20260701)   # reproducible

# Names used by eval/suite.py: FORBIDDEN in training (contamination).
EVAL_BLACKLIST = {
    "suma", "par", "es_par", "resta", "fib", "parse_config", "settings",
    "timeout", "config", "estructura", "app", "modelos", "servicios",
    "calc", "mult", "ok", "doble", "limpio", "division", "dividir",
    "constantes", "roto",
    # held-out tasks added in Phase 2:
    "biblioteca", "calcular_prestamo", "socios", "multas", "multa_por_retraso",
    "nucleo", "version", "arranque", "cadenas", "recortar", "conversor",
    "a_metros", "plantilla", "rellenar", "utilidades", "par_ordenado",
}


def _agent_spec(agent_id: str) -> dict:
    cfg = yaml.safe_load((ROOT / "config" / "swarm.yaml").read_text(encoding="utf-8"))
    return next(a for a in cfg["agents"] if a["id"] == agent_id)


def _make_prompt_builder(agent_id: str):
    """Returns (system_prompt, build_prompt(task, context, observations)) using
    exactly the same code as production.

    It filters against the REAL tool catalog: training with a prompt that lists
    nonexistent tools would diverge from the serving prompt (build_swarm already
    filters by catalog)."""
    from domains.code.tools import build_tools
    catalog = build_tools(str(ROOT))
    spec = _agent_spec(agent_id)
    tool_names = [n for n in spec.get("tools", []) if n in catalog]
    system = _system_prompt(agent_id, spec.get("purpose", ""), tool_names,
                            spec.get("instructions", ""))
    # NanoAgent is only used here to borrow _build_prompt; it never calls the backend.
    dummy_tools = {n: (lambda **kw: None) for n in tool_names}
    agent = NanoAgent(agent_id, spec["model"], system, dummy_tools, backend=None)
    return system, agent._build_prompt


def _check_blacklist(rows: list[dict], agent: str) -> None:
    for row in rows:
        text = (row["prompt"] + row["response"]).lower()
        for bad in EVAL_BLACKLIST:
            # Whole word: 'config' must not match 'configurar'.
            if re.search(rf"\b{re.escape(bad)}\.py\b", text) or \
               re.search(rf"\bdef {re.escape(bad)}\b", text) or \
               re.search(rf"\b{re.escape(bad)}\(", text):
                raise ValueError(
                    f"[{agent}] ejemplo contaminado con nombre de eval: {bad!r}")


# ============================================================================
# EDITOR — MBPP sanitized (CC-BY-4.0)
# ============================================================================

def build_editor() -> list[dict]:
    from datasets import load_dataset

    system, build_prompt = _make_prompt_builder("editor")
    rows: list[dict] = []
    ds = load_dataset("google-research-datasets/mbpp", "sanitized")
    for split in ("train", "validation", "prompt"):   # test stays untouched
        for ex in ds[split]:
            fn = _first_function_name(ex["code"])
            if fn is None or fn.lower() in EVAL_BLACKLIST:
                continue
            module = fn.lower()
            asserts = "\n".join(ex["test_list"][:2])
            task = (
                f"escribe un archivo {module}.py que resuelva: {ex['prompt']} "
                f"Debe pasar estas pruebas:\n{asserts}"
            )
            code = ex["code"].rstrip() + "\n"
            response = f"```file path={module}.py\n{code}```\nCONFIDENCE: 0.9"
            rows.append({
                "system": system,
                "prompt": build_prompt(task, "", []),
                "response": response,
            })
    return rows


def _first_function_name(code: str) -> str | None:
    m = re.search(r"^def ([a-zA-Z_]\w*)\s*\(", code, re.MULTILINE)
    return m.group(1) if m else None


# ============================================================================
# Shared synthetic vocabulary (no names from the eval)
# ============================================================================

MODULES = ["inventario", "pedidos", "facturas", "clientes", "envios", "almacen",
           "descuentos", "reportes", "monedas", "sensores", "alarmas", "rutas",
           "vehiculos", "nominas", "turnos", "recetas", "stock", "precios"]

FUNCS = [
    # (name, args, correct_body, buggy_body, bug_description, test)
    ("total_con_iva", "precio",
     "return round(precio * 1.21, 2)",
     "return round(precio * 0.21, 2)",
     "multiplica por 0.21 en vez de 1.21 (devuelve solo el IVA, no el total)",
     "assert total_con_iva(100) == 121.0"),
    ("promedio", "valores",
     "return sum(valores) / len(valores)",
     "return sum(valores) / (len(valores) + 1)",
     "divide entre len(valores)+1 en vez de len(valores)",
     "assert promedio([2, 4, 6]) == 4"),
    ("maximo_de_tres", "a, b, c",
     "return max(a, b, c)",
     "return max(a, b)",
     "ignora el tercer argumento c",
     "assert maximo_de_tres(1, 5, 3) == 5\n    assert maximo_de_tres(1, 2, 9) == 9"),
    ("contar_pares", "numeros",
     "return sum(1 for n in numeros if n % 2 == 0)",
     "return sum(1 for n in numeros if n % 2 == 1)",
     "cuenta impares (n % 2 == 1) en vez de pares",
     "assert contar_pares([1, 2, 3, 4]) == 2"),
    ("aplicar_descuento", "precio, pct",
     "return round(precio * (1 - pct / 100), 2)",
     "return round(precio * (pct / 100), 2)",
     "devuelve el descuento en vez del precio con descuento",
     "assert aplicar_descuento(200, 10) == 180.0"),
    ("invertir_cadena", "texto",
     "return texto[::-1]",
     "return texto[::1]",
     "usa paso 1 en el slice, devuelve la cadena igual",
     "assert invertir_cadena('abc') == 'cba'"),
    ("ultimo_elemento", "lista",
     "return lista[-1]",
     "return lista[0]",
     "devuelve el primer elemento en vez del último",
     "assert ultimo_elemento([7, 8, 9]) == 9"),
    ("es_multiplo", "n, m",
     "return n % m == 0",
     "return m % n == 0",
     "comprueba m % n en vez de n % m",
     "assert es_multiplo(10, 5) is True\n    assert es_multiplo(10, 4) is False"),
]

CONSTS = [("LIMITE", "100"), ("TASA", "0.19"), ("MONEDA", "'EUR'"),
          ("REINTENTOS", "5"), ("UMBRAL", "0.75"), ("CAPACIDAD", "64")]


# ============================================================================
# TESTER — synthetic, with REAL pytest outputs
# ============================================================================

def _run_real_tool(project: Path, tool: str) -> dict:
    """Runs run_tests/run_linter for real (same functions as production)."""
    from domains.code.tools import build_tools
    return build_tools(str(project))[tool]()


def build_tester(n_projects: int = 40) -> list[dict]:
    system, build_prompt = _make_prompt_builder("tester")
    rows: list[dict] = []

    for i in range(n_projects):
        name, args, good, bad, why, test = RNG.choice(FUNCS)
        module = RNG.choice(MODULES) + "_util"
        broken = RNG.random() < 0.5
        body = bad if broken else good

        with tempfile.TemporaryDirectory() as proj:
            p = Path(proj)
            (p / f"{module}.py").write_text(
                f"def {name}({args}):\n    return {body.split('return ', 1)[1]}\n",
                encoding="utf-8")
            (p / f"test_{module}.py").write_text(
                f"from {module} import {name}\n\n"
                f"def test_{name}():\n    {test}\n", encoding="utf-8")
            obs = _run_real_tool(p, "run_tests")

        task = RNG.choice([
            "corre los tests y dime cuál falla",
            "corre los tests del proyecto, ¿pasan todos?",
            "ejecuta la suite de tests y resume el resultado",
        ])
        # Round 1: the correct response is ONLY the tool call.
        rows.append({
            "system": system,
            "prompt": build_prompt(task, "", []),
            "response": "TOOL: run_tests {}",
        })
        # Round 2: with the real pytest output, the correct summary.
        observation = f"[run_tests {{}}] -> {obs}"
        if broken:
            answer = (f"Falla test_{name} (en test_{module}.py): la función {name} "
                      f"de {module}.py {why}.\nCONFIDENCE: 0.9")
        else:
            answer = f"Pasan todos los tests (1 passed). El proyecto está verde.\nCONFIDENCE: 0.9"
        rows.append({
            "system": system,
            "prompt": build_prompt(task, "", [observation]),
            "response": answer,
        })

    # Linter cases (clean and with a syntax error), also real.
    for broken in (False, True) * 6:
        module = RNG.choice(MODULES)
        cname, cval = RNG.choice(CONSTS)
        content = f"def f(:\n    pass\n" if broken else f"{cname} = {cval}\n"
        with tempfile.TemporaryDirectory() as proj:
            (Path(proj) / f"{module}.py").write_text(content, encoding="utf-8")
            obs = _run_real_tool(Path(proj), "run_linter")
        task = "corre el linter sobre el proyecto y dime si el código está limpio"
        rows.append({
            "system": system,
            "prompt": build_prompt(task, "", []),
            "response": "TOOL: run_linter {}",
        })
        answer = (f"El linter falla: {module}.py tiene un error de sintaxis; hay que "
                  f"corregirlo.\nCONFIDENCE: 0.9" if broken else
                  "El linter no reporta problemas: el código está limpio.\nCONFIDENCE: 0.9")
        rows.append({
            "system": system,
            "prompt": build_prompt(task, "", [f"[run_linter {{}}] -> {obs}"]),
            "response": answer,
        })
    return rows


# ============================================================================
# REVIEWER — synthetic: ALWAYS read before judging, verdict with a reason
# ============================================================================

FLAWS = [
    # (flawed content, reason for rejection)
    ("def repartir(total, partes):\n    return total / partes\n",
     "no maneja partes == 0: lanzará ZeroDivisionError"),
    ("def agregar(item, lista=[]):\n    lista.append(item)\n    return lista\n",
     "usa una lista mutable como valor por defecto: el estado se comparte entre llamadas"),
    ("def calcular(:\n    pass\n",
     "tiene un error de sintaxis en la firma de la función: no compila"),
    ("def leer_dato(d):\n    try:\n        return d['clave']\n    except:\n        pass\n",
     "usa un except desnudo que traga cualquier excepción y devuelve None en silencio"),
    ("PORCENTAJE_MAXIMO = 150\n",
     "un porcentaje máximo de 150 es sospechoso: los porcentajes van de 0 a 100"),
    ("def obtener_primero(items):\n    return items[0]\n",
     "no maneja la lista vacía: lanzará IndexError con items == []"),
]

CLEAN = [
    "GRAVEDAD = 9.81\nVELOCIDAD_LUZ = 299792458\n",
    "def cuadrado(x):\n    return x * x\n",
    "def saludo_formal(nombre):\n    return f'Estimado {nombre}:'\n",
    "NIVELES = ['bajo', 'medio', 'alto']\n\ndef nivel_valido(n):\n    return n in NIVELES\n",
]


def build_reviewer(n_each: int = 80) -> list[dict]:
    system, build_prompt = _make_prompt_builder("reviewer")
    rows: list[dict] = []

    for i in range(n_each):
        flawed = i % 2 == 0
        module = RNG.choice(MODULES) + RNG.choice(["", "_v2", "_core"])
        fname = f"{module}.py"
        if flawed:
            content, motivo = RNG.choice(FLAWS)
        else:
            content, motivo = RNG.choice(CLEAN), None
        task = RNG.choice([
            f"revisa el archivo {fname} antes de aceptarlo: ¿está bien o tiene algún problema?",
            f"revisa {fname}, ¿lo aceptamos?",
            f"haz code review de {fname} y da tu veredicto",
        ])
        # Round 1: read the file, nothing else.
        rows.append({
            "system": system,
            "prompt": build_prompt(task, "", []),
            "response": f'TOOL: read_file {{"path": "{fname}"}}',
        })
        # Round 2: verdict after seeing the content (same observation format
        # that core/agent.py produces).
        observation = f"[read_file {{'path': '{fname}'}}] -> {content}"
        if flawed:
            answer = f"Se RECHAZA {fname}: {motivo}. Hay que corregirlo antes de aceptar.\nCONFIDENCE: 0.9"
        else:
            answer = f"Se ACEPTA {fname}: el código es correcto y no veo riesgos.\nCONFIDENCE: 0.9"
        rows.append({
            "system": system,
            "prompt": build_prompt(task, "", [observation]),
            "response": answer,
        })
    return rows


# ============================================================================
# NAVIGATOR — synthetic: locating symbols and describing structure
# ============================================================================

def build_navigator(n_each: int = 30) -> list[dict]:
    system, build_prompt = _make_prompt_builder("navigator")
    rows: list[dict] = []

    for _ in range(n_each):
        files = RNG.sample(MODULES, 4)
        target_file = files[0]
        name, args, good, *_ = RNG.choice(FUNCS)
        # RAG-style context identical to the one ProjectIndex.relevant() produces.
        fragments = [f"Fragmentos relevantes del proyecto:"]
        fragments.append(f"\n### {target_file}.py:1\ndef {name}({args}):\n    {good}")
        for f in files[1:3]:
            cname, cval = RNG.choice(CONSTS)
            fragments.append(f"\n### {f}.py:1\n{cname} = {cval}")
        context = "\n".join(fragments)

        task = RNG.choice([
            f"localiza dónde está definida la función {name} y dime en qué archivo vive",
            f"encuentra dónde se define {name} y dime el archivo",
            f"¿dónde está la función {name}?",
        ])
        # Case A: the context already shows it -> answer directly, naming the path.
        rows.append({
            "system": system,
            "prompt": build_prompt(task, context, []),
            "response": f"La función {name} está definida en {target_file}.py.\nCONFIDENCE: 0.9",
        })
        # Case B: no context -> the first response is grep; then answer.
        rows.append({
            "system": system,
            "prompt": build_prompt(task, "", []),
            "response": f'TOOL: grep {{"pattern": "def {name}"}}',
        })
        hits = [f"{target_file}.py:1: def {name}({args}):"]
        obs = f"[grep {{'pattern': 'def {name}'}}] -> {hits}"
        rows.append({
            "system": system,
            "prompt": build_prompt(task, "", [obs]),
            "response": f"La función {name} está definida en {target_file}.py.\nCONFIDENCE: 0.9",
        })

    # Structure tasks: describe which files there are.
    for _ in range(12):
        files = RNG.sample(MODULES, 3)
        listing = sorted(f + ".py" for f in files)
        obs = f"[list_dir {{}}] -> {listing}"
        task = "explica brevemente la estructura del proyecto: qué archivos hay y para qué sirven"
        rows.append({
            "system": system,
            "prompt": build_prompt(task, "", []),
            "response": "TOOL: list_dir {}",
        })
        desc = ", ".join(f"{f}.py" for f in files)
        rows.append({
            "system": system,
            "prompt": build_prompt(task, "", [obs]),
            "response": f"El proyecto tiene {len(files)} módulos: {desc}. Cada uno agrupa "
                        f"la lógica de su nombre.\nCONFIDENCE: 0.85",
        })
    return rows


# ============================================================================
# main
# ============================================================================

BUILDERS = {
    "editor": build_editor,
    "tester": build_tester,
    "reviewer": build_reviewer,
    "navigator": build_navigator,
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for agent, builder in BUILDERS.items():
        rows = builder()
        RNG.shuffle(rows)
        if agent != "editor":   # MBPP checks itself; synthetic sets: belt and suspenders
            _check_blacklist(rows, agent)
        out = OUT_DIR / f"{agent}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary[agent] = len(rows)
        print(f"{agent}: {len(rows)} ejemplos -> {out}")
    print("\nResumen:", summary)


if __name__ == "__main__":
    main()
