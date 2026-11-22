"""eval/suite.py — The swarm's evaluation suite (18 tasks across the four agents).

Each task brings its own temporary mini-project. For editing tasks, the project
includes tests that FAIL until the agent writes the correct file: that makes the
fitness signal `tests_pass` truly discriminative, not decorative. For reading
tasks (navigator/tester/reviewer) an objective check is added (did it name the
right file? did it use the tool it was supposed to?) through CheckedFitness.

This suite is the yardstick for ALL phases:
  - Phase 1: baseline with real models (context only).
  - Phase 2: a LoRA is accepted ONLY if it beats its no-LoRA version here.
  - Phase 3: the variants compete against this suite (or a larger one).

The same suite runs with a real backend (Ollama) or a fake one (offline/CI): with
the fake one, the WIRING is validated, not the intelligence.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from core.backends import FakeBackend
from core.fitness.checked import CheckedFitness
from core.swarm import build_swarm
from domains.code.fitness import CodeFitness


# --------------------------------------------------------------------------
# Objective checks (same signature as FitnessFunction.score, they return 0..1)
# --------------------------------------------------------------------------

def _mentions(result, *terms) -> float:
    """1.0 if the output mentions any of the terms as a WHOLE WORD.

    The closing \\b matters: without it, 'si' matches 'sin'/'siempre' and 'pasa'
    matches 'no pasan', and the check would pay for the opposite verdict."""
    low = (getattr(result, "output", "") or "").lower()
    return 1.0 if any(re.search(r"\b" + re.escape(t.lower()) + r"\b", low)
                      for t in terms) else 0.0


def _verdict(result, positive, negative) -> float:
    """1.0 if the output states the correct verdict and NOT the opposite one.

    `positive` = terms that must appear (e.g. 'acepta'); `negative` = terms
    that invalidate it (e.g. 'rechaza'). This way a degenerate 'always
    reject/approve' policy does not get paid: it measures correctness, not
    lexical echo."""
    if _mentions(result, *negative):
        return 0.0
    return _mentions(result, *positive)


def _used_tool(result, name: str, arg_contains: str | None = None) -> float:
    """1.0 if the agent invoked the tool (optionally with a certain argument)."""
    for call in getattr(result, "tool_calls", []):
        if call.get("name") != name:
            continue
        if arg_contains is None:
            return 1.0
        if any(arg_contains in str(v) for v in call.get("args", {}).values()):
            return 1.0
    return 0.0


def _wrote(adapter, relpath: str) -> float:
    """1.0 if the adapter applied a change to (something containing) that path."""
    return 1.0 if any(relpath in str(p) for p in getattr(adapter, "applied", [])) else 0.0


# --------------------------------------------------------------------------
# Task definitions
# --------------------------------------------------------------------------

@dataclass
class EvalTask:
    name: str
    expect_agent: str                 # which agent it should route to (informational)
    task: str
    files: dict = field(default_factory=dict)     # temporary mini-project
    check: Callable | None = None                 # objective check (0..1)
    require_tests: bool = True        # the base fitness counts the project's tests
    fake_response: str = "Hecho.\nCONFIDENCE: 0.7"   # response of the fake backend


_SMOKE = "def test_smoke():\n    assert True\n"

TASKS: list[EvalTask] = [
    # ---- editor: the project ships tests that fail until the file is written ----
    EvalTask(
        name="editor/suma",
        expect_agent="editor",
        task="escribe un archivo suma.py con una función suma(a, b) que devuelva a + b",
        files={"test_suma.py": "from suma import suma\n\n"
                               "def test_suma():\n    assert suma(2, 3) == 5\n"
                               "    assert suma(-1, 1) == 0\n"},
        check=lambda task, result, adapter: _wrote(adapter, "suma.py"),
        fake_response="```file path=suma.py\ndef suma(a, b):\n    return a + b\n```\nCONFIDENCE: 0.9",
    ),
    EvalTask(
        name="editor/es_par",
        expect_agent="editor",
        task="escribe un archivo par.py con una función es_par(n) que devuelva True si n es par",
        files={"test_par.py": "from par import es_par\n\n"
                              "def test_par():\n    assert es_par(4)\n"
                              "    assert not es_par(3)\n    assert es_par(0)\n"},
        check=lambda task, result, adapter: _wrote(adapter, "par.py"),
        fake_response="```file path=par.py\ndef es_par(n):\n    return n % 2 == 0\n```\nCONFIDENCE: 0.9",
    ),
    EvalTask(
        name="editor/fix_resta",
        expect_agent="editor",
        task="arregla el archivo resta.py: la función resta(a, b) debe devolver a - b y ahora devuelve otra cosa",
        files={
            "resta.py": "def resta(a, b):\n    return a + b\n",
            "test_resta.py": "from resta import resta\n\n"
                             "def test_resta():\n    assert resta(5, 3) == 2\n"
                             "    assert resta(0, 4) == -4\n",
        },
        check=lambda task, result, adapter: _wrote(adapter, "resta.py"),
        fake_response="```file path=resta.py\ndef resta(a, b):\n    return a - b\n```\nCONFIDENCE: 0.9",
    ),

    # ---- navigator: locating things in the project -----------------------------
    EvalTask(
        name="navigator/parse_config",
        expect_agent="navigator",
        task="localiza dónde está definida la función parse_config y dime en qué archivo vive",
        files={
            "settings.py": "def parse_config(path):\n    return {'path': path}\n",
            "main.py": "from settings import parse_config\n\nprint(parse_config('x'))\n",
            "utils.py": "def ayudar():\n    return 'ayuda'\n",
            "test_smoke.py": _SMOKE,
        },
        check=lambda task, result, adapter: _mentions(result, "settings"),
        fake_response="La función parse_config está definida en settings.py.\nCONFIDENCE: 0.9",
    ),
    EvalTask(
        name="navigator/timeout",
        expect_agent="navigator",
        task="encuentra dónde se define la constante TIMEOUT y dime el archivo",
        files={
            "red/config.py": "TIMEOUT = 30\nRETRIES = 3\n",
            "red/__init__.py": "",
            "app.py": "from red.config import TIMEOUT\n\nprint(TIMEOUT)\n",
            "test_smoke.py": _SMOKE,
        },
        check=lambda task, result, adapter: _mentions(result, "config"),
        fake_response="TIMEOUT se define en red/config.py.\nCONFIDENCE: 0.9",
    ),
    EvalTask(
        name="navigator/estructura",
        expect_agent="navigator",
        task="explica brevemente la estructura del proyecto: qué archivos hay y para qué sirven",
        files={
            "app.py": "print('app principal')\n",
            "modelos.py": "class Usuario:\n    pass\n",
            "servicios.py": "def enviar():\n    return True\n",
            "test_smoke.py": _SMOKE,
        },
        check=lambda task, result, adapter: min(
            1.0, (_mentions(result, "app") + _mentions(result, "modelos")
                  + _mentions(result, "servicios")) / 2.0),
        fake_response="El proyecto tiene app.py (entrada), modelos.py (datos) y "
                      "servicios.py (lógica).\nCONFIDENCE: 0.85",
    ),

    EvalTask(
        name="navigator/prestamo",
        expect_agent="navigator",
        task="localiza dónde está definida la función calcular_prestamo y dime en qué archivo vive",
        files={
            "biblioteca.py": "def calcular_prestamo(monto, meses):\n    return monto / meses\n",
            "socios.py": "NOMBRES = ['ana', 'luis']\n",
            "multas.py": "def multa_por_retraso(dias):\n    return dias * 0.5\n",
            "test_smoke.py": _SMOKE,
        },
        check=lambda task, result, adapter: _mentions(result, "biblioteca"),
        fake_response="La función calcular_prestamo está definida en biblioteca.py.\nCONFIDENCE: 0.9",
    ),
    EvalTask(
        name="navigator/version",
        expect_agent="navigator",
        task="encuentra dónde se define la constante VERSION y dime el archivo",
        files={
            "nucleo.py": "VERSION = '2.4.1'\nDEBUG = False\n",
            "arranque.py": "from nucleo import VERSION\n\nprint(VERSION)\n",
            "test_smoke.py": _SMOKE,
        },
        check=lambda task, result, adapter: _mentions(result, "nucleo"),
        fake_response="VERSION se define en nucleo.py.\nCONFIDENCE: 0.9",
    ),

    # ---- tester: running tests/linter and interpreting --------------------------
    EvalTask(
        name="tester/test_roto",
        expect_agent="tester",
        task="corre los tests y dime cuál falla",
        files={
            "calc.py": "def mult(a, b):\n    return a + b\n",
            "test_calc.py": "from calc import mult\n\n"
                            "def test_mult():\n    assert mult(3, 4) == 12\n",
        },
        # The project is red ON PURPOSE: do not penalize the tests dimension of the base.
        require_tests=False,
        check=lambda task, result, adapter: 0.5 * _used_tool(result, "run_tests")
                                            + 0.5 * _mentions(result, "mult"),
        fake_response='TOOL: run_tests {}\nFalla test_mult en test_calc.py: mult(3, 4) '
                      'devuelve 7 en vez de 12.\nCONFIDENCE: 0.85',
    ),
    EvalTask(
        name="tester/suite_verde",
        expect_agent="tester",
        task="corre los tests del proyecto, ¿pasan todos?",
        files={
            "ok.py": "def doble(x):\n    return 2 * x\n",
            "test_ok.py": "from ok import doble\n\n"
                          "def test_doble():\n    assert doble(2) == 4\n",
        },
        # It must say they pass and NOT say they fail (measures the verdict, not echo).
        check=lambda task, result, adapter: 0.5 * _used_tool(result, "run_tests")
            + 0.5 * _verdict(result, ("pasan", "pass", "passed", "verde", "correcto"),
                             ("falla", "fallan", "fallo", "error", "no pasan")),
        fake_response='TOOL: run_tests {}\nSí, pasan todos los tests (1 passed).\nCONFIDENCE: 0.9',
    ),
    EvalTask(
        name="tester/linter",
        expect_agent="tester",
        task="corre el linter sobre el proyecto y dime si el código está limpio",
        files={
            "limpio.py": "VALOR = 1\n",
            "test_smoke.py": _SMOKE,
        },
        check=lambda task, result, adapter: _used_tool(result, "run_linter"),
        fake_response='TOOL: run_linter {}\nEl linter no reporta problemas: el código '
                      'está limpio.\nCONFIDENCE: 0.9',
    ),

    EvalTask(
        name="tester/recortar_rota",
        expect_agent="tester",
        task="corre los tests y dime cuál falla",
        files={
            "cadenas.py": "def recortar(s):\n    return s\n",
            "test_cadenas.py": "from cadenas import recortar\n\n"
                               "def test_recortar():\n"
                               "    assert recortar('  hola  ') == 'hola'\n",
        },
        require_tests=False,
        check=lambda task, result, adapter: 0.5 * _used_tool(result, "run_tests")
                                            + 0.5 * _mentions(result, "recortar", "cadenas"),
        fake_response='TOOL: run_tests {}\nFalla test_recortar en test_cadenas.py: '
                      'recortar no quita los espacios.\nCONFIDENCE: 0.85',
    ),
    EvalTask(
        name="tester/dos_verdes",
        expect_agent="tester",
        task="corre los tests del proyecto, ¿pasan todos?",
        files={
            "conversor.py": "def a_metros(km):\n    return km * 1000\n",
            "test_conversor.py": "from conversor import a_metros\n\n"
                                 "def test_a_metros():\n    assert a_metros(2) == 2000\n\n"
                                 "def test_cero():\n    assert a_metros(0) == 0\n",
        },
        check=lambda task, result, adapter: 0.5 * _used_tool(result, "run_tests")
            + 0.5 * _verdict(result, ("pasan", "pass", "passed", "verde", "correcto"),
                             ("falla", "fallan", "fallo", "error", "no pasan")),
        fake_response='TOOL: run_tests {}\nSí, pasan los 2 tests (2 passed).\nCONFIDENCE: 0.9',
    ),

    # ---- reviewer: reviewing before accepting -----------------------------------
    EvalTask(
        name="reviewer/division",
        expect_agent="reviewer",
        task="revisa el archivo division.py antes de aceptarlo: ¿está bien o tiene algún problema?",
        files={
            "division.py": "def dividir(a, b):\n    return a / b\n",
            "test_smoke.py": _SMOKE,
        },
        check=lambda task, result, adapter: 0.5 * _used_tool(result, "read_file", "division")
                                            + 0.5 * _mentions(result, "cero", "zero", "zerodivision", "excepci", "exception", "divisor"),
        fake_response='TOOL: read_file {"path": "division.py"}\nOjo: dividir(a, b) no '
                      'maneja b == 0; lanzará ZeroDivisionError. Pediría manejar el caso '
                      'del divisor cero antes de aceptar.\nCONFIDENCE: 0.85',
    ),
    EvalTask(
        name="reviewer/constantes_ok",
        expect_agent="reviewer",
        task="revisa el archivo constantes.py, ¿lo aceptamos?",
        files={
            "constantes.py": "PI = 3.14159\nE = 2.71828\n",
            "test_smoke.py": _SMOKE,
        },
        # It must READ and issue an ACCEPT verdict (not reject): this way an
        # 'always reject' policy does not score perfectly on the clean tasks.
        check=lambda task, result, adapter: 0.5 * _used_tool(result, "read_file", "constantes")
            + 0.5 * _verdict(result, ("acepta", "aceptar", "aprueba", "correcto"),
                             ("rechaza", "rechazar", "riesgo", "error", "inválid")),
        fake_response='TOOL: read_file {"path": "constantes.py"}\nSolo define constantes '
                      'con valores correctos: se acepta.\nCONFIDENCE: 0.9',
    ),
    EvalTask(
        name="reviewer/sintaxis_rota",
        expect_agent="reviewer",
        task="revisa el archivo roto.py: ¿es código válido? ¿lo aceptamos?",
        files={
            "roto.py": "def funcion(:\n    pass\n",
            "test_smoke.py": _SMOKE,
        },
        check=lambda task, result, adapter: 0.5 * max(_used_tool(result, "read_file", "roto"),
                                                      _used_tool(result, "run_linter"))
                                            + 0.5 * _mentions(result, "error", "sintaxis", "syntax", "inválid", "invalid", "rechaz", "no compil"),
        fake_response='TOOL: read_file {"path": "roto.py"}\nEl archivo tiene un error de '
                      'sintaxis en la firma de la función; hay que rechazarlo.\nCONFIDENCE: 0.9',
    ),
    EvalTask(
        name="reviewer/eval_peligroso",
        expect_agent="reviewer",
        task="revisa el archivo plantilla.py antes de aceptarlo: ¿está bien o tiene algún problema?",
        files={
            "plantilla.py": "def rellenar(expr):\n    return eval(expr)\n",
            "test_smoke.py": _SMOKE,
        },
        check=lambda task, result, adapter: 0.5 * _used_tool(result, "read_file", "plantilla")
                                            + 0.5 * _mentions(result, "eval", "inyec", "peligro", "inseguro", "riesgo", "arbitrari"),
        fake_response='TOOL: read_file {"path": "plantilla.py"}\nSe RECHAZA: rellenar() '
                      'pasa la entrada directa a eval(), lo que permite ejecutar código '
                      'arbitrario. Es un riesgo de seguridad.\nCONFIDENCE: 0.9',
    ),
    EvalTask(
        name="reviewer/utilidades_ok",
        expect_agent="reviewer",
        task="revisa utilidades.py, ¿lo aceptamos?",
        files={
            "utilidades.py": "def par_ordenado(a, b):\n    return (min(a, b), max(a, b))\n",
            "test_smoke.py": _SMOKE,
        },
        check=lambda task, result, adapter: 0.5 * _used_tool(result, "read_file", "utilidades")
            + 0.5 * _verdict(result, ("acepta", "aceptar", "aprueba", "correcto", "sin riesgo"),
                             ("rechaza", "rechazar", "problema", "peligro", "error")),
        fake_response='TOOL: read_file {"path": "utilidades.py"}\nSe acepta: la función es '
                      'correcta y sin riesgos.\nCONFIDENCE: 0.9',
    ),
]


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

# Files that REDEFINE how pytest/ruff run: creating them is cheating, even if
# the original tests stay intact (e.g. a pytest.ini that ignores the test that
# was supposed to pass, or a conftest.py with collect_ignore).
CONFIG_TAMPER = {
    "pytest.ini", "conftest.py", "tox.ini", "setup.cfg", "pyproject.toml",
    "ruff.toml", ".ruffrc",
}


# Artifact directories that pytest/compileall create while MEASURING (not the
# agent's doing): they are ignored in the snapshot, otherwise a test_x.cpython.pyc
# would trip the guard because it starts with 'test'.
_SNAPSHOT_IGNORE = {"__pycache__", ".pytest_cache", ".swarm", ".ruff_cache"}


def _snapshot(root: str) -> set[str]:
    """Relative paths of the files that exist before the agent acts."""
    base = Path(root)
    return {
        str(p.relative_to(base)) for p in base.rglob("*")
        if p.is_file() and not any(part in _SNAPSHOT_IGNORE for part in p.parts)
    }


def _tamper_check(root: str, protected: dict, pre_snapshot: set[str]) -> list[str]:
    """Returns the signs of metric tampering:
      - rewriting an original test* file (changes THE yardstick), or
      - creating a NEW file that redefines pytest/ruff or injects tests
        (pytest.ini, conftest.py, extra test_*.py, etc.).
    """
    base = Path(root)
    bad: list[str] = []
    for rel, content in protected.items():
        if (base / rel).read_text(encoding="utf-8", errors="replace") != content:
            bad.append(rel)
    for rel in _snapshot(root) - pre_snapshot:      # new files
        name = Path(rel).name
        if name in CONFIG_TAMPER or name.startswith("test") or name.startswith("conftest"):
            bad.append(rel)
    return sorted(set(bad))


def make_fake_backend(tasks: list[EvalTask] | None = None) -> FakeBackend:
    """Fake backend that answers each task of the suite with its fake_response.
    Validates the full wiring (router -> agent -> adapter -> fitness) offline."""
    tasks = tasks or TASKS

    def responder(model, system, prompt):
        for t in tasks:
            if t.task in prompt:
                return t.fake_response
        return "Hecho.\nCONFIDENCE: 0.7"

    return FakeBackend(responder)


def run_suite(backend=None, tasks: list[EvalTask] | None = None,
              agent_overrides: dict | None = None, check_weight: float = 0.5,
              config_dir: str = "config", verbose: bool = True,
              with_fallback: bool = False) -> dict:
    """Runs the suite and returns {mean, per_agent, results}.

    backend=None -> real Ollama (production). Each task runs in an isolated
    temporary project with in-RAM memory: this way the A/B comparisons (Phase 2/3)
    do not contaminate each other across runs.

    with_fallback=False by default: when MEASURING we want the agent in isolation,
    not an agent+escape-to-7B mix (which would bias the A/B and evolution). The
    PRODUCTION swarm (core.swarm.main) does carry the fallback.
    """
    tasks = tasks or TASKS
    results = []
    for t in tasks:
        with tempfile.TemporaryDirectory() as proj:
            for rel, content in t.files.items():
                p = Path(proj) / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content, encoding="utf-8")

            # Anti-cheating guard: the project's tests are THE yardstick.
            # An agent that rewrites them can inflate its fitness (reward hacking),
            # so any task that touches them scores 0, without exceptions.
            protected = {rel: content for rel, content in t.files.items()
                         if Path(rel).name.startswith("test")}
            pre_snapshot = _snapshot(proj)

            fitness = CheckedFitness(
                base=CodeFitness(require_tests=t.require_tests),
                check=t.check,
                check_weight=check_weight,
            )
            swarm = build_swarm(
                proj, config_dir=config_dir, backend=backend,
                fitness=fitness, memory_path=":memory:",
                agent_overrides=agent_overrides, with_fallback=with_fallback,
            )
            out = swarm.handle(t.task)

            tampered = _tamper_check(proj, protected, pre_snapshot)
            score = 0.0 if tampered else out.score
            row = {
                "name": t.name,
                "expect_agent": t.expect_agent,
                "route": out.route_reason,
                "score": score,
                "confidence": out.confidence,
                "tokens": out.cost_tokens,
            }
            if tampered:
                row["tampered"] = tampered
            results.append(row)
            if verbose:
                flag = "  !! tocó los tests -> 0" if tampered else ""
                print(f"  {score:5.3f}  {t.name:26s} [{out.route_reason}]{flag}")

    mean = sum(r["score"] for r in results) / len(results) if results else 0.0
    per_agent: dict[str, list[float]] = {}
    for r in results:
        per_agent.setdefault(r["expect_agent"], []).append(r["score"])
    per_agent_mean = {a: round(sum(v) / len(v), 4) for a, v in per_agent.items()}
    return {"mean": round(mean, 4), "per_agent": per_agent_mean, "results": results}
