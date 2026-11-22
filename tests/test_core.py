"""Core tests: agent, tools, memory, index and fitness."""

import pytest

from core.agent import NanoAgent
from core.backends import FakeBackend
from core.memory.store import MemoryStore
from core.memory.project_index import ProjectIndex
from domains.code.tools import build_tools
from domains.code.fitness import CodeFitness


# ---- Agent: tool loop + parsing --------------------------------------------

def test_agente_ejecuta_tool_de_lectura_y_parsea_cambio():
    # Round 1: the model asks to read a file. Round 2: it proposes a change.
    scripts = iter([
        'Necesito ver el archivo.\nTOOL: read_file {"path": "x.py"}',
        "Listo.\n```file path=x.py\nprint('hola')\n```\nCONFIDENCE: 0.8",
    ])
    fake = FakeBackend(lambda m, s, p: next(scripts))

    calls = []
    tools = {"read_file": lambda path: calls.append(path) or "contenido viejo"}
    agent = NanoAgent("editor", "mid", "", tools, backend=fake)

    res = agent.act("cambia x.py")
    assert calls == ["x.py"]                       # the read tool was executed
    assert len(res.proposed_changes) == 1          # the proposed change was parsed
    assert res.proposed_changes[0].path == "x.py"
    assert res.confidence == 0.8


def test_agente_convierte_tool_write_file_en_cambio_propuesto():
    """Regression: TOOL: write_file {path, content} must become a FileChange
    (it used to be dropped silently and the agent ended up with fitness 0)."""
    fake = FakeBackend(lambda m, s, p:
        'TOOL: write_file {"path": "x.py", "content": "y = 1\\n"}\nCONFIDENCE: 0.8')
    agent = NanoAgent("editor", "mid", "", {"write_file": lambda **k: None},
                      backend=fake)
    res = agent.act("crea x.py")
    assert len(res.proposed_changes) == 1
    assert res.proposed_changes[0].path == "x.py"
    assert res.proposed_changes[0].content == "y = 1\n"


def test_agente_no_ejecuta_herramienta_no_permitida():
    fake = FakeBackend(lambda m, s, p: 'TOOL: run_tests {}\nCONFIDENCE: 0.5')
    agent = NanoAgent("navigator", "small", "", tools={}, backend=fake)  # no run_tests
    res = agent.act("corre los tests")
    # It does not blow up; it simply does not run what it is not allowed to.
    assert res.tool_calls[0]["name"] == "run_tests"


def test_agente_sin_backend_falla_claro():
    agent = NanoAgent("x", "small", "", tools={}, backend=None)
    with pytest.raises(RuntimeError):
        agent.act("hola")


# ---- DispatchBackend: engine routing ---------------------------------------

def test_dispatch_enruta_hf_y_ollama_por_config(monkeypatch):
    from core.backends import DispatchBackend, Completion

    models = {
        "small": {"name": "qwen2.5:1.5b"},
        "small_lora": {"backend": "hf", "base": "Qwen/X", "adapter": "models/lora/x"},
    }
    d = DispatchBackend(models)

    llamadas = []
    class _FakeMotor:
        def __init__(self, tag):
            self.tag = tag
        def generate(self, model, system, prompt, options=None):
            llamadas.append((self.tag, model))
            return Completion(text="ok", tokens=1)
        def release(self):
            llamadas.append((self.tag, "release"))

    d._ollama = _FakeMotor("ollama")
    d._hf = _FakeMotor("hf")

    d.generate("small", "s", "p")           # ollama (releases the resident HF first)
    d.generate("small_lora", "s", "p")       # hf
    assert ("ollama", "small") in llamadas
    assert ("hf", "small_lora") in llamadas
    # The Ollama path releases the resident HF before serving (VRAM safeguard).
    assert ("hf", "release") in llamadas


# ---- Tools: sandbox --------------------------------------------------------

def test_tools_bloquean_ruta_fuera_del_proyecto(tmp_path):
    tools = build_tools(str(tmp_path))
    with pytest.raises(PermissionError):
        tools["read_file"]("../../etc/passwd")


def test_tools_leen_y_escriben_dentro(tmp_path):
    tools = build_tools(str(tmp_path))
    tools["write_file"]("sub/dir/a.py", "x = 1\n")
    assert tools["read_file"]("sub/dir/a.py") == "x = 1\n"
    assert "a.py" in tools["list_dir"]("sub/dir")


# ---- Memory ----------------------------------------------------------------

def test_memoria_recuerda_solo_lo_exitoso():
    from core.agent import AgentResult
    m = MemoryStore(":memory:")
    m.remember("arregla el router", AgentResult(output="ok", confidence=1, agent_id="editor"), 0.9)
    m.remember("rompe todo", AgentResult(output="mal", confidence=0, agent_id="editor"), 0.1)
    recalled = m.recall("arregla el router de nuevo", min_score=0.5)
    assert "arregla el router" in recalled
    assert "rompe todo" not in recalled            # low score -> not recalled


# ---- Index / RAG -----------------------------------------------------------

def test_indice_encuentra_archivo_relevante(tmp_path):
    (tmp_path / "router.py").write_text("def route(task):\n    return 'navigator'\n")
    (tmp_path / "otro.py").write_text("def suma(a, b):\n    return a + b\n")
    idx = ProjectIndex(str(tmp_path))
    idx.build()
    assert len(idx) >= 2
    out = idx.relevant("cómo funciona route")
    assert "router.py" in out


# ---- Fitness ---------------------------------------------------------------

class _FakeAdapter:
    """Fake adapter to test the fitness without a shell."""
    def __init__(self, tests_ok, lint_ok):
        self._t, self._l = tests_ok, lint_ok
    def run_tests(self): return {"ok": self._t}
    def run_linter(self): return {"ok": self._l}


class _Res:
    def __init__(self, cost=0, lat=0.0, changes=None):
        self.cost_tokens, self.latency_s = cost, lat
        self.proposed_changes = changes or []


def test_fitness_todo_verde_da_alto():
    f = CodeFitness()
    s = f.score("t", _Res(), _FakeAdapter(tests_ok=True, lint_ok=True))
    assert s > 0.95                                  # 0.60 + 0.20 + 0.15 + tie-break


def test_fitness_tests_rojos_baja_fuerte():
    f = CodeFitness()
    s_ok = f.score("t", _Res(), _FakeAdapter(True, True))
    s_bad = f.score("t", _Res(), _FakeAdapter(False, True))
    assert s_bad < s_ok
    assert s_bad < 0.5                               # with no tests passing, it stays below half


def test_fitness_se_niega_a_puntuar_con_entorno_roto():
    """If pytest/linter could not even be launched (infra), scoring would be lying."""
    class _AdapterRoto:
        def run_tests(self):
            return {"ok": False, "infra_error": True, "stderr": "WinError 1455"}
        def run_linter(self):
            return {"ok": True}
    f = CodeFitness()
    with pytest.raises(RuntimeError, match="me niego a puntuar"):
        f.score("t", _Res(), _AdapterRoto())


def test_fitness_codigo_invalido_no_suma_compila():
    from core.agent import FileChange
    f = CodeFitness()
    bad = _Res(changes=[FileChange("x.py", "def (:\n")])  # invalid syntax
    good = _Res(changes=[FileChange("x.py", "x = 1\n")])
    s_bad = f.score("t", bad, _FakeAdapter(True, True))
    s_good = f.score("t", good, _FakeAdapter(True, True))
    assert s_good > s_bad
