"""Phase 1 end-to-end: the swarm solves a task and the fitness scores it ON ITS
OWN, with no human and no real models (FakeBackend). This is the 'phase goal'."""

from core.agent import NanoAgent
from core.backends import FakeBackend
from core.memory.store import MemoryStore
from core.memory.project_index import ProjectIndex
from core.router import Router
from core.swarm import Swarm, build_swarm
from domains.code.tools import build_tools
from domains.code.fitness import CodeFitness
from domains.code.adapter import CodeAdapter


def test_loop_completo_aplica_cambio_y_puntua(tmp_path):
    # The "editor" proposes creating a valid file.
    fake = FakeBackend(lambda m, s, p:
        "Creo el archivo pedido.\n```file path=hola.py\nVALOR = 42\n```\nCONFIDENCE: 0.9")

    catalog = build_tools(str(tmp_path))
    editor = NanoAgent("editor", "mid", "", catalog, backend=fake)
    router = Router({"editor": editor})

    idx = ProjectIndex(str(tmp_path)); idx.build()
    swarm = Swarm(
        router=router,
        memory=MemoryStore(":memory:"),
        index=idx,
        fitness=CodeFitness(require_tests=False),   # tmp has no suite of its own
        adapter=CodeAdapter(str(tmp_path), tools=catalog),
    )

    out = swarm.handle("escribe un archivo hola.py con una constante")

    # The change was really applied to the "project":
    assert (tmp_path / "hola.py").read_text() == "VALOR = 42\n"
    assert out.touched_files == ["hola.py"]
    # The fitness scored it on its own and the memory stored it. Without a suite
    # of its own (require_tests=False) the tests weight is worth 0 and is NOT
    # redistributed, so the ceiling is 0.40 (lint+compiles+tie-break): a valid
    # change gets close to it.
    assert out.score > 0.35
    assert len(swarm.memory.all_episodes()) == 1


def test_build_swarm_cablea_desde_config(tmp_path):
    # Verifies the real wiring from config/*.yaml using a FakeBackend.
    fake = FakeBackend(lambda m, s, p: "ok\nCONFIDENCE: 0.7")
    swarm = build_swarm(str(tmp_path), config_dir="config", backend=fake)

    # The 4 agents from the config were registered, each with its model.
    ids = set(swarm.router.agents)
    assert {"navigator", "editor", "tester", "reviewer"} <= ids
    assert swarm.router.fallback is not None
    # The editor has write tools; the navigator does not.
    assert "write_file" in swarm.router.agents["editor"].tools
    assert "write_file" not in swarm.router.agents["navigator"].tools
