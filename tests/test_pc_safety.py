"""Phase 4 tests: one block per lock + the full adapter.

The global property under test: NO action touches the real world without going
through the 4 locks, the default mode is dry, and everything real is reversible.
"""

import pytest

from core.agent import AgentResult, NanoAgent
from domains.pc.actions import PCAction, extract_actions
from domains.pc.adapter import PCAdapter
from domains.pc.fitness import PCFitness
from domains.pc.safety.capabilities import Capability, CapabilityGuard
from domains.pc.safety.dry_run import DryRun
from domains.pc.safety.kill_switch import KillSwitch, KillSwitchActive
from domains.pc.safety.sandbox import Sandbox


def _res(tool_calls=None, agent_id="organizador"):
    return AgentResult(output="ok", confidence=0.9, agent_id=agent_id,
                       tool_calls=tool_calls or [])


def _guard(agent_id="organizador", actions=("move", "mkdir"), paths=("**",)):
    return CapabilityGuard({agent_id: [Capability.make(actions, paths)]})


# ---- Lock 4: kill switch + rollback -----------------------------------------

def test_kill_switch_bloquea_todo(tmp_path):
    kill = KillSwitch(tmp_path / ".swarm")
    kill.activate("prueba")
    adapter = PCAdapter(tmp_path, _guard(), kill=kill)
    with pytest.raises(KillSwitchActive):
        adapter.execute(_res([{"name": "mkdir", "args": {"path": "x"}}]))
    kill.clear()
    out = adapter.execute(_res([{"name": "mkdir", "args": {"path": "x"}}]))
    assert out["ok"]                                   # cleared -> it operates again


def test_rollback_deshace_write_move_y_delete(tmp_path):
    (tmp_path / "a.txt").write_text("original", encoding="utf-8")
    (tmp_path / "b.txt").write_text("para borrar", encoding="utf-8")
    caps = _guard(actions=("write", "move", "delete", "mkdir"))
    adapter = PCAdapter(tmp_path, caps, live=True)     # EXPLICIT live mode

    out = adapter.execute(_res([
        {"name": "write_file", "args": {"path": "a.txt", "content": "pisado"}},
        {"name": "move_file", "args": {"src": "b.txt", "dst": "sub/b.txt"}},
        {"name": "delete_file", "args": {"path": "sub/b.txt"}},
    ]))
    assert out["ok"] and len(out["applied"]) == 3
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "pisado"
    assert not (tmp_path / "b.txt").exists()

    adapter.rollback()
    # World restored: original content, and the deleted file back in its place.
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "original"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "para borrar"
    assert not (tmp_path / "sub" / "b.txt").exists()


# ---- Lock 2: capability-based permissions ------------------------------------

def test_capacidades_default_deny(tmp_path):
    guard = CapabilityGuard({})                        # nobody has anything
    ok, why = guard.authorize("organizador", PCAction("move", {"src": "a", "dst": "b"}))
    assert not ok and "default-deny" in why


def test_capacidades_bloquean_traversal_con_puntos(tmp_path):
    """Regression: 'descargas/../secretos/x' must NOT pass under a
    'descargas/**' scope (lock 2 collapses '..' before matching)."""
    guard = CapabilityGuard({"organizador": [
        Capability.make({"write"}, ("descargas/**",)),
    ]})
    ok, why = guard.authorize("organizador",
                              PCAction("write", {"path": "descargas/../secretos/passwords.txt"}))
    assert not ok and ("raíz" in why or "concedidas" in why)
    # And end to end: the adapter in live mode rejects it and writes NOTHING.
    (tmp_path / "secretos").mkdir()
    (tmp_path / "secretos" / "passwords.txt").write_text("SECRETO", encoding="utf-8")
    adapter = PCAdapter(tmp_path, guard, live=True)
    out = adapter.execute(_res(
        [{"name": "write_file", "args": {"path": "descargas/../secretos/passwords.txt",
                                         "content": "PWNED"}}]))
    assert not out["ok"]
    assert (tmp_path / "secretos" / "passwords.txt").read_text(encoding="utf-8") == "SECRETO"


def test_kill_switch_corta_plan_a_mitad(tmp_path):
    """Regression: a kill switch activated during a multi-action plan stops the
    remaining actions (a stop 'between actions', not only 'between plans')."""
    caps = _guard(actions=("mkdir",))
    kill = KillSwitch(tmp_path / ".swarm")
    adapter = PCAdapter(tmp_path, caps, live=True, kill=kill)

    # Real apply_action that activates the kill after the first folder.
    import domains.pc.adapter as adp
    original = adp.apply_action
    calls = {"n": 0}
    def espia(action, root, kill=None):
        calls["n"] += 1
        out = original(action, root, kill=kill)
        if calls["n"] == 1:
            KillSwitch(tmp_path / ".swarm").activate("parada a mitad")
        return out
    adp.apply_action = espia
    try:
        with pytest.raises(KillSwitchActive):
            adapter.execute(_res([
                {"name": "mkdir", "args": {"path": "a"}},
                {"name": "mkdir", "args": {"path": "b"}},
                {"name": "mkdir", "args": {"path": "c"}},
            ]))
    finally:
        adp.apply_action = original
    assert (tmp_path / "a").exists()           # the first one, yes
    assert not (tmp_path / "b").exists()       # the second one, NOT anymore (kill midway)
    assert not (tmp_path / "c").exists()


def test_capacidades_limitan_accion_y_ruta():
    guard = CapabilityGuard({"organizador": [
        Capability.make({"move"}, ("descargas/**",)),
    ]})
    ok, _ = guard.authorize("organizador",
                            PCAction("move", {"src": "descargas/a.txt", "dst": "descargas/txt/a.txt"}))
    assert ok
    # Action not granted:
    ok, why = guard.authorize("organizador", PCAction("delete", {"path": "descargas/a.txt"}))
    assert not ok and "delete" in why
    # Path outside what was granted (the destination escapes descargas/):
    ok, why = guard.authorize("organizador",
                              PCAction("move", {"src": "descargas/a.txt", "dst": "system32/a.txt"}))
    assert not ok and "system32" in why


def test_adapter_aborta_entero_si_una_accion_es_denegada(tmp_path):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    guard = CapabilityGuard({"organizador": [Capability.make({"mkdir"}, ("**",))]})
    adapter = PCAdapter(tmp_path, guard, live=True)
    out = adapter.execute(_res([
        {"name": "mkdir", "args": {"path": "carpeta"}},          # allowed
        {"name": "delete_file", "args": {"path": "a.txt"}},      # NOT allowed
    ]))
    assert not out["ok"]
    assert "DENEGADO" in out["locks"]["capabilities"]
    assert out["applied"] == []                        # fail-closed: NOTHING was applied
    assert not (tmp_path / "carpeta").exists()


# ---- Lock 3: dry-run -----------------------------------------------------------

def test_dry_run_no_toca_el_disco_y_detecta_errores(tmp_path):
    (tmp_path / "real.txt").write_text("x", encoding="utf-8")
    plan = DryRun().simulate([
        PCAction("move", {"src": "no_existe.txt", "dst": "a.txt"}),
        PCAction("write", {"path": "nuevo.txt", "content": "hola"}),
    ], tmp_path)
    assert not plan.ok                                 # one invalid step -> invalid plan
    assert "no existe" in plan.steps[0].reason
    assert plan.steps[1].ok
    # Pure: nothing was created.
    assert not (tmp_path / "nuevo.txt").exists()


def test_dry_run_bloquea_rutas_protegidas_y_escapes(tmp_path):
    plan = DryRun().simulate([
        PCAction("write", {"path": ".swarm/pc_journal.jsonl", "content": "hack"}),
        PCAction("delete", {"path": "../../fuera.txt"}),
    ], tmp_path)
    assert not plan.ok
    assert "protegida" in plan.steps[0].reason
    assert "fuera de la raíz" in plan.steps[1].reason


def test_dry_run_simula_estado_encadenado(tmp_path):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    # move a -> b and then delete b: valid ONLY if the plan understands its own effect.
    plan = DryRun().simulate([
        PCAction("move", {"src": "a.txt", "dst": "b.txt"}),
        PCAction("delete", {"path": "b.txt"}),
    ], tmp_path)
    assert plan.ok


# ---- Lock 1: sandbox -----------------------------------------------------------

def test_sandbox_ensaya_sin_tocar_lo_real(tmp_path):
    (tmp_path / "doc.txt").write_text("contenido", encoding="utf-8")
    report = Sandbox().run([
        PCAction("mkdir", {"path": "texto"}),
        PCAction("move", {"src": "doc.txt", "dst": "texto/doc.txt"}),
    ], tmp_path)
    assert report.ok
    # The real world was left intact:
    assert (tmp_path / "doc.txt").exists()
    assert not (tmp_path / "texto").exists()


def test_sandbox_reporta_accion_que_revienta(tmp_path):
    report = Sandbox().run([
        PCAction("delete", {"path": "fantasma.txt"}),   # does not exist -> unlink blows up
    ], tmp_path)
    assert not report.ok and "sandbox" in report.error


# ---- Full adapter + fitness -----------------------------------------------------

def test_modo_seco_por_defecto_valida_pero_no_aplica(tmp_path):
    (tmp_path / "informe.txt").write_text("x", encoding="utf-8")
    adapter = PCAdapter(tmp_path, _guard())            # live NOT passed -> dry
    out = adapter.execute(_res([
        {"name": "mkdir", "args": {"path": "docs"}},
        {"name": "move_file", "args": {"src": "informe.txt", "dst": "docs/informe.txt"}},
    ]))
    assert out["ok"] and out["mode"] == "dry-run"
    assert all(v.startswith("ok") for v in out["locks"].values())
    assert out["applied"] == []
    assert (tmp_path / "informe.txt").exists()         # the real world, intact
    assert not (tmp_path / "docs").exists()


def test_extract_actions_convierte_tool_calls_y_bloques_file():
    res = _res([{"name": "move_file", "args": {"src": "a", "dst": "b"}},
                {"name": "list_dir", "args": {}}])     # read: not an action
    from core.agent import FileChange
    res.proposed_changes = [FileChange("notas.txt", "hola")]
    actions = extract_actions(res)
    assert [a.kind for a in actions] == ["move", "write"]


def test_agente_pc_no_inventa_write_por_el_ejemplo_del_prompt():
    """With file_block_changes=False, the ```file``` mechanism (and its example in
    the prompt) is disabled: in PC, writes go through TOOL, with no phantom write."""
    from core.backends import FakeBackend
    # The "model" returns exactly the example block from the code prompt.
    fake = FakeBackend(lambda m, s, p:
        "```file path=ruta/al/archivo.py\n<contenido nuevo>\n```\nCONFIDENCE: 0.8")
    agent = NanoAgent("organizador", "small", "", {"write_file": lambda **k: None},
                      backend=fake, mutating_tools={"write_file"},
                      file_block_changes=False)
    res = agent.act("organiza")
    assert res.proposed_changes == []                  # the example was not parsed
    assert extract_actions(res) == []


def test_pc_fitness_premia_plan_valido_y_castiga_violacion(tmp_path):
    (tmp_path / "x.txt").write_text("x", encoding="utf-8")
    f = PCFitness()

    adapter = PCAdapter(tmp_path, _guard(actions=("move", "mkdir")))
    res = _res([{"name": "mkdir", "args": {"path": "d"}}])
    adapter.execute(res)
    assert f.score("organiza", res, adapter) >= 0.7    # locks green in dry mode

    # Capability violation -> 0.0, no nuance.
    res_mala = _res([{"name": "delete_file", "args": {"path": "x.txt"}}])
    adapter.execute(res_mala)
    assert f.score("borra", res_mala, adapter) == 0.0
