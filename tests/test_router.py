"""Router tests. The router is the hub of Phase 1, so it is tested first.

No real model is needed: route() only picks an agent by rules, and escalation is
tested with a deterministic FakeBackend.
"""

from core.agent import NanoAgent, AgentResult
from core.backends import FakeBackend
from core.router import Router


def _agent(agent_id, backend=None, model="small"):
    return NanoAgent(agent_id, model, system_prompt="", tools={}, backend=backend)


def _swarm_agents():
    return {a: _agent(a) for a in ("navigator", "editor", "tester", "reviewer")}


def test_route_localiza_va_a_navigator():
    r = Router(_swarm_agents())
    assert r.route("localiza dónde está la función de routing").agent.agent_id == "navigator"


def test_route_escribe_va_a_editor():
    r = Router(_swarm_agents())
    assert r.route("escribe una función que sume dos números").agent.agent_id == "editor"


def test_route_tests_va_a_tester():
    r = Router(_swarm_agents())
    assert r.route("corre los tests, parece que algo falla").agent.agent_id == "tester"


def test_route_revisa_va_a_reviewer():
    r = Router(_swarm_agents())
    assert r.route("revisa este cambio antes de aceptarlo").agent.agent_id == "reviewer"


def test_route_sin_match_default_navigator():
    r = Router(_swarm_agents())
    # Ambiguous sentence without strong keywords -> default navigator (understand first).
    assert r.route("mmm a ver qué onda con esto").agent.agent_id == "navigator"


def test_escala_cuando_confianza_baja():
    # The fallback "answers" with high confidence; the primary had low confidence.
    fallback = _agent("fallback", backend=FakeBackend(
        lambda model, system, prompt: "Lo resolví con más cuidado.\nCONFIDENCE: 0.95"
    ), model="fallback")
    r = Router(_swarm_agents(), escalate_below=0.6, fallback=fallback)

    low = AgentResult(output="no estoy seguro", confidence=0.3, agent_id="editor", cost_tokens=10)
    out = r.maybe_escalate(low, task="arregla esto")

    assert out.confidence == 0.95                       # it returned the fallback's result
    assert out.cost_tokens >= 10                        # adds up the cost of both attempts
    assert out.tool_calls[0]["name"] == "_escalated_from"


def test_no_escala_si_confianza_alta():
    fallback = _agent("fallback", backend=FakeBackend(lambda m, s, p: "no debería usarse"))
    r = Router(_swarm_agents(), escalate_below=0.6, fallback=fallback)

    high = AgentResult(output="seguro", confidence=0.9, agent_id="editor")
    out = r.maybe_escalate(high, task="x")
    assert out is high                                  # it touched nothing


def test_escalar_sin_mejora_conserva_original_pero_suma_su_coste():
    """Regression: if the fallback does NOT improve, the original is returned but
    with the cost of the failed attempt added (otherwise the fitness rewards
    useless escalations)."""
    fallback = _agent("fallback", backend=FakeBackend(
        lambda model, system, prompt: "peor intento.\nCONFIDENCE: 0.2"), model="fallback")
    r = Router(_swarm_agents(), escalate_below=0.6, fallback=fallback)

    low = AgentResult(output="dudoso", confidence=0.4, agent_id="editor", cost_tokens=10)
    out = r.maybe_escalate(low, task="arregla esto")
    assert out is low                                   # it kept the original (better)
    assert out.cost_tokens > 10                         # but it was charged the fallback's cost
    assert out.tool_calls[0]["name"] == "_escalated_from"
