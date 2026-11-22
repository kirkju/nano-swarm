"""core/router.py — The hub of the swarm.

The router decides WHICH agent (or agents) handles each request, and when to
ESCALATE to a larger model because confidence is low. It is the most important
piece: a good router with mediocre agents works; a bad router with excellent
agents does not.

Phase 1: simple rules (by task type).
Phase 2+: it can be a tiny trained classifier.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.agent import NanoAgent, AgentResult


@dataclass
class Route:
    agent: NanoAgent
    reason: str


# Phase 1 rules: keywords -> agent id. ORDER matters (the first match
# wins). Designed for code tasks in Spanish and English.
RULES: list[tuple[str, list[str]]] = [
    ("tester", [
        "test", "tests", "prueba", "pruebas", "falla", "fallo", "error",
        "pytest", "lint", "linter", "cobertura", "coverage",
    ]),
    ("reviewer", [
        "revisa", "revisar", "review", "¿está bien", "esta bien", "huele",
        "code review", "aprueba", "aceptar el cambio",
    ]),
    ("editor", [
        "escribe", "escribir", "modifica", "modificar", "arregla", "arreglar",
        "implementa", "implementar", "refactor", "añade", "agrega", "crea",
        "write", "edit", "fix", "implement", "add", "patch", "parchea",
    ]),
    ("navigator", [
        "localiza", "localizar", "dónde está", "donde esta", "estructura",
        "encuentra", "busca", "explica", "entiende", "navega",
        "where", "find", "locate", "structure", "explain",
    ]),
]


class Router:
    def __init__(self, agents: dict[str, NanoAgent], escalate_below: float = 0.6,
                 fallback: NanoAgent | None = None):
        self.agents = agents
        self.escalate_below = escalate_below
        self.fallback = fallback

    def route(self, task: str) -> Route:
        """Picks the agent for this task (keyword rules).

        If nothing matches clearly, it starts with `navigator` (understand before
        acting). If `navigator` does not exist either, it uses any available agent.
        """
        low = task.lower()
        for agent_id, keywords in RULES:
            if agent_id in self.agents and any(k in low for k in keywords):
                hit = next(k for k in keywords if k in low)
                return Route(self.agents[agent_id], reason=f"regla: '{hit}' -> {agent_id}")

        # No match: start by understanding the structure.
        if "navigator" in self.agents:
            return Route(self.agents["navigator"], reason="default: navigator")
        any_id, any_agent = next(iter(self.agents.items()))
        return Route(any_agent, reason=f"default: sin navigator, uso '{any_id}'")

    def maybe_escalate(self, result: AgentResult, task: str,
                       context: str = "") -> AgentResult:
        """If the agent's confidence falls below the threshold, retries with the
        fallback model (more capable, more expensive). This is the 'central
        coordinator with explicit escalation' mechanism.

        Returns the fallback's result only if it improves confidence; otherwise
        it keeps the original (escalating must not make things worse).
        """
        if self.fallback is None or result.confidence >= self.escalate_below:
            return result

        escalated = self.fallback.act(task, context)
        escalated.tool_calls = (
            [{"name": "_escalated_from", "args": {"agent": result.agent_id}}]
            + escalated.tool_calls
        )
        # The real cost is that of BOTH attempts, whichever result is kept. If the
        # fallback does not improve and we return the original, the cost of the
        # failed attempt must still be added to it (otherwise the fitness rewards
        # exactly the useless escalations for being cheap).
        if escalated.confidence > result.confidence:
            escalated.cost_tokens += result.cost_tokens
            return escalated
        result.cost_tokens += escalated.cost_tokens
        result.tool_calls = (
            [{"name": "_escalated_from", "args": {"agent": result.agent_id,
                                                  "kept": "original"}}]
            + result.tool_calls
        )
        return result
