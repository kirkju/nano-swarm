"""core/agent.py — The base nano-agent.

ALL nano-agents share this interface. Specialization is NOT done with a
different subclass per agent, but with:
  - a different system prompt,
  - a bounded set of tools,
  - (Phase 2) a different LoRA adapter.

This keeps the swarm homogeneous and easy to evolve later.

Execution model (important for the safety of Phase 4):
  - During `act()` the agent ONLY runs READ tools (read_file, grep,
    list_dir...). They are how it gathers information.
  - MUTATIONS (writing/patching files) are not executed here: the agent
    PROPOSES them in `result.proposed_changes`, and the domain ADAPTER is the
    one that applies them. In code that is direct; in PC that same point is
    where the 4 locks step in. Keeping this separation from Phase 1 on is what
    lets the core be ported to PC without rewriting it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable


# Tools that MUTATE the project. The agent does not invoke them directly:
# they become proposed changes that the adapter applies.
MUTATING_TOOLS = {"write_file", "apply_patch"}


@dataclass
class FileChange:
    """A proposed change to a file (Phase 1: full-content replacement, which
    is the most robust option with tiny models)."""
    path: str
    content: str


@dataclass
class AgentResult:
    """What an agent returns after acting."""
    output: str
    confidence: float                 # 0..1 — the router uses it to decide on escalation
    tool_calls: list = field(default_factory=list)
    proposed_changes: list[FileChange] = field(default_factory=list)
    cost_tokens: int = 0
    latency_s: float = 0.0
    agent_id: str = ""


class NanoAgent:
    """A tiny model + a prompt + bounded tools."""

    def __init__(
        self,
        agent_id: str,
        model: str,                   # key in config/models.yaml (tiny/small/mid/...)
        system_prompt: str,
        tools: dict[str, Callable],   # only the tools this agent may use
        backend=None,                 # Injected backend (Ollama or Fake). See core/backends.py
        lora_adapter: str | None = None,   # Phase 2: path to the specialized adapter
        max_rounds: int = 3,          # cap on tool rounds (avoids expensive loops)
        gen_options: dict | None = None,   # decoding (temperature...); mutated by the variants
        mutating_tools: set[str] | None = None,   # the PC domain extends this set
        file_block_changes: bool = True,   # ```file```: mechanism for proposing code
    ):
        self.agent_id = agent_id
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools
        self.backend = backend
        self.lora_adapter = lora_adapter
        self.max_rounds = max_rounds
        self.gen_options = gen_options
        # Which tools MUTATE the world (they are not executed in act(); the domain
        # adapter applies them behind its locks). Code: write/patch. PC: also
        # move/copy/delete/mkdir.
        self.mutating_tools = mutating_tools if mutating_tools is not None else MUTATING_TOOLS
        # In code, a change is proposed as a ```file path=...``` block. In PC,
        # writes go through TOOL: write_file, so that mechanism (and its example
        # in the prompt) is disabled to avoid injecting a phantom write.
        self.file_block_changes = file_block_changes

    # ---- Public API ------------------------------------------------------

    def act(self, task: str, context: str = "") -> AgentResult:
        """Takes a task + project context and acts.

        1. Builds the prompt: system_prompt + context (memory/RAG) + task.
        2. Calls the tiny model through the injected backend.
        3. If the model asks for an allowed READ tool, runs it and feeds the
           result back (up to `max_rounds`).
        4. Collects the proposed changes (the adapter applies them later).
        5. Returns an AgentResult with output, confidence and token cost.
        """
        if self.backend is None:
            raise RuntimeError(
                f"El agente '{self.agent_id}' no tiene backend. Inyecta uno "
                f"(OllamaBackend en producción, FakeBackend en tests)."
            )

        observations: list[str] = []
        tool_calls: list[dict] = []
        proposed: list[FileChange] = []
        executed: set[tuple] = set()   # (name, args) already executed: do not repeat
        total_tokens = 0
        total_latency = 0.0
        last_text = ""

        for _ in range(self.max_rounds):
            prompt = self._build_prompt(task, context, observations)
            completion = self.backend.generate(self.model, self.system_prompt, prompt,
                                               options=self.gen_options)
            last_text = completion.text
            total_tokens += completion.tokens
            total_latency += completion.latency_s

            # Proposed changes: accumulated, not executed here.
            if self.file_block_changes:
                proposed.extend(_parse_file_changes(completion.text))

            # Tools requested in this round.
            requests = _parse_tool_calls(completion.text)
            if not requests:
                break  # the model gave its final answer; it asks for no more tools

            made_progress = False
            for call in requests:
                name, args = call["name"], call["args"]
                tool_calls.append(call)
                if name in self.mutating_tools:
                    # Not executed here (the adapter applies it behind its locks).
                    # In the code domain, a `TOOL: write_file {path, content}`
                    # becomes a proposed change: otherwise an editor/fallback that
                    # uses that syntax would end with no changes and fitness 0.
                    # It does not set made_progress (there is no new observation to
                    # react to) and it is deduplicated by (path, content).
                    if (self.file_block_changes and "path" in args
                            and "content" in args):
                        wkey = ("_write", args["path"], args["content"])
                        if wkey not in executed:
                            executed.add(wkey)
                            proposed.append(FileChange(path=args["path"],
                                                       content=args["content"]))
                    continue
                if name not in self.tools:
                    observations.append(
                        f"[denegado] '{name}' no está entre las herramientas de "
                        f"este agente ({list(self.tools)})."
                    )
                    continue
                key = (name, json.dumps(args, sort_keys=True, default=str))
                if key in executed:
                    # Tiny models tend to repeat the same call in a loop; the
                    # result is already above, so it does not count as progress.
                    observations.append(
                        f"[repetido] ya ejecutaste {name} {args}; usa el resultado "
                        "de arriba y da tu respuesta final."
                    )
                    continue
                executed.add(key)
                try:
                    out = self.tools[name](**args)
                except Exception as e:  # the tool failed; the agent must find out
                    out = f"ERROR: {e}"
                observations.append(f"[{name} {args}] -> {out}")
                made_progress = True

            if not made_progress:
                break  # it only asked for mutations or disallowed tools: stop

        # If the visible text ended up empty (the model spent every round asking
        # for tools and never answered), force ONE final synthesis round without
        # tools. Without this, an agent can "work well" and still return
        # nothing.
        if not _strip_control(last_text) and not proposed and observations:
            prompt = self._build_prompt(task, context, observations, final=True)
            completion = self.backend.generate(self.model, self.system_prompt, prompt,
                                               options=self.gen_options)
            last_text = completion.text
            total_tokens += completion.tokens
            total_latency += completion.latency_s
            if self.file_block_changes:
                proposed.extend(_parse_file_changes(completion.text))

        return AgentResult(
            output=_strip_control(last_text),
            confidence=_parse_confidence(last_text),
            tool_calls=tool_calls,
            proposed_changes=proposed,
            cost_tokens=total_tokens,
            latency_s=total_latency,
            agent_id=self.agent_id,
        )

    # ---- internals -------------------------------------------------------

    def _build_prompt(self, task: str, context: str, observations: list[str],
                      final: bool = False) -> str:
        parts = []
        if context:
            parts.append(f"## Contexto del proyecto\n{context}")
        readable = [t for t in self.tools if t not in self.mutating_tools]
        if readable and not final:
            parts.append(
                "## Herramientas de lectura disponibles\n"
                f"{', '.join(readable)}\n"
                'Para usar una, escribe una línea: TOOL: <nombre> {"arg": "valor"}'
            )
        if self.file_block_changes and any(t in self.mutating_tools for t in self.tools):
            parts.append(
                "## Para proponer un cambio de archivo\n"
                "Devuelve un bloque cercado así (contenido COMPLETO del archivo):\n"
                "```file path=ruta/al/archivo.py\n<contenido nuevo>\n```"
            )
        if observations:
            parts.append("## Resultados de herramientas\n" + "\n".join(observations))
        parts.append(f"## Tarea\n{task}")
        if final:
            parts.append(
                "Ya NO puedes usar herramientas. Con los resultados de arriba, "
                "da tu respuesta final AHORA, en prosa clara."
            )
        parts.append("Termina tu respuesta con una línea: CONFIDENCE: <0..1>")
        return "\n\n".join(parts)


# --------------------------------------------------------------------------
# Parsers (tolerant: tiny models are not always tidy)
# --------------------------------------------------------------------------

_TOOL_RE = re.compile(r"^\s*TOOL:\s*([a-zA-Z_]\w*)\s*(\{.*\})?\s*$", re.MULTILINE)
_FILE_RE = re.compile(
    r"```file\s+path=(?P<path>\S+)\s*\n(?P<body>.*?)```",
    re.DOTALL,
)
_CONF_RE = re.compile(r"CONFIDENCE:\s*([01](?:\.\d+)?)", re.IGNORECASE)


def _parse_tool_calls(text: str) -> list[dict]:
    calls = []
    for name, raw_args in _TOOL_RE.findall(text):
        args = {}
        if raw_args:
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}  # malformed arguments: treat as no args
        calls.append({"name": name, "args": args})
    return calls


def _parse_file_changes(text: str) -> list[FileChange]:
    return [
        FileChange(path=m.group("path"), content=m.group("body"))
        for m in _FILE_RE.finditer(text)
    ]


def _parse_confidence(text: str) -> float:
    m = _CONF_RE.search(text)
    if not m:
        return 0.5  # no explicit signal: medium confidence (neither forces nor prevents escalation)
    return max(0.0, min(1.0, float(m.group(1))))


def _strip_control(text: str) -> str:
    """Removes the control lines (TOOL:, CONFIDENCE:, file blocks) from the
    visible output, leaving only the prose answer."""
    text = _FILE_RE.sub("", text)
    text = _TOOL_RE.sub("", text)
    text = _CONF_RE.sub("", text)
    return text.strip()
