"""core/backends.py — Where the tiny models' text comes from.

The core must NOT know whether it is talking to Ollama, vLLM or a fake model. That
is why every agent receives an injected `Backend`. This is what makes Phase 1
testable without a downloaded model: in production you use `OllamaBackend`, in the
tests you use `FakeBackend`.

Minimal contract of a backend:
    generate(model, system, prompt) -> Completion(text, tokens)

`model` is the KEY in config/models.yaml (tiny/small/mid/fallback), not the real name;
the backend resolves it. This way the rest of the code never hardcodes "qwen2.5:0.5b".
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol, Callable


@dataclass
class Completion:
    """What a backend returns after generating."""
    text: str
    tokens: int           # approximate cost in tokens (for fitness and tie-breaks)
    latency_s: float = 0.0


class Backend(Protocol):
    """Anything that can turn (model, system, prompt) into text.

    `options` is an optional dict of decoding parameters (temperature,
    num_predict...). The Phase 3 variants mutate these values, which is why the
    contract already accepts them.
    """

    def generate(self, model: str, system: str, prompt: str,
                 options: dict | None = None) -> Completion: ...


class OllamaBackend:
    """Real backend: talks to a local Ollama server (config/models.yaml).

    `models` is the already loaded `models:` dict from config/models.yaml, e.g.
        {"tiny": {"name": "qwen2.5:0.5b", ...}, "small": {...}, ...}
    So `generate("small", ...)` resolves to its real `name`.
    """

    def __init__(self, models: dict, host: str = "http://localhost:11434",
                 timeout: float = 120.0):
        self.models = models
        self.host = host.rstrip("/")
        self.timeout = timeout

    def _resolve(self, model: str) -> str:
        if model not in self.models:
            raise KeyError(
                f"Modelo '{model}' no está en config/models.yaml "
                f"(claves: {list(self.models)})"
            )
        return self.models[model]["name"]

    def generate(self, model: str, system: str, prompt: str,
                 options: dict | None = None) -> Completion:
        real_name = self._resolve(model)
        # Low temperature, bounded output and fixed seed by default: stable,
        # cheap and REPRODUCIBLE behavior (the Phase 2-3 A/B tests compare runs).
        # Each agent/variant can override them through `options`.
        opts = {"temperature": 0.2, "num_predict": 1024, "seed": 42}
        opts.update(options or {})
        payload = json.dumps({
            "model": real_name,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "options": opts,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                f"No pude contactar Ollama en {self.host}. "
                f"¿Corriste `make pull-models` y está `ollama serve` activo? ({e})"
            ) from e
        dt = time.perf_counter() - t0

        text = data.get("response", "")
        # Ollama reports the real count of generated tokens when it is available.
        tokens = int(data.get("eval_count") or _approx_tokens(text))
        return Completion(text=text, tokens=tokens, latency_s=dt)


class DispatchBackend:
    """Routes each model KEY to its engine according to config/models.yaml.

    Entries without a `backend` field (or backend: ollama) go to Ollama; entries
    with `backend: hf` go to HFLocalBackend (base + optional LoRA, Phase 2). This
    way a single swarm mixes agents served by Ollama with specialized agents
    served by transformers, without the core noticing.
    """

    def __init__(self, models: dict, host: str = "http://localhost:11434",
                 root: str = "."):
        self.models = models
        self._ollama = OllamaBackend(models, host=host)
        self._hf = None       # lazy: torch is only imported if some agent uses it
        self._root = root

    def generate(self, model: str, system: str, prompt: str,
                 options: dict | None = None) -> Completion:
        spec = self.models.get(model, {})
        if spec.get("backend") == "hf":
            if self._hf is None:
                from core.backends_hf import HFLocalBackend
                self._hf = HFLocalBackend(self.models, root=self._root)
            return self._hf.generate(model, system, prompt, options)
        # Ollama path: if an HF model is resident, release it first. The critical
        # case is escalation (HF agent with low confidence -> 7B fallback through
        # Ollama): the two do not fit in 8 GB of VRAM at the same time.
        if self._hf is not None:
            self._hf.release()
        return self._ollama.generate(model, system, prompt, options)


class FakeBackend:
    """Deterministic backend for tests and for running the loop without models.

    You pass it a `responder(model, system, prompt) -> str` function and it
    returns that. Useful for testing the router, the swarm loop and the fitness
    end-to-end without a network.
    """

    def __init__(self, responder: Callable[[str, str, str], str]):
        self.responder = responder
        self.calls: list[dict] = []   # history, handy for asserts in tests

    def generate(self, model: str, system: str, prompt: str,
                 options: dict | None = None) -> Completion:
        text = self.responder(model, system, prompt)
        self.calls.append({"model": model, "system": system, "prompt": prompt,
                           "options": options})
        return Completion(text=text, tokens=_approx_tokens(text), latency_s=0.0)


def _approx_tokens(text: str) -> int:
    """Cheap token estimate for when the backend does not give the real count.

    ~4 characters per token is the usual rule of thumb. It is not exact, but it
    is more than enough to break ties by cost in the fitness.
    """
    return max(1, len(text) // 4)
