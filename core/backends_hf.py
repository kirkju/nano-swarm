"""core/backends_hf.py — Local backend on top of transformers/peft (Phase 2).

Serves a Hugging Face base model with an optional LoRA adapter. It is the engine
that serves the specialized agents: Ollama does not convert Qwen2 safetensors
adapters, so we serve base+LoRA directly with transformers.

config/models.yaml entries handled by this backend:
    small_hf:           {backend: hf, base: "Qwen/Qwen2.5-1.5B-Instruct"}
    reviewer_lora:      {backend: hf, base: "Qwen/Qwen2.5-1.5B-Instruct",
                         adapter: "models/lora/reviewer"}

Decoding: GREEDY by default (deterministic — the Phase 2 acceptance A/B tests
compare runs and we do not want sampling noise). `options` can ask for
temperature > 0 with a fixed seed.

Only one HF model fits in 8 GB of VRAM at a time: the cache keeps the last
(base, adapter) used and evicts the previous one.
"""

from __future__ import annotations

import time
from pathlib import Path

from core.backends import Completion


class HFLocalBackend:
    def __init__(self, models: dict, root: str | Path = "."):
        self.models = models
        self.root = Path(root)
        self._key = None        # (base, adapter) loaded right now
        self._tok = None
        self._model = None

    # ---- loading with eviction (a single network in VRAM) -----------------

    def release(self) -> None:
        """Releases the HF model from VRAM. DispatchBackend calls it before
        serving through Ollama: otherwise the HF model (~3 GB) and the 7B fallback
        (~4.7 GB) would compete for the GPU's 8 GB during escalation."""
        if self._model is None:
            return
        import gc
        import torch
        del self._model
        self._model = None
        self._tok = None
        self._key = None
        gc.collect()
        torch.cuda.empty_cache()

    def _load(self, base: str, adapter: str | None):
        if self._key == (base, adapter):
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # AGGRESSIVE eviction of the previous model: on Windows, leaving two models
        # resident blows up the memory commit and the subprocesses (pytest for the
        # fitness) fail to start with WinError 1455 — and that poisons the whole
        # measurement.
        self.release()

        tok = AutoTokenizer.from_pretrained(base)
        model = AutoModelForCausalLM.from_pretrained(
            base,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )
        if adapter:
            from peft import PeftModel
            adapter_path = self.root / adapter
            if not adapter_path.exists():
                raise FileNotFoundError(
                    f"Adaptador LoRA no encontrado: {adapter_path} "
                    f"(¿corriste python -m training.train_lora?)")
            model = PeftModel.from_pretrained(model, str(adapter_path))
        model.eval()
        self._tok, self._model, self._key = tok, model, (base, adapter)

    # ---- Backend API -------------------------------------------------------

    def generate(self, model: str, system: str, prompt: str,
                 options: dict | None = None) -> Completion:
        import torch

        spec = self.models[model]
        self._load(spec["base"], spec.get("adapter"))
        opts = {"temperature": 0.0, "num_predict": 1024, "seed": 42}
        opts.update(options or {})

        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": prompt}]
        text = self._tok.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        inputs = self._tok(text, return_tensors="pt").to(self._model.device)

        do_sample = opts["temperature"] > 0.0
        if do_sample:
            torch.manual_seed(int(opts["seed"]))
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=int(opts["num_predict"]),
                do_sample=do_sample,
                temperature=opts["temperature"] if do_sample else None,
                top_p=0.9 if do_sample else None,
                pad_token_id=self._tok.pad_token_id or self._tok.eos_token_id,
            )
        dt = time.perf_counter() - t0

        gen = out[0][inputs["input_ids"].shape[1]:]
        return Completion(
            text=self._tok.decode(gen, skip_special_tokens=True),
            tokens=int(gen.shape[0]),
            latency_s=dt,
        )
