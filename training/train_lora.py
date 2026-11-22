"""training/train_lora.py — Per-agent SFT with LoRA, with hard CAPS (Phase 2).

Caps (non-negotiable, so as not to burn days of GPU):
  --max-steps    cap on optimization steps (default 300)
  --max-minutes  time cap for the run (default 20); when it is reached, training
                 is cut off and whatever there is gets saved.
  In addition, the effective steps are limited to ~4 epochs of the dataset: with
  small protocol datasets, more epochs = memorizing, not learning.

The loss is computed ONLY on the response (the prompt tokens are set to -100):
we want to teach the swarm protocol, not how to regenerate the prompt.

Output: models/lora/<agent>/  (adapter_model.safetensors + adapter_config.json
        + training_meta.json with provenance, hyperparameters and losses)

Usage:
    python -m training.train_lora --agent reviewer
    python -m training.train_lora --agent editor --base Qwen/Qwen2.5-Coder-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = "Qwen/Qwen2.5-1.5B-Instruct"


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_dataset(rows, tokenizer, max_seq: int):
    """Tokenizes each example with the real chat template and masks the prompt."""
    examples = []
    dropped = 0
    for row in rows:
        messages = [
            {"role": "system", "content": row["system"]},
            {"role": "user", "content": row["prompt"]},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        full_text = prompt_text + row["response"] + tokenizer.eos_token

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        if len(full_ids) > max_seq:
            dropped += 1
            continue
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        examples.append({"input_ids": full_ids, "labels": labels})
    if dropped:
        print(f"  ({dropped} ejemplos descartados por superar {max_seq} tokens)")
    return examples


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        import torch
        width = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            n = width - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [self.pad_id] * n)
            labels.append(b["labels"] + [-100] * n)
            attn.append([1] * len(b["input_ids"]) + [0] * n)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn),
        }


def main():
    ap = argparse.ArgumentParser(description="Entrena un LoRA por agente (con topes)")
    ap.add_argument("--agent", required=True,
                    choices=["navigator", "editor", "tester", "reviewer"])
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--data", default=None, help="JSONL (default training/data/<agent>.jsonl)")
    ap.add_argument("--out", default=None, help="salida (default models/lora/<agent>)")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--max-minutes", type=float, default=20.0)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-seq", type=int, default=1536)
    ap.add_argument("--val-frac", type=float, default=0.06)
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainerCallback, TrainingArguments)

    data_path = Path(args.data) if args.data else ROOT / "training" / "data" / f"{args.agent}.jsonl"
    out_dir = Path(args.out) if args.out else ROOT / "models" / "lora" / args.agent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"== LoRA {args.agent} | base={args.base} | datos={data_path}")
    rows = load_rows(data_path)
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = build_dataset(rows, tokenizer, args.max_seq)
    n_val = max(4, int(len(examples) * args.val_frac))
    val_set, train_set = examples[:n_val], examples[n_val:]
    print(f"  {len(train_set)} train / {len(val_set)} val")

    # Step cap: whatever the user asks for, but never more than ~4 epochs.
    eff_batch = args.batch_size * args.grad_accum
    steps_per_epoch = max(1, math.ceil(len(train_set) / eff_batch))
    max_steps = min(args.max_steps, steps_per_epoch * 4)
    print(f"  pasos: {max_steps} (tope pedido {args.max_steps}, "
          f"{steps_per_epoch} pasos/época, tope 4 épocas) | "
          f"tope de tiempo: {args.max_minutes} min")

    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora = LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    class TimeLimit(TrainerCallback):
        """Hard time cap: it cuts training short, it does not crash it."""
        def __init__(self, minutes: float):
            self.deadline = time.time() + minutes * 60

        def on_step_end(self, targs, state, control, **kw):
            if time.time() > self.deadline:
                print(f"  [tope de tiempo] corte en el paso {state.global_step}")
                control.should_training_stop = True
            return control

    targs = TrainingArguments(
        output_dir=str(out_dir / "trainer"),
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=min(20, max_steps // 10),
        logging_steps=10,
        save_strategy="no",              # we only save the final adapter
        eval_strategy="no",
        bf16=True,
        report_to=[],
        seed=42,
        dataloader_pin_memory=False,     # Windows
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_set,
                      data_collator=Collator(tokenizer.pad_token_id),
                      callbacks=[TimeLimit(args.max_minutes)])

    def val_loss() -> float:
        collate = Collator(tokenizer.pad_token_id)
        model.eval()
        losses = []
        with torch.no_grad():
            for i in range(0, len(val_set), 2):
                batch = collate(val_set[i:i + 2])
                batch = {k: v.to(model.device) for k, v in batch.items()}
                losses.append(model(**batch).loss.item())
        model.train()
        return sum(losses) / len(losses)

    loss_before = val_loss()
    print(f"  pérdida val ANTES: {loss_before:.4f}")
    t0 = time.time()
    result = trainer.train()
    minutes = (time.time() - t0) / 60
    loss_after = val_loss()
    print(f"  pérdida val DESPUÉS: {loss_after:.4f}  ({minutes:.1f} min)")

    model.save_pretrained(str(out_dir))
    meta = {
        "agent": args.agent,
        "base": args.base,
        "data": str(data_path.relative_to(ROOT)),
        "n_train": len(train_set), "n_val": len(val_set),
        "steps_run": int(result.global_step), "max_steps": max_steps,
        "minutes": round(minutes, 1), "max_minutes": args.max_minutes,
        "lr": args.lr, "rank": args.rank,
        "val_loss_before": round(loss_before, 4),
        "val_loss_after": round(loss_after, 4),
        "train_loss_final": round(result.training_loss, 4),
    }
    (out_dir / "training_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"== guardado en {out_dir}")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
