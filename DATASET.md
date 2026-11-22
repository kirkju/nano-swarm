# DATASET.md — Provenance and licenses of the training data (Phase 2)

All the datasets are generated/transformed with `python -m training.build_datasets`
and end up in `training/data/<agent>.jsonl`. The prompts are built with the
same functions production uses (`core.swarm._system_prompt` and
`NanoAgent._build_prompt`), so the training format is identical to the serving
format.

**Anti-contamination**: the names of files, functions and constants used by
the evaluation suite (`eval/suite.py`) are on a blacklist
(`EVAL_BLACKLIST` in `training/build_datasets.py`) and the builder fails if any
training example contains them. The evaluation is truly held-out.

## Per agent

### editor — `training/data/editor.jsonl`
- **Source**: [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp)
  (Mostly Basic Python Problems, Google Research), **`sanitized`** configuration
  (hand-verified subset), splits `train`, `validation` and `prompt`.
  The `test` split is **not touched** (it stays available as an extra held-out set).
- **License**: **CC-BY-4.0** (attribution: Google Research, authors of MBPP —
  Austin et al., 2021, "Program Synthesis with Large Language Models").
- **Transformation**: each problem is converted into a swarm task
  ("escribe un archivo X.py que resuelva: … Debe pasar estas pruebas: …") and the
  target response is the reference code in the swarm protocol
  (```` ```file path=…``` ```` block + `CONFIDENCE:`).

### tester — `training/data/tester.jsonl`
- **Source**: **in-house synthetic** (this repo). Mini-projects are generated with
  functions that are correct or have planted bugs (wrong operator, off-by-one,
  ignored argument…), **real pytest** is run on each one with the
  same tools from `domains/code/tools.py`, and the real output is used as the
  observation. Targets: (1) first response = `TOOL: run_tests {}` /
  `TOOL: run_linter {}`, (2) after the observation, a correct summary naming the
  failing test/file.
- **License**: that of the repository (own work; no third-party data).

### reviewer — `training/data/reviewer.jsonl`
- **Source**: **in-house synthetic**. Clean files and files with planted
  flaws (division by zero, mutable default, syntax error, bare `except`,
  suspicious values, `IndexError` on an empty list). Targets:
  (1) first response = `TOOL: read_file {…}` (ALWAYS read before judging),
  (2) an ACEPTA/RECHAZA (accept/reject) verdict naming the specific reason.
- **License**: that of the repository (own work).

### navigator — `training/data/navigator.jsonl`
- **Source**: **in-house synthetic**. Random mini-projects (vocabulary
  disjoint from the evaluation's); tasks about locating functions/constants and
  describing structure. Targets: answer by naming the file path, or
  `TOOL: grep/list_dir` first when the context is not enough.
- **License**: that of the repository (own work).

### Counts (can be regenerated with `python -m training.build_datasets`)
editor 170 · tester 104 · reviewer 160 · navigator 114 examples. The prompts are
built by filtering against the REAL tool catalog (`domains/code/tools.py`),
so training and serving share exactly the same prompt format
(an adversarial review found and fixed an earlier divergence caused by a
`project_index` tool that did not exist in the catalog).

## Base models (weights)

| Use | Model | License |
|---|---|---|
| LoRA base for navigator/tester/reviewer | `Qwen/Qwen2.5-1.5B-Instruct` | Apache-2.0 |
| LoRA base for editor | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | Apache-2.0 |
| Phase 1 serving (Ollama) | `qwen2.5:0.5b/1.5b` | Apache-2.0 |
| Phase 1 serving (Ollama) | `qwen2.5:3b` | Qwen Research License (non-commercial) — only local serving of the editor agent in Phase 1 |
| Fallback (Ollama) | `qwen2.5-coder:7b` | Apache-2.0 |

Note: **Apache-2.0** bases were chosen for the LoRAs on purpose (Qwen's 3B
has a research license, which is why the editor trains on Coder-1.5B).

## Reproducing

```bash
python -m training.build_datasets     # regenerates the 4 JSONL files (fixed seed)
```
