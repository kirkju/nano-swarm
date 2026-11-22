# Architecture and design rules

> The architecture and the BUILD ORDER described here are binding: no phase is
> started ahead of its turn.

## What this is

A swarm of **nano-agents** based on **tiny language models** (0.5B–4B,
quantized) that specialize, adapt to a specific project and — only at the
end — evolve by selecting the best variants.

Core philosophy (not to be broken):
- **A single shared core**, two bodies (domains). The brain is the same; only
  the tools and the source of the fitness signal change.
- **Small, narrow models**, not one big model that knows everything. Each agent
  is sharp within its task.
- **The router is the hub.** It decides which agent handles each request and
  escalates to a larger model only when confidence is low.
- **The metric comes before evolution.** Without a reliable fitness function,
  no variant is generated or selected. Period.

## Architecture in one sentence

`request → router → specialized nano-agent(s) → domain tools → fitness measures the result → memory remembers it`

```
core/        domain-agnostic brain (router, base agent, swarm, memory, abstract fitness)
domains/code/   FIRST domain. The fitness signal is FREE (tests, lint, compiles, latency).
domains/pc/     SECOND domain. Sparse signal + 4 mandatory safety locks.
agents/      concrete definitions of each specialized nano-agent
evolution/   LAST. Variant generation (LoRA) and selection. NOT to be touched until fitness works.
eval/        evaluation suites per agent and for the full flow
```

## BUILD ORDER (strict)

Work proceeds in phases. A phase does not start until the previous one works and has been measured.

### Phase 1 — Swarm on code, adaptation through context only
- 3–4 small Qwen nano-agents (see `config/swarm.yaml`).
- **Rule-based** router (no ML yet).
- Project memory/RAG: index the code, remember what worked.
- **Zero weight training.** All the "adaptation" is context + memory.
- Fitness function = the tests of the target project (free and automatic).
- Phase goal: the swarm solves tasks and the fitness scores them on its own.

### Phase 2 — Specialization with LoRA
- Only once Phase 1 works and there are reliable metrics.
- Fine-tune each agent with LoRA/QLoRA on its task (few parameters, modest hardware).
- Still the code domain.

### Phase 3 — Evolution
- Only once measurement is reliable.
- Generate agent variants, test them against the fitness, keep the best ones.
- Without a solid fitness function this is random drift that burns compute. It must not be done earlier.

### Phase 4 — Port to the PC domain
- Same core, new adapter.
- Go in with the 4 locks active and with low-risk tasks first.

## The 4 locks of the PC domain (non-negotiable)

They live in `domains/pc/safety/`. Without them, the system is not deployable.
1. **Sandbox** — act in an isolated environment before touching the real one.
2. **Capability-based permissions** — each agent gets ONLY what it needs, not general access.
3. **Dry-run** — propose and validate the action in a dry run before executing it.
4. **Kill switch + rollback** — be able to kill the swarm and undo what it did.

These locks are what separates "bounded, reversible expansion" from "software that
replicates without control". They are part of the design, not an extra.

## Code rules

- Python 3.11+. Dependencies in `pyproject.toml`.
- Models are served through Ollama or vLLM with quantized weights (see `config/models.yaml`).
- Every nano-agent exposes the same interface (`core/agent.py::NanoAgent`). Specialization is done
  through prompt + tools + (later) LoRA adapter, NOT with a different class per agent.
- Nothing in `evolution/` may run without going through a fitness function from `core/fitness/`.
- Everything in `core/` gets tests in `tests/`.

## Commands

```bash
make install      # installs dependencies
make pull-models  # downloads the tiny models defined in config/models.yaml
make run          # starts the swarm on the code domain
make eval         # runs the evaluation suites
make test         # core tests
```

## Quick glossary

- **nano-agent**: a tiny model + a specialized prompt + a bounded set of tools.
- **router**: classifier/rules that assigns each request and decides whether to escalate.
- **fitness function**: objective signal of whether the result was good. Free in code, expensive in PC.
- **adaptation**: learning the project without touching weights (RAG + memory). Cheap.
- **evolution**: generating and selecting variants. Expensive. Only with reliable fitness.
