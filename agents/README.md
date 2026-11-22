# agents/

Concrete definitions of each nano-agent (navigator, editor, tester, reviewer...).
Each one is an instance of `core.agent.NanoAgent`: same code, different prompt
and tools. See `config/swarm.yaml` for the active list.

Do not create a new class per agent. Specialize through configuration.
