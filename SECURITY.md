# SECURITY.md — The 4 locks of the PC domain

The PC domain (`domains/pc/`) lets nano-agents act on real files.
What makes that "bounded, reversible expansion" instead of "software that
replicates without control" is **4 mandatory locks**, in this order, in
`domains/pc/adapter.py::PCAdapter.execute`:

| # | Lock | File | What it guarantees |
|---|---|---|---|
| 4 | Kill switch + rollback | `safety/kill_switch.py` | Emergency stop (`.swarm/KILL`) checked **before every action**; append-only journal with backups to undo everything |
| 2 | Capability-based permissions | `safety/capabilities.py` | Default-deny: each agent gets only specific `(actions, paths)`; paths are **normalized by collapsing `..`** before matching (without this, a `descargas/**` scope would let `descargas/../otro` escape) |
| 3 | Dry-run | `safety/dry_run.py` | A **pure** plan (it does not touch the disk) that validates the jail, protected paths (`.swarm`, `.git`) and chained state; one invalid step invalidates the whole plan |
| 1 | Sandbox | `safety/sandbox.py` | Rehearsal of the full plan on a **replica** of the tree, with postcondition checks, before promoting it to real |

## Dry mode by default

`PCAdapter(root, caps)` starts in **dry mode** (`live=False`): it validates the 4
locks and returns the plan, but **does not touch the real world**. Only an explicit
`live=True` promotes it to real, and even then it re-checks the kill switch before
every action and records everything for the rollback.

## How to stop and revert

```bash
# Emergency stop (any process/human can create it):
echo stop > .swarm/KILL

# Undo what the swarm applied in live mode:
python -c "from domains.pc.safety.kill_switch import KillSwitch; \
           print(KillSwitch('.swarm').rollback())"
```

## Test coverage

`tests/test_pc_safety.py` covers each lock separately and the full adapter,
including regressions for two bypasses found by an adversarial review:
- **path traversal** with `..` that escaped the granted capability scope, and
- a **kill switch** that only stopped between plans and not between the actions of a long plan.

The low-risk end-to-end demonstration (dry → live → rollback) is in
`domains/pc/demo.py`:

```bash
python -m domains.pc.demo                  # dry (recommended)
python -m domains.pc.demo --live --rollback
```
