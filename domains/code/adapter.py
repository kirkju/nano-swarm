"""domains/code/adapter.py — The swarm's body in the CODE domain.

Connects the agent's actions to a real code project: reading/writing files,
applying patches, running tests, git. The fitness signal here is FREE.

The adapter is the ONLY point that MUTATES the project. The agent proposes changes
in `result.proposed_changes`; the adapter applies them using the write tool (the
same one the config grants the agent, for least privilege). This boundary is
exactly where the PC domain (Phase 4) inserts its 4 locks without touching the core.
"""

from __future__ import annotations

from pathlib import Path

from domains.code.tools import build_tools


class CodeAdapter:
    def __init__(self, project_root: str, tools: dict | None = None):
        self.project_root = str(Path(project_root).resolve())
        # If none are injected, it builds its own catalog bound to the root.
        self._tools = tools or build_tools(self.project_root)
        self.applied: list[str] = []   # log of what was applied (basis for future rollback)

    def execute(self, result) -> list[str]:
        """Applies the changes proposed by the agent inside project_root.

        Returns the list of touched files. If the agent proposed no changes
        (e.g. navigator, which only reads), it does nothing and returns [].
        """
        write = self._tools["write_file"]
        touched: list[str] = []
        for change in getattr(result, "proposed_changes", []):
            write(change.path, change.content)
            touched.append(change.path)
            self.applied.append(change.path)
        return touched

    # Convenience for the fitness: run the target project's suite.
    def run_tests(self) -> dict:
        return self._tools["run_tests"]()

    def run_linter(self) -> dict:
        return self._tools["run_linter"]()
