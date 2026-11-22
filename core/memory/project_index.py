"""core/memory/project_index.py — Learns the STRUCTURE of the project (RAG).

This is the "cheap adaptation": the swarm learns the project without touching
weights. It indexes the code and returns the relevant fragments as context.

Phase 1 implementation: per-file chunking + retrieval by word overlap (no
dependencies). It is enough for the agents to "see" the relevant code. Moving up
to embeddings + FAISS/sqlite-vec is an internal change that does NOT touch the
interface (build/relevant), as docs/architecture.md asks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


# What to index and what to ignore (cheap and focused on code).
CODE_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".cfg", ".ini"}
IGNORE_DIRS = {".git", "__pycache__", ".swarm", ".venv", "venv", "node_modules", ".pytest_cache"}
CHUNK_LINES = 60   # chunks of ~60 lines: enough for context, not huge


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


@dataclass
class Chunk:
    path: str
    start_line: int
    text: str
    _tok: set[str]


class ProjectIndex:
    def __init__(self, root: str):
        self.root = Path(root).resolve()
        self.chunks: list[Chunk] = []

    def build(self) -> None:
        """Walk the repo, chunk the code files and leave the index ready.

        Idempotent: it rebuilds from scratch every time it is called.
        """
        self.chunks = []
        for path in self._walk():
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            rel = str(path.relative_to(self.root))
            for i in range(0, len(lines), CHUNK_LINES):
                block = "\n".join(lines[i:i + CHUNK_LINES])
                if block.strip():
                    self.chunks.append(
                        Chunk(path=rel, start_line=i + 1, text=block, _tok=_tokens(block))
                    )

    def relevant(self, task: str, k: int = 5) -> str:
        """Return the k most relevant fragments for the task, already formatted
        to go into the agent's prompt."""
        if not self.chunks:
            return ""
        q = _tokens(task)
        if not q:
            return ""

        scored = []
        for c in self.chunks:
            overlap = len(q & c._tok)
            if overlap == 0:
                continue
            # Slight bias towards file names that match the task.
            path_bonus = 2 if q & _tokens(c.path) else 0
            scored.append((overlap + path_bonus, c))
        scored.sort(key=lambda x: x[0], reverse=True)

        if not scored:
            return ""
        out = ["Fragmentos relevantes del proyecto:"]
        for _, c in scored[:k]:
            out.append(f"\n### {c.path}:{c.start_line}\n{c.text}")
        return "\n".join(out)

    def _walk(self):
        for p in self.root.rglob("*"):
            if not p.is_file() or p.suffix not in CODE_SUFFIXES:
                continue
            if any(part in IGNORE_DIRS for part in p.parts):
                continue
            yield p

    def __len__(self) -> int:
        return len(self.chunks)
