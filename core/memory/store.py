"""core/memory/store.py — Episodic memory of the swarm.

Remembers which tasks were done, what worked and with what score. This is what
makes the swarm "improve with use" without training anything in Phase 1, and what
feeds variant selection in Phase 3.

Phase 1 implementation: sqlite (zero dependencies) + retrieval by word overlap.
It is deliberately simple; it can be upgraded to embeddings/vector search later
without changing the interface (remember/recall).
"""

from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


class MemoryStore:
    def __init__(self, path: str = ".swarm/memory.db"):
        self.path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False in case the swarm is used from several threads.
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS episodes (
                   id      INTEGER PRIMARY KEY AUTOINCREMENT,
                   task    TEXT NOT NULL,
                   output  TEXT NOT NULL,
                   agent   TEXT,
                   score   REAL NOT NULL,
                   ts      REAL NOT NULL
               )"""
        )
        self.conn.commit()

    def remember(self, task: str, result, score: float) -> None:
        """Persist (task, result, score)."""
        self.conn.execute(
            "INSERT INTO episodes (task, output, agent, score, ts) VALUES (?,?,?,?,?)",
            (task, getattr(result, "output", str(result)),
             getattr(result, "agent_id", ""), float(score), time.time()),
        )
        self.conn.commit()

    def recall(self, task: str, k: int = 3, min_score: float = 0.5) -> str:
        """Fetch SIMILAR and SUCCESSFUL past episodes as prompt context.

        We only recall what went reasonably well (>= min_score): repeating past
        mistakes does not help. Similarity is word overlap (Jaccard).
        """
        rows = self.conn.execute(
            "SELECT task, output, score FROM episodes WHERE score >= ? ORDER BY id DESC LIMIT 500",
            (min_score,),
        ).fetchall()
        if not rows:
            return ""

        q = _tokens(task)
        scored = []
        for r_task, r_out, r_score in rows:
            t = _tokens(r_task)
            sim = len(q & t) / len(q | t) if (q or t) else 0.0
            if sim > 0:
                scored.append((sim, r_score, r_task, r_out))
        scored.sort(reverse=True)

        if not scored:
            return ""
        lines = ["Episodios pasados relevantes (lo que ya funcionó):"]
        for sim, sc, r_task, r_out in scored[:k]:
            snippet = (r_out or "").strip().replace("\n", " ")[:240]
            lines.append(f"- (similitud {sim:.2f}, fitness {sc:.2f}) {r_task}: {snippet}")
        return "\n".join(lines)

    def all_episodes(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT task, output, agent, score, ts FROM episodes ORDER BY id"
        ).fetchall()
        return [dict(task=t, output=o, agent=a, score=s, ts=ts) for t, o, a, s, ts in rows]

    def close(self) -> None:
        self.conn.close()
