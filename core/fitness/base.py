"""core/fitness/base.py — The fitness signal, abstract.

This is the piece that EVOLUTION needs. Without a reliable fitness, no variant
is generated or selected (see docs/architecture.md). Each domain implements its
own: in code it is free and automatic; in PC it is sparse and often human.
"""

from __future__ import annotations
from abc import ABC, abstractmethod


class FitnessFunction(ABC):
    @abstractmethod
    def score(self, task: str, result, adapter) -> float:
        """Returns a 0..1 score of how well the task was solved.
        It must be as objective and automatic as possible."""
        ...
