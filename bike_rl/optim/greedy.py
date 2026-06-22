"""Greedy selection baseline. See OPTIMIZER_SPEC.md §5."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bike_rl.config import Config
    from bike_rl.objective import ObjectiveWeights


@dataclass
class Solution:
    """A solver solution."""

    edges: list[object] = field(default_factory=list)
    objective: float = 0.0
    spent: float = 0.0
    runtime_s: float = 0.0
    solver: str = ""
    extra: dict[str, object] = field(default_factory=dict)


class GreedySolver:
    """Greedy selection: pick best candidate by objective delta, repeat until budget exhausted."""

    def __init__(self, cfg: Config, weights: ObjectiveWeights) -> None:
        raise NotImplementedError("Optimiser plan")

    def solve(self, graph: object, candidates: list[object], budget: float) -> Solution:
        raise NotImplementedError("Optimiser plan")
