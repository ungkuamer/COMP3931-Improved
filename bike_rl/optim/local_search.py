"""Local search baseline. See OPTIMIZER_SPEC.md §6."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bike_rl.config import Config
    from bike_rl.objective import ObjectiveWeights
    from bike_rl.optim.greedy import Solution


class LocalSearchSolver:
    """Local search over candidate swaps, initialised from a GreedySolver solution."""

    def __init__(
        self,
        cfg: Config,
        weights: ObjectiveWeights,
        max_iter: int | None = None,
        time_limit_s: float | None = None,
    ) -> None:
        raise NotImplementedError("Optimiser plan")

    def solve(self, graph: object, candidates: list[object], budget: float) -> Solution:
        raise NotImplementedError("Optimiser plan")
