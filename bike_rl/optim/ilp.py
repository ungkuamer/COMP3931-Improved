"""ILP oracle baseline using OR-Tools. See OPTIMIZER_SPEC.md §7."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bike_rl.config import Config
    from bike_rl.objective import ObjectiveWeights
    from bike_rl.optim.greedy import Solution


class ILPSolver:
    """ILP solver using OR-Tools CP-SAT for budgeted max-coverage."""

    def __init__(
        self,
        cfg: Config,
        weights: ObjectiveWeights,
        solver: str | None = None,
        time_limit_s: float | None = None,
    ) -> None:
        raise NotImplementedError("Optimiser plan")

    def solve(self, graph: object, candidates: list[object], budget: float) -> Solution:
        raise NotImplementedError("Optimiser plan")
