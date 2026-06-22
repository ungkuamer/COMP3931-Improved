"""Budget/cost utilities. See OPTIMIZER_SPEC.md."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bike_rl.config import Config


def cost(edge: object, cfg: Config) -> float:
    """Return the cost of an edge: length * edge_cost_factor."""
    raise NotImplementedError("Optimiser plan")


def remaining_budget(spent: float, budget: float) -> float:
    """Return remaining budget."""
    raise NotImplementedError("Optimiser plan")
