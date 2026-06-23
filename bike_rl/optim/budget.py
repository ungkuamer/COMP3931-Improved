"""Budget/cost utilities shared by all optimiser solvers.

See OPTIMIZER_SPEC.md §2 and §4. ``cost`` delegates to
:func:`bike_rl.candidates.candidate_cost`, the single source of truth for
edge cost shared with the RL env, so the optimisers and the env always agree
on what an edge costs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bike_rl.candidates import candidate_cost

if TYPE_CHECKING:
    from bike_rl.candidates import Candidate
    from bike_rl.config import Config


def cost(edge: Candidate, cfg: Config) -> float:
    """Return the construction cost of ``edge``: ``length * edge_cost_factor``.

    Thin delegate to :func:`bike_rl.candidates.candidate_cost` (single source
    of truth — RECREATE_SPEC §3.4 / OPTIMIZER_SPEC §4) so the optimisers and
    the RL env never drift on cost.

    Args:
        edge: The candidate edge.
        cfg: Config providing ``edge_cost_factor``.

    Returns:
        ``edge.length * cfg.edge_cost_factor``.
    """
    return candidate_cost(edge, cfg)


def remaining_budget(spent: float, budget: float) -> float:
    """Return the non-negative remaining budget: ``max(budget - spent, 0.0)``.

    Clamped at zero so a solver that has overspent (or exactly hit the budget)
    reports no headroom rather than a negative number.

    Args:
        spent: Budget already consumed.
        budget: Total budget.

    Returns:
        ``max(budget - spent, 0.0)``.
    """
    return max(budget - spent, 0.0)
