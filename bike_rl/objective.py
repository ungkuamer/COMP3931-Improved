"""Canonical scalar objective — the single source of truth for 'what we optimise'.

Both the RL reward (derived) and the direct optimisers (direct call) use this
module so RL and optimisers solve exactly the same problem. See
OPTIMIZER_SPEC.md §3 and RESEARCH_DIRECTION.md §4.1-4.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bike_rl.config import Config


@dataclass(frozen=True)
class ObjectiveWeights:
    """Weights for the canonical objective (OPTIMIZER_SPEC §3.1).

    Attributes:
        connectivity: Weight on graph connectivity in [0,1].
        coverage: Weight on bike-lane-reachable population (coverage) in [0,1].
        fragmentation: Weight on the fragmentation penalty in [0,1]; subtracted.
    """

    connectivity: float = 0.4
    coverage: float = 0.4
    fragmentation: float = 0.2


def objective(
    graph: object,
    added_edges: list[tuple[int | str, int | str]],
    weights: ObjectiveWeights,
    cfg: Config,
) -> float:
    """Return the canonical objective value of ``graph`` with ``added_edges``.

    Higher is better. Pure and deterministic.
    """
    raise NotImplementedError


def objective_delta(
    graph: object,
    added_edges: list[tuple[int | str, int | str]],
    new_edge: tuple[int | str, int | str],
    weights: ObjectiveWeights,
    cfg: Config,
) -> float:
    """Return the marginal objective change from adding ``new_edge``."""
    raise NotImplementedError


def apply_added_edges(graph: object, added_edges: list[tuple[int | str, int | str]]) -> object:
    """Return a copy of ``graph`` with ``added_edges`` applied (bike_lane='yes')."""
    raise NotImplementedError
