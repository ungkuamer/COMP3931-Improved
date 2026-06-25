"""Greedy selection baseline. See OPTIMIZER_SPEC.md §5."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import networkx as nx

from bike_rl.candidates import Candidate
from bike_rl.objective import ObjectiveWeights, objective, objective_delta
from bike_rl.optim.budget import cost

if TYPE_CHECKING:
    from bike_rl.config import Config

    _NXGraph = nx.MultiDiGraph[Any, Any, Any]
else:
    _NXGraph = nx.MultiDiGraph


@dataclass
class Solution:
    """A solver solution (OPTIMIZER_SPEC §8).

    Attributes:
        edges: The selected candidate edges.
        objective: Canonical objective value of ``edges`` (shared with RL).
        spent: Total cost of ``edges``.
        runtime_s: Wall-clock solve time in seconds.
        solver: Solver name (e.g. ``"greedy"``).
        extra: Solver-specific metadata (e.g. candidate/selected counts).
    """

    edges: list[Any] = field(default_factory=list)
    objective: float = 0.0
    spent: float = 0.0
    runtime_s: float = 0.0
    solver: str = ""
    extra: dict[str, object] = field(default_factory=dict)


class GreedySolver:
    """Greedy-by-marginal-gain-per-cost selection (OPTIMIZER_SPEC §5).

    Each round, among affordable candidates with positive marginal objective
    gain, pick the one maximising ``objective_delta / cost``; tie-break by
    ``(road_priority, -cost)`` for determinism. Repeat until no affordable
    candidate improves the objective. Deterministic — no seed required.

    Under a cardinality constraint on a monotone submodular objective, greedy
    achieves ``(1 - 1/e)`` of optimal; under a budget constraint the standard
    bound is ``1/2(1 - 1/e)`` (in practice much closer). See OPTIMIZER_SPEC §5.
    """

    def __init__(self, cfg: Config, weights: ObjectiveWeights) -> None:
        self.cfg = cfg
        self.weights = weights

    def solve(
        self,
        graph: _NXGraph,
        candidates: list[Candidate],
        budget: float,
    ) -> Solution:
        """Run greedy selection and return a :class:`Solution`.

        Args:
            graph: The base bike network graph (not mutated).
            candidates: Candidate edges to consider (consumed internally via
                ``list.remove``; the input list is copied first).
            budget: Total budget; never exceeded.

        Returns:
            A ``Solution`` with the selected edges, shared objective value,
            spent cost, runtime, and counts in ``extra``.
        """
        start = time.perf_counter()
        S: list[Candidate] = []
        remaining = list(candidates)
        spent = 0.0
        while remaining:
            # Score = (gain-per-cost, road_priority, -cost, -index, edge).
            # The first three keys are the documented OPTIMIZER_SPEC §5
            # tie-break. The final ``-index`` key is a stable, always-comparable
            # tie-breaker: ``Candidate`` is a frozen dataclass without
            # ``order=True`` (identity = (u, v, length, road_priority), plan
            # 002), so without it ``max`` would compare ``Candidate`` objects
            # and raise ``TypeError`` on exact score ties. ``-index`` makes
            # the earliest candidate in the (current) remaining order win a
            # full tie; it never reorders non-tied candidates.
            scored: list[tuple[float, int, float, int, Candidate]] = []
            for i, e in enumerate(remaining):
                c = cost(e, self.cfg)
                if spent + c > budget:
                    continue
                delta = objective_delta(graph, S, e, self.weights, self.cfg)
                if delta <= 0.0:
                    continue
                scored.append((delta / c, e.road_priority, -c, -i, e))
            if not scored:
                break
            _, _, _, _, best = max(scored)
            S.append(best)
            spent += cost(best, self.cfg)
            remaining.remove(best)
        runtime_s = time.perf_counter() - start
        return Solution(
            edges=list(S),
            objective=objective(graph, S, self.weights, self.cfg),
            spent=spent,
            runtime_s=runtime_s,
            solver="greedy",
            extra={"n_candidates": len(candidates), "n_selected": len(S)},
        )
