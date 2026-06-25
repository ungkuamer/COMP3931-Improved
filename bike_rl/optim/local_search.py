"""Local search baseline: greedy seed + 1-opt/2-opt polish. See OPTIMIZER_SPEC.md §6."""

from __future__ import annotations

import itertools
import time
from typing import TYPE_CHECKING, Any

import networkx as nx

from bike_rl.objective import ObjectiveWeights, objective
from bike_rl.optim.budget import cost
from bike_rl.optim.greedy import GreedySolver, Solution

if TYPE_CHECKING:
    from bike_rl.candidates import Candidate
    from bike_rl.config import Config

    _NXGraph = nx.MultiDiGraph[Any, Any, Any]
else:
    _NXGraph = nx.MultiDiGraph


class LocalSearchSolver:
    """Greedy seed refined by 1-opt swaps and 2-opt exchanges (OPTIMIZER_SPEC §6).

    Seeds from :class:`GreedySolver`, then repeatedly applies first-improvement
    moves until a local optimum is reached or ``max_iter``/``time_limit_s``
    bounds it. Neighbourhoods: 1-opt (replace one chosen edge with one
    non-chosen edge fitting the freed budget) and 2-opt (drop one chosen edge,
    add two non-chosen edges fitting the freed budget). Deterministic given
    the greedy seed — no RNG.

    Never returns a solution worse than its greedy seed: only strict
    objective improvements (``> cur + 1e-9``) are accepted.
    """

    def __init__(
        self,
        cfg: Config,
        weights: ObjectiveWeights,
        max_iter: int | None = None,
        time_limit_s: float | None = None,
    ) -> None:
        self.cfg = cfg
        self.weights = weights
        self.max_iter = cfg.local_search_max_iter if max_iter is None else max_iter
        self.time_limit_s = cfg.local_search_time_limit_s if time_limit_s is None else time_limit_s

    def solve(
        self,
        graph: _NXGraph,
        candidates: list[Candidate],
        budget: float,
    ) -> Solution:
        """Run greedy + local search and return a :class:`Solution`.

        Args:
            graph: The base bike network graph (not mutated).
            candidates: Candidate edges to consider (copied internally; the
                input list is not mutated).
            budget: Total budget; never exceeded.

        Returns:
            A ``Solution`` with ``solver="local_search"`` and ``extra``
            carrying ``n_candidates``/``n_selected``/``iterations``/
            ``seed_solver``/``seed_objective``.
        """
        start = time.perf_counter()
        seed = GreedySolver(self.cfg, self.weights).solve(graph, candidates, budget)
        S: list[Candidate] = list(seed.edges)
        cand: list[Candidate] = list(candidates)
        spent = seed.spent
        eps = 1e-9
        iterations = 0
        improved = True
        while (
            improved
            and iterations < self.max_iter
            and (time.perf_counter() - start) < self.time_limit_s
        ):
            improved = False
            iterations += 1
            cur = objective(graph, S, self.weights, self.cfg)
            sset = set(S)
            non_chosen: list[Candidate] = [e for e in cand if e not in sset]
            # 1-opt: replace one chosen edge with one non-chosen edge.
            for e_out in S:
                freed = cost(e_out, self.cfg)
                head = budget - (spent - freed)
                for e_in in non_chosen:
                    if (time.perf_counter() - start) >= self.time_limit_s:
                        break
                    if cost(e_in, self.cfg) <= head + eps:
                        new_s = [e for e in S if e is not e_out]
                        new_s.append(e_in)
                        if objective(graph, new_s, self.weights, self.cfg) > cur + eps:
                            S = new_s
                            spent = spent - freed + cost(e_in, self.cfg)
                            improved = True
                            break
                if improved or (time.perf_counter() - start) >= self.time_limit_s:
                    break
            if improved:
                continue
            # 2-opt: drop one chosen edge, add two non-chosen edges.
            for e_out in S:
                freed = cost(e_out, self.cfg)
                head = budget - (spent - freed)
                for e1, e2 in itertools.combinations(non_chosen, 2):
                    if (time.perf_counter() - start) >= self.time_limit_s:
                        break
                    if cost(e1, self.cfg) + cost(e2, self.cfg) <= head + eps:
                        new_s = [e for e in S if e is not e_out]
                        new_s.append(e1)
                        new_s.append(e2)
                        if objective(graph, new_s, self.weights, self.cfg) > cur + eps:
                            S = new_s
                            spent = spent - freed + cost(e1, self.cfg) + cost(e2, self.cfg)
                            improved = True
                            break
                if improved or (time.perf_counter() - start) >= self.time_limit_s:
                    break
        return Solution(
            edges=list(S),
            objective=objective(graph, S, self.weights, self.cfg),
            spent=spent,
            runtime_s=time.perf_counter() - start,
            solver="local_search",
            extra={
                "n_candidates": len(candidates),
                "n_selected": len(S),
                "iterations": iterations,
                "seed_solver": "greedy",
                "seed_objective": seed.objective,
            },
        )
