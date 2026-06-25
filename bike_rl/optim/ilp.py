"""ILP coverage-only oracle using OR-Tools CP-SAT. See OPTIMIZER_SPEC.md §7."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import networkx as nx
from ortools.sat.python import cp_model

from bike_rl.candidates import Candidate
from bike_rl.metrics import _metres_between
from bike_rl.objective import ObjectiveWeights, objective
from bike_rl.optim.budget import cost
from bike_rl.optim.greedy import Solution

if TYPE_CHECKING:
    from bike_rl.config import Config

    _NXGraph = nx.MultiDiGraph[Any, Any, Any]
else:
    _NXGraph = nx.MultiDiGraph


_STATUS_NAMES = {
    cp_model.OPTIMAL: "OPTIMAL",
    cp_model.FEASIBLE: "FEASIBLE",
    cp_model.INFEASIBLE: "INFEASIBLE",
    cp_model.MODEL_INVALID: "MODEL_INVALID",
    cp_model.UNKNOWN: "UNKNOWN",
}

# CP-SAT linear coefficients must be integers. Scale float costs/budget by
# this and round; spent is recomputed from Candidate costs (float) for output.
_COST_SCALE = 1000.0


def _reachable_count(graph: _NXGraph, chosen: list[Candidate], cfg: Config) -> int:
    """Number of graph nodes within cfg.coverage_radius_m of any bike-lane endpoint.

    Mirrors the radius branch of bike_rl.metrics.coverage; uses the SAME
    _metres_between so it cannot drift from coverage().
    """
    bike: set[int | str] = {
        n for u, v, d in graph.edges(data=True) if d.get("bike_lane") == "yes" for n in (u, v)
    }
    for e in chosen:
        bike.add(e.u)
        bike.add(e.v)
    covered = 0
    for _n, ndata in graph.nodes(data=True):
        for bn in bike:
            if _metres_between(ndata, graph.nodes[bn]) <= cfg.coverage_radius_m:
                covered += 1
                break
    return covered


class ILPSolver:
    """Coverage-only ILP oracle (OPTIMIZER_SPEC §7).

    Solves the budgeted maximum-coverage problem: choose ``S ⊆ candidates``
    maximising the number of bike-lane-reachable graph nodes (the
    ``coverage_ratio`` of ``bike_rl.metrics.coverage`` in the radius branch)
    subject to ``Σ cost(e) ≤ budget``. This is the exact linearisable oracle
    the spec calls "maximise bike-lane-reachable population under budget".

    Reported ``Solution.objective`` is the shared canonical
    :func:`bike_rl.objective.objective` of the chosen ``S`` (OPTIMIZER_SPEC
    §8), so the ILP row is directly comparable to greedy / local-search / RL
    rows. The optimality *guarantee*, however, is over the **coverage_ratio
    surrogate** (``Solution.extra["surrogate"]`` / ``surrogate_count``) — not
    over the full objective. To use the ILP as an optimality ceiling in the
    §10 table, run it AND the heuristic solvers with coverage-only weights
    ``ObjectiveWeights(connectivity=0.0, coverage=1.0, fragmentation=0.0)``
    and compare the reachable-node count; report the optimality gap against
    the ILP.

    Deterministic — no seed. Only ``coverage_mode == "radius"`` is supported
    here (the component branch and the full-objective ILP with connectivity
    via flow/Steiner vars are deferred to a follow-up plan per §7).

    Engine: OR-Tools CP-SAT only. ``solver="ortools"`` (or ``None``) selects
    it; any other name raises ``ValueError``. PuLP/CBC and gurobipy fallbacks
    are deferred.
    """

    def __init__(
        self,
        cfg: Config,
        weights: ObjectiveWeights,
        solver: str | None = None,
        time_limit_s: float | None = None,
    ) -> None:
        engine = solver if solver is not None else cfg.ilp_default_solver
        if engine != "ortools":
            raise ValueError(
                f"ILPSolver: unsupported engine {engine!r}; only 'ortools' "
                "is implemented (PuLP/gurobipy deferred)."
            )
        self.cfg = cfg
        self.weights = weights
        self.engine = engine
        self.time_limit_s = cfg.ilp_time_limit_s if time_limit_s is None else time_limit_s

    def solve(
        self,
        graph: _NXGraph,
        candidates: list[Candidate],
        budget: float,
    ) -> Solution:
        """Solve the coverage-only ILP and return a :class:`Solution`.

        Args:
            graph: The base bike network graph (not mutated).
            candidates: Candidate edges to consider (input is not mutated).
            budget: Total budget; never exceeded.

        Returns:
            A ``Solution`` with ``solver="ilp"`` and ``extra`` carrying
            ``status`` (OPTIMAL/FEASIBLE/INFEASIBLE/UNKNOWN), ``objective_kind``
            (``"coverage_only"``), ``surrogate`` (coverage_ratio of chosen S),
            ``surrogate_count`` (integer reachable-node count), ``n_candidates``,
            ``n_selected``, ``solver_engine``, ``coverage_mode``, and
            ``opt_gap`` (0.0 when status is OPTIMAL).
        """
        start = time.perf_counter()

        if self.cfg.coverage_mode != "radius":
            raise NotImplementedError(
                f"ILPSolver: coverage_mode={self.cfg.coverage_mode!r} is not "
                "supported by the coverage-only ILP. Only 'radius' mode is "
                "implemented. The full-objective ILP with connectivity via "
                "flow/Steiner auxiliary variables (and the 'component' "
                "coverage_mode linearisation) is deferred to a follow-up plan "
                "(OPTIMIZER_SPEC §7)."
            )

        total = graph.number_of_nodes()
        base_reachable = _reachable_count(graph, [], self.cfg)

        if total == 0 or not candidates or budget <= 0:
            surrogate = base_reachable / total if total else 0.0
            return Solution(
                edges=[],
                objective=objective(graph, [], self.weights, self.cfg),
                spent=0.0,
                runtime_s=time.perf_counter() - start,
                solver="ilp",
                extra={
                    "status": "OPTIMAL",
                    "objective_kind": "coverage_only",
                    "surrogate": surrogate,
                    "surrogate_count": base_reachable,
                    "n_candidates": len(candidates),
                    "n_selected": 0,
                    "solver_engine": self.engine,
                    "coverage_mode": self.cfg.coverage_mode,
                    "opt_gap": 0.0,
                },
            )

        # --- Precompute coverage relation ---
        base_bike: set[int | str] = {
            n for u, v, d in graph.edges(data=True) if d.get("bike_lane") == "yes" for n in (u, v)
        }

        node_list = list(graph.nodes())

        # free[n] = already covered by base bike-lane edges
        free: list[bool] = [False] * len(node_list)
        # near[n] = list of candidate indices that can cover node n
        near: list[list[int]] = [[] for _ in range(len(node_list))]

        for idx_n, n in enumerate(node_list):
            ndata = graph.nodes[n]
            # Check if already covered by base bike-lane endpoints
            for bn in base_bike:
                if _metres_between(ndata, graph.nodes[bn]) <= self.cfg.coverage_radius_m:
                    free[idx_n] = True
                    break
            # Check which candidates could cover this node
            for j, c in enumerate(candidates):
                if (
                    _metres_between(ndata, graph.nodes[c.u]) <= self.cfg.coverage_radius_m
                    or _metres_between(ndata, graph.nodes[c.v]) <= self.cfg.coverage_radius_m
                ):
                    near[idx_n].append(j)

        # --- Build CP-SAT model ---
        model = cp_model.CpModel()

        x = [model.NewBoolVar(f"x{j}") for j in range(len(candidates))]  # type: ignore[attr-defined]
        y: list[cp_model.IntVar] = []
        for idx_n in range(len(node_list)):
            y.append(model.NewBoolVar(f"y{idx_n}"))  # type: ignore[attr-defined]

        # Coverage constraints
        for idx_n in range(len(node_list)):
            if free[idx_n]:
                model.Add(y[idx_n] == 1)  # type: ignore[attr-defined]
            else:
                if near[idx_n]:
                    model.Add(y[idx_n] <= sum(x[j] for j in near[idx_n]))  # type: ignore[attr-defined]
                else:
                    model.Add(y[idx_n] == 0)  # type: ignore[attr-defined]

        # Budget constraint (integer-scaled)
        costs_int = [int(round(cost(c, self.cfg) * _COST_SCALE)) for c in candidates]
        budget_int = int(round(budget * _COST_SCALE))
        model.Add(sum(costs_int[j] * x[j] for j in range(len(candidates))) <= budget_int)  # type: ignore[attr-defined]

        # Lex-maximise: (reachable count) >> (lower cost) >> (fewer edges).
        # The 100_000 multiplier on reachable count ensures the secondary
        # terms (costs_int[j]//100 and 1 per edge) never flip the primary
        # count (max nodes ≤ ~1e6 for any graph the ILP is ever called on).
        model.Maximize(  # type: ignore[attr-defined]
            sum(y) * 100_000
            - sum(costs_int[j] // 100 * x[j] for j in range(len(candidates)))
            - sum(x)
        )

        # --- Solve ---
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit_s
        solver.parameters.num_search_workers = 1
        try:
            status_code = solver.Solve(model)
        except RuntimeError as exc:
            raise RuntimeError(f"ILPSolver: OR-Tools failed: {exc}") from exc

        status = _STATUS_NAMES.get(status_code, "UNKNOWN")

        # --- Read results ---
        chosen_idx = [j for j in range(len(candidates)) if solver.Value(x[j]) == 1]
        chosen = [candidates[j] for j in chosen_idx]
        spent = sum(cost(c, self.cfg) for c in chosen)
        surrogate_count = _reachable_count(graph, chosen, self.cfg)
        surrogate = surrogate_count / total if total else 0.0

        if status == "OPTIMAL":
            opt_gap = 0.0
        elif status == "FEASIBLE":
            obj_val = solver.ObjectiveValue()
            if obj_val > 0:
                opt_gap = abs(solver.BestObjectiveBound() - obj_val) / abs(obj_val)
            else:
                opt_gap = 1.0
        elif status == "INFEASIBLE":
            opt_gap = 1.0
        else:  # UNKNOWN / MODEL_INVALID
            opt_gap = float("nan")

        return Solution(
            edges=list(chosen),
            objective=objective(graph, list(chosen), self.weights, self.cfg),
            spent=spent,
            runtime_s=time.perf_counter() - start,
            solver="ilp",
            extra={
                "status": status,
                "objective_kind": "coverage_only",
                "surrogate": surrogate,
                "surrogate_count": surrogate_count,
                "n_candidates": len(candidates),
                "n_selected": len(chosen),
                "solver_engine": self.engine,
                "coverage_mode": self.cfg.coverage_mode,
                "opt_gap": opt_gap,
            },
        )
