"""Tests for the local search solver (OPTIMIZER_SPEC §6, §9).

Covers: never-worse-than-greedy-seed (§9 invariant), 2-opt escape, iteration
and time caps, determinism, no input mutation, well-formed Solution record,
and empty-seed path.
"""

from __future__ import annotations

import networkx as nx
import pytest

from bike_rl.candidates import Candidate, extract_candidates
from bike_rl.config import Config
from bike_rl.objective import ObjectiveWeights, objective
from bike_rl.optim.greedy import GreedySolver
from bike_rl.optim.local_search import LocalSearchSolver

# ── Helper ───────────────────────────────────────────────────────────────


def _edge(u: int | str, v: int | str, length: float, road_priority: int = 1) -> Candidate:
    """Build a Candidate satisfying the Edge protocol."""
    return Candidate(
        u=u,
        v=v,
        length=float(length),
        road_priority=road_priority,
        connects_to_bike_path=False,
        data={"length": float(length)},
    )


# ── Local fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def default_weights() -> ObjectiveWeights:
    """Default objective weights (0.4 / 0.4 / 0.2)."""
    return ObjectiveWeights()


@pytest.fixture
def default_cfg() -> Config:
    """Default config."""
    return Config()


@pytest.fixture
def one_opt_instance() -> tuple[nx.MultiDiGraph, list[Candidate], float]:
    """Fixture where greedy is suboptimal and a 1-opt swap improves it.

    Two separate components each with existing bike edges:
      - Component A: nodes 1-2 (bike edge)
      - Component B: nodes 3-4 (bike edge)
    Candidates:
      - (5,6) [cost 500, rp=1] — connects two new nodes to each other
      - (2,5) [cost 1000, rp=3] — connects new node 5 to component A
      - (4,6) [cost 1000, rp=2] — connects new node 6 to component B

    At budget=1000, greedy picks (5,6) [best delta/c=0.000148, obj=0.255]
    because the remaining 500 cannot afford another candidate. 1-opt drops
    (5,6) and picks (2,5) [obj=0.302], which has lower delta/c but higher
    absolute objective.
    """
    g = nx.MultiDiGraph()
    for n, (x, y) in {
        1: (0.0, 0.0),
        2: (0.001, 0.0),
        3: (0.01, 0.01),
        4: (0.011, 0.01),
        5: (0.005, 0.005),
        6: (0.006, 0.006),
    }.items():
        g.add_node(n, x=x, y=y)
    g.add_edge(1, 2, length=100.0, highway="residential", bike_lane="yes")
    g.add_edge(3, 4, length=100.0, highway="residential", bike_lane="yes")
    candidates = [
        _edge(5, 6, 50.0, road_priority=1),  # cost 500  — best delta/c
        _edge(2, 5, 100.0, road_priority=3),  # cost 1000 — highest single-edge obj
        _edge(4, 6, 100.0, road_priority=2),  # cost 1000
    ]
    return g, candidates, 1000.0


@pytest.fixture
def two_opt_instance() -> tuple[nx.MultiDiGraph, list[Candidate], float]:
    """Fixture where greedy is suboptimal and a 2-opt move improves it.

    Greedy picks only (3,2) [cost 2500, obj 0.302] because the remaining 500
    budget cannot afford any other candidate (cheapest is 1000). 2-opt drops
    (3,2) and adds (5,1)+(6,5) [combined cost 2500 <= 3000, obj 0.655333].
    No 1-opt move improves the seed, so this is a pure 2-opt escape.
    """
    g = nx.MultiDiGraph()
    coords = {
        1: (0.0322, 0.0352),
        2: (0.027, 0.0491),
        3: (0.0423, 0.0491),
        4: (0.0152, 0.0303),
        5: (0.0326, 0.035),
        6: (0.001, 0.0487),
    }
    for n, (x, y) in coords.items():
        g.add_node(n, x=x, y=y)
    g.add_edge(4, 2, length=120.0, highway="residential", bike_lane="yes")
    g.add_edge(1, 6, length=130.0, highway="residential", bike_lane="yes")
    candidates = [
        _edge(5, 1, 150.0, road_priority=5),  # cost 1500
        _edge(6, 5, 100.0, road_priority=1),  # cost 1000
        _edge(3, 2, 250.0, road_priority=2),  # cost 2500  <- greedy's only pick
        _edge(5, 3, 300.0, road_priority=3),  # cost 3000
        _edge(4, 5, 100.0, road_priority=5),  # cost 1000
    ]
    return g, candidates, 3000.0


# ── LocalSearchSolver ────────────────────────────────────────────────────


class TestLocalSearchSolver:
    """Tests for LocalSearchSolver (OPTIMIZER_SPEC §6, §9)."""

    def test_never_worse_than_greedy_seed(
        self,
        tiny_bike_graph: nx.MultiDiGraph,
        tiny_walk_graph: nx.MultiDiGraph,
        default_cfg: Config,
        default_weights: ObjectiveWeights,
    ) -> None:
        """Local search objective >= greedy objective for any budget (§9 invariant)."""
        candidates = extract_candidates(tiny_bike_graph, tiny_walk_graph, default_cfg)
        greedy_solver = GreedySolver(default_cfg, default_weights)
        ls_solver = LocalSearchSolver(default_cfg, default_weights)
        for budget in (1.0, 1500.0, 3000.0, 1e9):
            greedy_sol = greedy_solver.solve(tiny_bike_graph, candidates, budget)
            ls_sol = ls_solver.solve(tiny_bike_graph, candidates, budget)
            assert ls_sol.objective >= greedy_sol.objective - 1e-9, (
                f"LS objective {ls_sol.objective} < greedy {greedy_sol.objective} "
                f"at budget={budget}"
            )
            assert ls_sol.spent <= budget + 1e-9, (
                f"LS overspent budget={budget}: spent={ls_sol.spent}"
            )

    def test_1opt_improves_on_greedy(
        self,
        one_opt_instance: tuple[nx.MultiDiGraph, list[Candidate], float],
        default_cfg: Config,
        default_weights: ObjectiveWeights,
    ) -> None:
        """1-opt escape fixture: local search strictly improves on greedy."""
        graph, candidates, budget = one_opt_instance
        greedy_sol = GreedySolver(default_cfg, default_weights).solve(graph, candidates, budget)
        ls_sol = LocalSearchSolver(default_cfg, default_weights).solve(graph, candidates, budget)
        assert ls_sol.objective > greedy_sol.objective + 1e-9, (
            f"LS obj {ls_sol.objective} not > greedy obj {greedy_sol.objective}"
        )
        assert ls_sol.extra["iterations"] >= 1

    def test_2opt_escape_improves_on_greedy(
        self,
        two_opt_instance: tuple[nx.MultiDiGraph, list[Candidate], float],
        default_cfg: Config,
        default_weights: ObjectiveWeights,
    ) -> None:
        """2-opt escape fixture: local search strictly improves on greedy."""
        graph, candidates, budget = two_opt_instance
        greedy_sol = GreedySolver(default_cfg, default_weights).solve(graph, candidates, budget)
        ls_sol = LocalSearchSolver(default_cfg, default_weights).solve(graph, candidates, budget)
        assert ls_sol.objective > greedy_sol.objective + 1e-9, (
            f"LS obj {ls_sol.objective} not > greedy obj {greedy_sol.objective}"
        )
        assert ls_sol.extra["iterations"] >= 1

    def test_respects_max_iter_zero(
        self,
        two_opt_instance: tuple[nx.MultiDiGraph, list[Candidate], float],
        default_cfg: Config,
        default_weights: ObjectiveWeights,
    ) -> None:
        """max_iter=0 returns the greedy seed unchanged."""
        graph, candidates, budget = two_opt_instance
        greedy_sol = GreedySolver(default_cfg, default_weights).solve(graph, candidates, budget)
        ls_sol = LocalSearchSolver(default_cfg, default_weights, max_iter=0).solve(
            graph, candidates, budget
        )
        assert ls_sol.edges == greedy_sol.edges
        assert ls_sol.objective == pytest.approx(greedy_sol.objective)
        assert ls_sol.extra["iterations"] == 0

    def test_respects_max_iter_one(
        self,
        two_opt_instance: tuple[nx.MultiDiGraph, list[Candidate], float],
        default_cfg: Config,
        default_weights: ObjectiveWeights,
    ) -> None:
        """max_iter=1 runs exactly one improving pass (the 2-opt move)."""
        graph, candidates, budget = two_opt_instance
        ls_sol = LocalSearchSolver(default_cfg, default_weights, max_iter=1).solve(
            graph, candidates, budget
        )
        assert ls_sol.extra["iterations"] == 1
        assert ls_sol.objective == pytest.approx(0.655333, rel=1e-4)

    def test_respects_time_limit_zero(
        self,
        two_opt_instance: tuple[nx.MultiDiGraph, list[Candidate], float],
        default_cfg: Config,
        default_weights: ObjectiveWeights,
    ) -> None:
        """time_limit_s=0 returns the greedy seed (no passes run)."""
        graph, candidates, budget = two_opt_instance
        greedy_sol = GreedySolver(default_cfg, default_weights).solve(graph, candidates, budget)
        ls_sol = LocalSearchSolver(default_cfg, default_weights, time_limit_s=0.0).solve(
            graph, candidates, budget
        )
        assert ls_sol.extra["iterations"] == 0
        assert ls_sol.edges == greedy_sol.edges

    def test_deterministic_across_runs(
        self,
        tiny_bike_graph: nx.MultiDiGraph,
        tiny_walk_graph: nx.MultiDiGraph,
        default_cfg: Config,
        default_weights: ObjectiveWeights,
    ) -> None:
        """Two solves produce identical edges, objective, spent, and iterations."""
        candidates = extract_candidates(tiny_bike_graph, tiny_walk_graph, default_cfg)
        solver = LocalSearchSolver(default_cfg, default_weights)
        sol1 = solver.solve(tiny_bike_graph, candidates, 1e9)
        sol2 = solver.solve(tiny_bike_graph, candidates, 1e9)
        assert sol1.edges == sol2.edges
        assert sol1.objective == pytest.approx(sol2.objective)
        assert sol1.spent == pytest.approx(sol2.spent)
        assert sol1.extra["iterations"] == sol2.extra["iterations"]

    def test_does_not_mutate_input_candidates(
        self,
        tiny_bike_graph: nx.MultiDiGraph,
        tiny_walk_graph: nx.MultiDiGraph,
        default_cfg: Config,
        default_weights: ObjectiveWeights,
    ) -> None:
        """The candidates list passed in is unchanged after solve."""
        candidates = extract_candidates(tiny_bike_graph, tiny_walk_graph, default_cfg)
        snapshot = list(candidates)
        solver = LocalSearchSolver(default_cfg, default_weights)
        solver.solve(tiny_bike_graph, candidates, 1e9)
        assert candidates == snapshot

    def test_solution_record_is_well_formed(
        self,
        tiny_bike_graph: nx.MultiDiGraph,
        tiny_walk_graph: nx.MultiDiGraph,
        default_cfg: Config,
        default_weights: ObjectiveWeights,
    ) -> None:
        """Solution has correct solver name, runtime, and extra metadata."""
        candidates = extract_candidates(tiny_bike_graph, tiny_walk_graph, default_cfg)
        ls_sol = LocalSearchSolver(default_cfg, default_weights).solve(
            tiny_bike_graph, candidates, 1e9
        )
        assert ls_sol.solver == "local_search"
        assert ls_sol.runtime_s >= 0.0
        assert ls_sol.extra["n_candidates"] == 4
        assert ls_sol.extra["n_selected"] == len(ls_sol.edges)
        assert isinstance(ls_sol.extra["iterations"], int)
        assert ls_sol.extra["iterations"] >= 0
        assert ls_sol.extra["seed_solver"] == "greedy"
        assert ls_sol.objective == pytest.approx(
            objective(tiny_bike_graph, ls_sol.edges, default_weights, default_cfg)
        )

    def test_empty_seed_when_no_affordable_edge(
        self,
        tiny_bike_graph: nx.MultiDiGraph,
        default_cfg: Config,
        default_weights: ObjectiveWeights,
    ) -> None:
        """Budget below every candidate cost: local search returns empty, no iteration."""
        # Two costly candidates; budget 1.0 affords neither.
        candidates = [
            _edge(2, 1, 200.0, road_priority=5),  # cost = 2000
            _edge(2, 1, 150.0, road_priority=4),  # cost = 1500
        ]
        solver = LocalSearchSolver(default_cfg, default_weights)
        sol = solver.solve(tiny_bike_graph, candidates, 1.0)
        assert sol.edges == []
        assert sol.spent == 0.0
        assert sol.extra["iterations"] == 1  # one pass entered, no moves possible
        assert sol.objective == pytest.approx(
            objective(tiny_bike_graph, [], default_weights, default_cfg)
        )
