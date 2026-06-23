"""Tests for the greedy solver and budget utilities (OPTIMIZER_SPEC §5, §9).

Covers: correct-edge selection on a tiny instance, budget respect, termination
on no-affordable-edge, determinism, the Solution contract, and the budget
cost/remaining helpers. See OPTIMIZER_SPEC.md §9 and plans/011-greedy-and-tests.md.
"""

from __future__ import annotations

import networkx as nx
import pytest

from bike_rl.candidates import Candidate, candidate_cost, extract_candidates
from bike_rl.config import Config
from bike_rl.objective import ObjectiveWeights, objective
from bike_rl.optim.budget import cost, remaining_budget
from bike_rl.optim.greedy import GreedySolver, Solution

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


# ── Budget utilities ────────────────────────────────────────────────────


class TestBudget:
    """Tests for optim/budget.py."""

    def test_cost_delegates_to_candidate_cost(self, default_cfg: Config) -> None:
        """cost(edge, cfg) == candidate_cost(edge, cfg)."""
        e = _edge(1, 2, 200.0)
        assert cost(e, default_cfg) == candidate_cost(e, default_cfg)
        assert cost(e, default_cfg) == 200.0 * default_cfg.edge_cost_factor

    def test_remaining_budget_non_negative(self) -> None:
        """remaining_budget is max(budget - spent, 0.0)."""
        assert remaining_budget(0.0, 1000.0) == 1000.0
        assert remaining_budget(300.0, 1000.0) == 700.0
        assert remaining_budget(1000.0, 1000.0) == 0.0
        assert remaining_budget(1500.0, 1000.0) == 0.0  # clamped, not negative


# ── GreedySolver ────────────────────────────────────────────────────────


class TestGreedySolver:
    """Tests for GreedySolver (OPTIMIZER_SPEC §5, §9)."""

    def test_picks_obviously_correct_edge_first(
        self,
        default_cfg: Config,
        default_weights: ObjectiveWeights,
    ) -> None:
        """On a tight budget, greedy picks the best gain-per-cost edge.

        With a budget that affords only the cheapest candidate, greedy must
        select exactly that one and stop. The chosen edge must improve the
        shared objective over the empty selection (the performance floor).
        """
        # Build a graph with node 5 pre-populated so adding (2,5) improves
        # coverage (node 5 needs x,y for the coverage metric).
        bike_graph = nx.MultiDiGraph()
        for n, (x, y) in {
            1: (0.0, 0.0),
            2: (0.001, 0.0),
            3: (0.0, 0.001),
            4: (0.001, 0.001),
            5: (0.002, 0.002),
        }.items():
            bike_graph.add_node(n, x=x, y=y)
        bike_graph.add_edge(1, 2, length=120.0, highway="residential", bike_lane="yes")
        bike_graph.add_edge(3, 4, length=130.0, highway="residential", bike_lane="yes")

        candidates = [
            _edge(2, 5, 200.0, road_priority=5),  # cost=2000
            _edge(3, 5, 150.0, road_priority=4),  # cost=1500
            _edge(4, 5, 600.0, road_priority=3),  # cost=6000
        ]

        # Budget affords only the cheapest candidate (150m * factor=10 = 1500).
        budget = 1500.0
        solver = GreedySolver(default_cfg, default_weights)
        sol = solver.solve(bike_graph, candidates, budget)

        assert len(sol.edges) == 1
        assert sol.spent == pytest.approx(1500.0)
        # The selected edge actually improves the objective vs empty selection.
        base_obj = objective(bike_graph, [], default_weights, default_cfg)
        assert sol.objective > base_obj + 1e-9

    def test_respects_budget_never_overspends(
        self,
        tiny_bike_graph: nx.MultiDiGraph,
        tiny_walk_graph: nx.MultiDiGraph,
        default_cfg: Config,
        default_weights: ObjectiveWeights,
    ) -> None:
        """Spent <= budget for a range of budgets, including very tight ones."""
        candidates = extract_candidates(tiny_bike_graph, tiny_walk_graph, default_cfg)
        solver = GreedySolver(default_cfg, default_weights)
        for budget in (1.0, 1500.0, 5000.0, 1e9):
            sol = solver.solve(tiny_bike_graph, candidates, budget)
            assert sol.spent <= budget + 1e-9, f"overspent budget={budget}: spent={sol.spent}"
            # Every selected edge is individually within budget.
            for e in sol.edges:
                assert candidate_cost(e, default_cfg) <= budget + 1e-9

    def test_terminates_when_no_affordable_edge(
        self,
        tiny_bike_graph: nx.MultiDiGraph,
        default_cfg: Config,
        default_weights: ObjectiveWeights,
    ) -> None:
        """With budget below every candidate cost, greedy returns empty."""
        candidates = [
            _edge(2, 5, 200.0, road_priority=5),
            _edge(3, 5, 150.0, road_priority=4),
        ]
        # All costs are >= 150 * 10 = 1500; budget 1.0 affords none.
        solver = GreedySolver(default_cfg, default_weights)
        sol = solver.solve(tiny_bike_graph, candidates, 1.0)
        assert sol.edges == []
        assert sol.spent == 0.0
        assert sol.objective == pytest.approx(
            objective(tiny_bike_graph, [], default_weights, default_cfg)
        )

    def test_terminates_when_no_improving_edge(
        self,
        tiny_bike_graph: nx.MultiDiGraph,
        default_cfg: Config,
        default_weights: ObjectiveWeights,
    ) -> None:
        """A candidate whose addition does not improve the objective is skipped.

        A duplicate of an existing bike-lane edge has delta ~ 0 (plan 010's
        duplicate-edge guarantee), so greedy must not select it even with a
        huge budget.
        """
        # (1,2) is already a bike_lane edge in tiny_bike_graph.
        dup = _edge(1, 2, 120.0, road_priority=2)
        solver = GreedySolver(default_cfg, default_weights)
        sol = solver.solve(tiny_bike_graph, [dup], 1e9)
        assert sol.edges == []
        assert sol.spent == 0.0

    def test_deterministic_across_runs(
        self,
        tiny_bike_graph: nx.MultiDiGraph,
        tiny_walk_graph: nx.MultiDiGraph,
        default_cfg: Config,
        default_weights: ObjectiveWeights,
    ) -> None:
        """Two solves produce identical edge sets, objective, and spent."""
        candidates = extract_candidates(tiny_bike_graph, tiny_walk_graph, default_cfg)
        solver = GreedySolver(default_cfg, default_weights)
        sol1 = solver.solve(tiny_bike_graph, candidates, 1e9)
        sol2 = solver.solve(tiny_bike_graph, candidates, 1e9)
        assert sol1.edges == sol2.edges
        assert sol1.objective == pytest.approx(sol2.objective)
        assert sol1.spent == pytest.approx(sol2.spent)

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
        solver = GreedySolver(default_cfg, default_weights)
        solver.solve(tiny_bike_graph, candidates, 1e9)
        assert candidates == snapshot

    def test_solution_record_is_well_formed(
        self,
        tiny_bike_graph: nx.MultiDiGraph,
        tiny_walk_graph: nx.MultiDiGraph,
        default_cfg: Config,
        default_weights: ObjectiveWeights,
    ) -> None:
        """Solution has solver='greedy', non-negative runtime, and counts."""
        candidates = extract_candidates(tiny_bike_graph, tiny_walk_graph, default_cfg)
        sol = GreedySolver(default_cfg, default_weights).solve(tiny_bike_graph, candidates, 1e9)
        assert isinstance(sol, Solution)
        assert sol.solver == "greedy"
        assert sol.runtime_s >= 0.0
        assert sol.extra["n_candidates"] == 4
        assert sol.extra["n_selected"] == len(sol.edges)
        # objective field equals a fresh objective() recompute on the edges.
        assert sol.objective == pytest.approx(
            objective(tiny_bike_graph, sol.edges, default_weights, default_cfg)
        )

    def test_beats_random_selection_floor(
        self,
        tiny_bike_graph: nx.MultiDiGraph,
        tiny_walk_graph: nx.MultiDiGraph,
        default_cfg: Config,
        default_weights: ObjectiveWeights,
    ) -> None:
        """Greedy objective >= objective of any single-edge selection (§12 DoD).

        The greedy result must be at least as good as the best single-edge
        pick — the minimum bar. (On this fixture greedy selects >= 1 edge and
        its objective dominates every one-edge selection.)
        """
        candidates = extract_candidates(tiny_bike_graph, tiny_walk_graph, default_cfg)
        solver = GreedySolver(default_cfg, default_weights)
        sol = solver.solve(tiny_bike_graph, candidates, 1e9)
        best_single = max(
            objective(tiny_bike_graph, [e], default_weights, default_cfg) for e in candidates
        )
        assert sol.objective >= best_single - 1e-9
