"""Tests for the ILP coverage-only oracle (OPTIMIZER_SPEC §7, §9, §12).

Plans 013a, 013b, 013c share this file. 013a defines the fixture, helpers,
and headline tests (brute-force parity, budget respect, OPTIMAL status).
013b and 013c append determinism, time-limit, well-formedness, guard, and
edge-case tests.
"""

from __future__ import annotations

import itertools

import networkx as nx
import pytest

from bike_rl.candidates import Candidate, candidate_cost
from bike_rl.config import Config
from bike_rl.objective import ObjectiveWeights
from bike_rl.optim.ilp import ILPSolver, _reachable_count

# ── Helper ───────────────────────────────────────────────────────────────


def _edge(u: int | str, v: int | str, length: float, road_priority: int = 1) -> Candidate:
    """Build a Candidate satisfying the Edge protocol."""
    return Candidate(
        u=u,
        v=v,
        length=float(length),
        road_priority=road_priority,
        connects_to_bike_path=False,
        data={"length": float(length), "highway": "residential"},
    )


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def default_cfg() -> Config:
    """Default config."""
    return Config()


@pytest.fixture
def default_weights() -> ObjectiveWeights:
    """Default objective weights (0.4 / 0.4 / 0.2)."""
    return ObjectiveWeights()


@pytest.fixture
def cov_only_weights() -> ObjectiveWeights:
    """Coverage-only weights: connectivity=0, fragmentation=0, coverage=1."""
    return ObjectiveWeights(connectivity=0.0, coverage=1.0, fragmentation=0.0)


@pytest.fixture
def max_coverage_instance() -> tuple[nx.MultiDiGraph, list[Candidate]]:
    """Clean budgeted max-coverage fixture (radius does not bleed).

    9 nodes. Base bike-lane edges among {1,2,3} cover only {1,2,3}
    (3/9). Far nodes 10..15 are mutually >300 m apart, so each candidate
    edge covers exactly its two endpoints. Base coverage_ratio = 3/9.

    Candidates (length -> cost = length * edge_cost_factor=10):
      (1,10)  len 100 -> 1000 : adds node 10
      (10,11) len  60 ->  600  : adds 10,11
      (11,12) len  60 ->  600  : adds 11,12
      (12,13) len  60 ->  600  : adds 12,13
      (13,14) len  60 ->  600  : adds 13,14
      (14,15) len  60 ->  600  : adds 14,15
      (2,15)  len 100 -> 1000 : adds node 15
      (1,11)  len 120 -> 1200 : adds node 11

    Brute-force optima (covered-count / 9), VERIFIED during planning:
      budget  600 -> {(10,11)}            covered=5  cr=5/9  spent=600
      budget 1000 -> {(10,11)}            covered=5  cr=5/9  spent=600
      budget 1200 -> any 2 disjoint        covered=7  cr=7/9  spent=1200
      budget 1800 -> {(10,11),(12,13),(14,15)} covered=9 cr=1.0  spent=1800
    """
    coords = {
        1: (0.000, 0.000),
        2: (0.001, 0.000),
        3: (0.0005, 0.001),
        10: (0.010, 0.000),
        11: (0.015, 0.000),
        12: (0.020, 0.000),
        13: (0.025, 0.000),
        14: (0.030, 0.000),
        15: (0.035, 0.000),
    }
    g = nx.MultiDiGraph()
    for n, (x, y) in coords.items():
        g.add_node(n, x=x, y=y)
    g.add_edge(1, 2, length=120.0, highway="residential", bike_lane="yes")
    g.add_edge(2, 3, length=120.0, highway="residential", bike_lane="yes")
    candidates = [
        _edge(1, 10, 100.0, road_priority=5),
        _edge(10, 11, 60.0, road_priority=4),
        _edge(11, 12, 60.0, road_priority=4),
        _edge(12, 13, 60.0, road_priority=4),
        _edge(13, 14, 60.0, road_priority=3),
        _edge(14, 15, 60.0, road_priority=3),
        _edge(2, 15, 100.0, road_priority=5),
        _edge(1, 11, 120.0, road_priority=2),
    ]
    return g, candidates


# ── Brute-force helper (parity reference) ────────────────────────────────


def _brute_best(
    graph: nx.MultiDiGraph,
    candidates: list[Candidate],
    cfg: Config,
    budget: float,
) -> tuple[int, list[Candidate]]:
    """Return (best_reachable_count, best_chosen_set) over all affordable subsets."""
    best = -1
    best_set: list[Candidate] = []
    for r in range(len(candidates) + 1):
        for S in itertools.combinations(candidates, r):
            spent = sum(candidate_cost(c, cfg) for c in S)
            if spent > budget + 1e-9:
                continue
            cnt = _reachable_count(graph, list(S), cfg)
            if cnt > best + 1e-12 or (abs(cnt - best) <= 1e-12 and len(S) < len(best_set)):
                best = cnt
                best_set = list(S)
    return best, best_set


# ── ILPSolver ─────────────────────────────────────────────────────────────


class TestILPSolver:
    """Tests for ILPSolver (OPTIMIZER_SPEC §7, §9, §12)."""

    @pytest.mark.parametrize(
        "budget, expected",
        [(600.0, 5), (1000.0, 5), (1200.0, 7), (1800.0, 9)],
    )
    def test_matches_brute_force_on_tiny_instance(
        self,
        max_coverage_instance: tuple[nx.MultiDiGraph, list[Candidate]],
        default_cfg: Config,
        default_weights: ObjectiveWeights,
        budget: float,
        expected: int,
    ) -> None:
        """ILP reachable count matches brute-force optimum (§9/§12 oracle)."""
        graph, candidates = max_coverage_instance
        sol = ILPSolver(default_cfg, default_weights).solve(graph, candidates, budget)
        best, _ = _brute_best(graph, candidates, default_cfg, budget)
        assert sol.extra["surrogate_count"] == best == expected
        assert sol.spent <= budget + 1e-9

    @pytest.mark.parametrize("budget", [1.0, 600.0, 1200.0, 1e9])
    def test_respects_budget_never_overspends(
        self,
        max_coverage_instance: tuple[nx.MultiDiGraph, list[Candidate]],
        default_cfg: Config,
        default_weights: ObjectiveWeights,
        budget: float,
    ) -> None:
        """ILP never exceeds the budget."""
        graph, candidates = max_coverage_instance
        sol = ILPSolver(default_cfg, default_weights).solve(graph, candidates, budget)
        assert sol.spent <= budget + 1e-9

    def test_optimal_status_and_zero_gap_on_small_instance(
        self,
        max_coverage_instance: tuple[nx.MultiDiGraph, list[Candidate]],
        default_cfg: Config,
        default_weights: ObjectiveWeights,
    ) -> None:
        """ILP returns OPTIMAL status and zero gap on a small instance."""
        graph, candidates = max_coverage_instance
        sol = ILPSolver(default_cfg, default_weights, time_limit_s=10.0).solve(
            graph, candidates, 1800.0
        )
        assert sol.extra["status"] == "OPTIMAL"
        assert sol.extra["opt_gap"] == 0.0
        assert sol.solver == "ilp"
