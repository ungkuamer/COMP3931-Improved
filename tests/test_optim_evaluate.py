"""Tests for the shared optimiser/RL evaluation harness (OPTIMIZER_SPEC §8/§9).

Covers: §8 comparability (every solver scored into a ``Solution`` by the
shared ``objective``), ``evaluate_solver``/``evaluate_rl_policy`` producing
comparable records, ``run_comparison`` running the full roster, and the §10
reporting table.
"""

from __future__ import annotations

import networkx as nx
import pytest

from bike_rl.candidates import Candidate
from bike_rl.config import Config
from bike_rl.objective import ObjectiveWeights, objective
from bike_rl.optim.evaluate import (
    Instance,
    build_instance_from_graphs,
    default_solvers,
    evaluate_rl_policy,
    evaluate_solver,
    format_comparison_table,
    run_comparison,
)
from bike_rl.optim.greedy import GreedySolver, Solution
from bike_rl.optim.ilp import ILPSolver

# ── Helpers ───────────────────────────────────────────────────────────────


def _edge(u: int, v: int, length: float, road_priority: int = 1) -> Candidate:
    """Build a Candidate satisfying the Edge protocol."""
    return Candidate(
        u=u,
        v=v,
        length=float(length),
        road_priority=road_priority,
        connects_to_bike_path=False,
        data={"length": float(length), "highway": "residential"},
    )


class _FixedSolver:
    """A stub solver that returns a prescribed edge set (for §8 parity tests)."""

    def __init__(self, edges: list[Candidate], solver: str = "fixed") -> None:
        self._edges = edges
        self._solver = solver

    def solve(
        self,
        graph: nx.MultiDiGraph,
        candidates: list[Candidate],
        budget: float,
    ) -> Solution:
        spent = sum(e.length for e in self._edges) * 10.0  # edge_cost_factor default
        return Solution(
            edges=list(self._edges),
            objective=objective(graph, self._edges, ObjectiveWeights(), Config()),
            spent=spent,
            runtime_s=0.0,
            solver=self._solver,
            extra={"n_candidates": len(candidates), "n_selected": len(self._edges)},
        )


class _FirstActionPolicy:
    """A fake RL policy: always pick the first legal (masked-True) action.

    Mimics the ``MaskablePPO.predict`` contract used by
    :func:`evaluate_rl_policy`: returns ``(action_int, None)``.
    """

    def predict(self, obs, deterministic: bool = False, action_masks=None):  # type: ignore[no-untyped-def]
        import numpy as np

        mask = np.asarray(action_masks)
        legal = np.nonzero(mask)[0]
        action = int(legal[0]) if legal.size else 0
        return action, None


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def default_cfg() -> Config:
    """Default Config."""
    return Config()


@pytest.fixture
def default_weights() -> ObjectiveWeights:
    """Default objective weights (0.4 / 0.4 / 0.2)."""
    return ObjectiveWeights()


@pytest.fixture
def small_instance(default_cfg: Config, default_weights: ObjectiveWeights) -> Instance:
    """A 6-node instance with 4 candidates (bike + walk graphs paired).

    Base bike-lane edge (1,2). Candidates (2,3),(3,4),(4,5),(5,6) all
    affordable at the given budget. ``walk_graph`` is set so RL rollouts work.
    """
    coords = {
        1: (0.000, 0.000),
        2: (0.001, 0.000),
        3: (0.002, 0.000),
        4: (0.003, 0.000),
        5: (0.004, 0.000),
        6: (0.005, 0.000),
    }
    bike = nx.MultiDiGraph()
    for n, (x, y) in coords.items():
        bike.add_node(n, x=x, y=y)
    bike.add_edge(1, 2, length=120.0, highway="residential", bike_lane="yes")

    walk = nx.MultiDiGraph()
    for n, (x, y) in coords.items():
        walk.add_node(n, x=x, y=y)
    walk.add_edge(1, 2, length=120.0, highway="residential")  # already bike → skipped
    walk.add_edge(2, 3, length=150.0, highway="secondary")
    walk.add_edge(3, 4, length=150.0, highway="residential")
    walk.add_edge(4, 5, length=150.0, highway="residential")
    walk.add_edge(5, 6, length=150.0, highway="residential")

    candidates = [
        _edge(2, 3, 150.0, road_priority=4),
        _edge(3, 4, 150.0, road_priority=2),
        _edge(4, 5, 150.0, road_priority=2),
        _edge(5, 6, 150.0, road_priority=2),
    ]
    return Instance(
        graph=bike,
        candidates=candidates,
        budget=100_000.0,
        cfg=default_cfg,
        weights=default_weights,
        walk_graph=walk,
        label="small_test",
    )


# ── evaluate_solver ───────────────────────────────────────────────────────


class TestEvaluateSolver:
    """§8: every solver is scored into a comparable ``Solution``."""

    def test_greedy_rescored_by_shared_objective(self, small_instance: Instance) -> None:
        """evaluate_solver re-scores greedy's edges with the shared objective."""
        sol = evaluate_solver(
            GreedySolver(small_instance.cfg, small_instance.weights), small_instance
        )
        assert sol.solver == "greedy"
        expected = objective(
            small_instance.graph, sol.edges, small_instance.weights, small_instance.cfg
        )
        assert sol.objective == pytest.approx(expected)
        assert sol.extra["n_edges"] == len(sol.edges)

    def test_ilp_rescored_by_shared_objective(self, small_instance: Instance) -> None:
        """evaluate_solver re-scores the ILP's edges with the shared objective."""
        sol = evaluate_solver(ILPSolver(small_instance.cfg, small_instance.weights), small_instance)
        assert sol.solver == "ilp"
        expected = objective(
            small_instance.graph, sol.edges, small_instance.weights, small_instance.cfg
        )
        assert sol.objective == pytest.approx(expected)

    def test_same_objective_for_identical_edge_sets_regardless_of_solver(
        self, small_instance: Instance
    ) -> None:
        """§8: identical edge sets → identical objective regardless of solver."""
        edges = small_instance.candidates[:2]
        sol_a = evaluate_solver(_FixedSolver(edges, solver="A"), small_instance)
        sol_b = evaluate_solver(_FixedSolver(edges, solver="B"), small_instance)
        assert sol_a.solver == "A"
        assert sol_b.solver == "B"
        assert sol_a.objective == pytest.approx(sol_b.objective)
        # And both equal a direct objective call (the canonical scorer).
        direct = objective(small_instance.graph, edges, small_instance.weights, small_instance.cfg)
        assert sol_a.objective == pytest.approx(direct)

    def test_respects_budget(self, small_instance: Instance) -> None:
        """Solutions never exceed the instance budget."""
        small_instance.budget = 2000.0  # only one ~1500-cost edge fits
        for solver in default_solvers(small_instance):
            sol = evaluate_solver(solver, small_instance)
            assert sol.spent <= small_instance.budget + 1e-6


# ── evaluate_rl_policy ────────────────────────────────────────────────────


class TestEvaluateRLPolicy:
    """§8: the RL policy is scored by the shared objective, not the reward."""

    def test_returns_comparable_solution(self, small_instance: Instance) -> None:
        """evaluate_rl_policy returns a Solution scored by the shared objective."""
        sol = evaluate_rl_policy(_FirstActionPolicy(), small_instance)
        assert sol.solver == "rl"
        # Rollout picks every affordable candidate (budget huge) → all 4 added.
        assert len(sol.edges) == 4
        expected = objective(
            small_instance.graph, sol.edges, small_instance.weights, small_instance.cfg
        )
        assert sol.objective == pytest.approx(expected)
        assert sol.extra["n_edges"] == 4
        assert sol.extra["episode_reward"] is not None

    def test_objective_comparable_to_greedy_same_edges(self, small_instance: Instance) -> None:
        """RL row and a fixed-solver row with the same edges score identically."""
        sol_rl = evaluate_rl_policy(_FirstActionPolicy(), small_instance)
        sol_fixed = evaluate_solver(
            _FixedSolver(list(sol_rl.edges), solver="fixed"), small_instance
        )
        assert sol_rl.objective == pytest.approx(sol_fixed.objective)

    def test_raises_when_walk_graph_missing(self, small_instance: Instance) -> None:
        """evaluate_rl_policy requires instance.walk_graph."""
        small_instance.walk_graph = None
        with pytest.raises(ValueError, match="walk_graph"):
            evaluate_rl_policy(_FirstActionPolicy(), small_instance)


# ── run_comparison + table ────────────────────────────────────────────────


class TestRunComparison:
    """§10: the full roster runs and renders the reporting table."""

    def test_default_roster_runs_all_three_solvers(self, small_instance: Instance) -> None:
        """run_comparison returns one Solution per default solver."""
        sols = run_comparison(small_instance)
        assert [s.solver for s in sols] == ["greedy", "local_search", "ilp"]
        for s in sols:
            expected = objective(
                small_instance.graph, s.edges, small_instance.weights, small_instance.cfg
            )
            assert s.objective == pytest.approx(expected)

    def test_roster_with_rl_appends_rl_row(self, small_instance: Instance) -> None:
        """run_comparison appends an 'rl' row when a policy is given."""
        sols = run_comparison(small_instance, rl_policy=_FirstActionPolicy())
        assert [s.solver for s in sols] == ["greedy", "local_search", "ilp", "rl"]

    def test_local_search_never_worse_than_greedy(self, small_instance: Instance) -> None:
        """Sanity: local search ≥ greedy (it seeds from greedy, strict improvements)."""
        sols = run_comparison(small_instance)
        by = {s.solver: s.objective for s in sols}
        assert by["local_search"] >= by["greedy"] - 1e-9

    def test_format_comparison_table_has_one_row_per_solver(self, small_instance: Instance) -> None:
        """The §10 table renders a header + one row per solution."""
        sols = run_comparison(small_instance, rl_policy=_FirstActionPolicy())
        table = format_comparison_table(sols, small_instance)
        assert "| Solver |" in table
        assert "|---|" in table
        for s in sols:
            assert s.solver in table
        # Four data rows (greedy, local_search, ilp, rl).
        data_rows = [
            ln
            for ln in table.splitlines()
            if ln.startswith("| ") and "---" not in ln and "Solver" not in ln
        ]
        assert len(data_rows) == 4

    def test_format_comparison_table_empty(self) -> None:
        """An empty solution list renders a placeholder."""
        assert format_comparison_table([]) == "(no solutions)"

    def test_fmt_runtime_branches(self) -> None:
        """_fmt_runtime covers seconds/minutes/hours formats."""
        from bike_rl.optim.evaluate import _fmt_runtime

        assert _fmt_runtime(0.5) == "0.50s"
        assert _fmt_runtime(125.0) == "2m05s"
        assert _fmt_runtime(3725.0) == "1h02m"

    def test_format_comparison_table_coverage_only_no_footnote(
        self, small_instance: Instance
    ) -> None:
        """Coverage-only weights with an OPTIMAL ILP row produce no surrogate footnote."""
        small_instance.weights = ObjectiveWeights(connectivity=0.0, coverage=1.0, fragmentation=0.0)
        sols = run_comparison(small_instance)
        table = format_comparison_table(sols, small_instance)
        assert "coverage surrogate" not in table


# ── build_instance_from_graphs ────────────────────────────────────────────


class TestBuildInstanceFromGraphs:
    """The instance builder derives candidates once for fairness (§8)."""

    def test_candidates_match_extract_candidates(self, small_instance: Instance) -> None:
        """build_instance_from_graphs derives the same candidate set the env uses."""
        inst = build_instance_from_graphs(
            small_instance.graph,
            small_instance.walk_graph,  # type: ignore[arg-type]
            small_instance.cfg,
            small_instance.weights,
            small_instance.budget,
            label="rebuilt",
        )
        # Same candidate identities (u, v, length, road_priority) as the env
        # would extract — the §8 fairness guarantee.
        assert [(c.u, c.v, c.length, c.road_priority) for c in inst.candidates] == [
            (c.u, c.v, c.length, c.road_priority) for c in small_instance.candidates
        ]
        assert inst.label == "rebuilt"
        assert inst.walk_graph is not None

    def test_walk_graph_preserved_for_rl(self, small_instance: Instance) -> None:
        """The rebuilt instance carries walk_graph so RL rollouts work."""
        inst = build_instance_from_graphs(
            small_instance.graph,
            small_instance.walk_graph,  # type: ignore[arg-type]
            small_instance.cfg,
            small_instance.weights,
            small_instance.budget,
        )
        sol = evaluate_rl_policy(_FirstActionPolicy(), inst)
        assert sol.solver == "rl"
