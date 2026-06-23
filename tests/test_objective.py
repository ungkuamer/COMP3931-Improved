"""Tests for the canonical objective (objective.py).

Covers purity, determinism, delta-vs-recompute, monotonicity, submodularity,
weighted decomposition, the empty-selection base case, input non-mutation,
bike-lane tagging, self-loop skip, and duplicate-edge zero delta.

See OPTIMIZER_SPEC.md §9 and plans/010-objective-and-tests.md.
"""

from __future__ import annotations

import networkx as nx
import pytest

from bike_rl.candidates import Candidate
from bike_rl.config import Config
from bike_rl.metrics import connectivity, coverage, fragmentation
from bike_rl.objective import (
    ObjectiveWeights,
    apply_added_edges,
    objective,
    objective_delta,
)

# ── Helper ───────────────────────────────────────────────────────────────


def _edge(u: int | str, v: int | str, length: float, **extra: object) -> Candidate:
    """Build a Candidate satisfying the Edge protocol.

    Args:
        u: Source node id.
        v: Target node id.
        length: Edge length in metres.
        **extra: Extra attributes to include in the data dict.

    Returns:
        A Candidate with default road_priority and connects_to_bike_path;
        the data dict includes length and any extras.
    """
    return Candidate(
        u=u,
        v=v,
        length=float(length),
        road_priority=1,
        connects_to_bike_path=False,
        data={"length": float(length), **extra},
    )


# ── Local fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def default_weights() -> ObjectiveWeights:
    """Default objective weights (0.4 / 0.4 / 0.2)."""
    return ObjectiveWeights()


@pytest.fixture
def default_cfg() -> Config:
    """Default config (radius mode, 300m)."""
    return Config()


@pytest.fixture
def two_component_bike_graph() -> nx.MultiDiGraph:
    """5 nodes: bike-lane edges (1,2) and (3,4); node 5 unconnected.

    bike_lane='yes' on both edges. Used for fragmentation/coverage known
    answers and the §5.7 sensitivity test.
    """
    g = nx.MultiDiGraph()
    for n, (x, y) in {
        1: (0.0, 0.0),
        2: (0.001, 0.0),
        3: (0.0, 0.001),
        4: (0.001, 0.001),
        5: (0.002, 0.002),
    }.items():
        g.add_node(n, x=x, y=y)
    g.add_edge(1, 2, length=120.0, bike_lane="yes")
    g.add_edge(3, 4, length=130.0, bike_lane="yes")
    return g


# ── Purity / determinism ────────────────────────────────────────────────


class TestObjectivePurity:
    """Tests for the purity and determinism of objective."""

    def test_objective_is_pure_and_deterministic(
        self,
        two_component_bike_graph: nx.MultiDiGraph,
        default_weights: ObjectiveWeights,
        default_cfg: Config,
    ) -> None:
        """Calling objective twice yields identical results.

        The input graph is not mutated.
        """
        edges = [_edge(2, 5, 200.0)]
        snapshot = two_component_bike_graph.copy()

        v1 = objective(two_component_bike_graph, edges, default_weights, default_cfg)
        v2 = objective(two_component_bike_graph, edges, default_weights, default_cfg)

        assert v1 == v2

        # Input graph must not have been mutated: same edge structure
        assert list(two_component_bike_graph.edges(data=True)) == list(snapshot.edges(data=True))

    def test_objective_empty_added_edges_equals_base(
        self,
        two_component_bike_graph: nx.MultiDiGraph,
        default_weights: ObjectiveWeights,
        default_cfg: Config,
    ) -> None:
        """objective(g, [], w, cfg) equals the weighted sum of base metrics.

        Comparison within 1e-9.
        """
        obj = objective(two_component_bike_graph, [], default_weights, default_cfg)
        conn = connectivity(two_component_bike_graph, default_cfg)
        cov = coverage(two_component_bike_graph, default_cfg)
        frag = fragmentation(two_component_bike_graph, default_cfg)
        expected = (
            default_weights.connectivity * conn
            + default_weights.coverage * cov
            - default_weights.fragmentation * frag
        )
        assert abs(obj - expected) < 1e-9


# ── objective_delta vs full recompute ──────────────────────────────────


class TestObjectiveDelta:
    """Tests for objective_delta correctness."""

    def test_objective_delta_matches_full_recompute(
        self,
        two_component_bike_graph: nx.MultiDiGraph,
        default_weights: ObjectiveWeights,
        default_cfg: Config,
    ) -> None:
        """objective_delta equals objective(g, S+[e]) - objective(g, S).

        Comparison within 1e-9.
        """
        e = _edge(2, 5, 200.0)
        S = [_edge(3, 5, 150.0)]

        delta = objective_delta(two_component_bike_graph, S, e, default_weights, default_cfg)
        after = objective(two_component_bike_graph, [*S, e], default_weights, default_cfg)
        before = objective(two_component_bike_graph, S, default_weights, default_cfg)
        expected = after - before

        assert abs(delta - expected) < 1e-9

    def test_objective_delta_zero_for_duplicate_edge(
        self,
        two_component_bike_graph: nx.MultiDiGraph,
        default_weights: ObjectiveWeights,
        default_cfg: Config,
    ) -> None:
        """Adding a duplicate edge yields delta close to zero.

        The second add changes no bike-lane structure.
        """
        e = _edge(2, 5, 200.0)
        delta = objective_delta(two_component_bike_graph, [e], e, default_weights, default_cfg)
        assert abs(delta) < 1e-9


# ── Monotonicity ────────────────────────────────────────────────────────


class TestMonotonicity:
    """Tests that adding a useful bike-lane edge improves the objective."""

    def test_objective_monotone_for_useful_edge(
        self,
        two_component_bike_graph: nx.MultiDiGraph,
        default_weights: ObjectiveWeights,
        default_cfg: Config,
    ) -> None:
        """Adding edge (2,5) to two_component_bike_graph increases objective.

        Both direct evaluation and delta confirm delta > 0.
        """
        e = _edge(2, 5, 200.0)
        before = objective(two_component_bike_graph, [], default_weights, default_cfg)
        after = objective(two_component_bike_graph, [e], default_weights, default_cfg)
        assert after > before
        delta = objective_delta(two_component_bike_graph, [], e, default_weights, default_cfg)
        assert delta > 0.0


# ── Submodularity (empirical) ────────────────────────────────────────────


class TestSubmodularity:
    """Empirical check of decreasing returns (submodularity, §9)."""

    def test_objective_submodular_decreasing_returns(
        self,
        two_component_bike_graph: nx.MultiDiGraph,
        default_weights: ObjectiveWeights,
        default_cfg: Config,
    ) -> None:
        """Adding an edge gives delta_sparse >= delta_rich.

        The richer selection already connects the two components, making
        the second connection redundant.
        """
        e = _edge(2, 3, 100.0)  # connects the two components
        rich_edges = [_edge(1, 4, 110.0)]  # already connects components

        delta_sparse = objective_delta(
            two_component_bike_graph, [], e, default_weights, default_cfg
        )
        delta_rich = objective_delta(
            two_component_bike_graph, rich_edges, e, default_weights, default_cfg
        )

        assert delta_sparse >= delta_rich - 1e-9, (
            f"Expected delta_sparse ({delta_sparse}) >= delta_rich ({delta_rich})"
        )


# ── Weighted decomposition ───────────────────────────────────────────────


class TestWeightedDecomposition:
    """Tests objective decomposition into weighted metric components."""

    def test_objective_decomposes_into_weighted_metrics(
        self,
        two_component_bike_graph: nx.MultiDiGraph,
        default_cfg: Config,
    ) -> None:
        """With one non-zero weight, objective equals that metric component.

        Fragmentation is subtracted (negated).
        """
        edges = [_edge(2, 5, 200.0)]
        g = apply_added_edges(two_component_bike_graph, edges)

        # connectivity-only weights
        w_conn = ObjectiveWeights(connectivity=1.0, coverage=0.0, fragmentation=0.0)
        obj_conn = objective(two_component_bike_graph, edges, w_conn, default_cfg)
        assert abs(obj_conn - connectivity(g, default_cfg)) < 1e-9

        # coverage-only weights
        w_cov = ObjectiveWeights(connectivity=0.0, coverage=1.0, fragmentation=0.0)
        obj_cov = objective(two_component_bike_graph, edges, w_cov, default_cfg)
        assert abs(obj_cov - coverage(g, default_cfg)) < 1e-9

        # fragmentation-only weights (fragmentation is subtracted)
        w_frag = ObjectiveWeights(connectivity=0.0, coverage=0.0, fragmentation=1.0)
        obj_frag = objective(two_component_bike_graph, edges, w_frag, default_cfg)
        assert abs(obj_frag - (-fragmentation(g, default_cfg))) < 1e-9


# ── apply_added_edges ────────────────────────────────────────────────────


class TestApplyAddedEdges:
    """Tests for the apply_added_edges function."""

    def test_apply_added_edges_does_not_mutate_input(
        self,
        two_component_bike_graph: nx.MultiDiGraph,
    ) -> None:
        """The original graph is unchanged after apply_added_edges.

        Verified by snapshot comparison.
        """
        snapshot = list(two_component_bike_graph.edges(data=True))
        result = apply_added_edges(two_component_bike_graph, [_edge(2, 5, 200.0)])
        assert result is not two_component_bike_graph
        assert list(two_component_bike_graph.edges(data=True)) == snapshot

    def test_apply_added_edges_sets_bike_lane_tag(
        self,
        two_component_bike_graph: nx.MultiDiGraph,
    ) -> None:
        """The returned graph has bike_lane='yes' and length preserved."""
        result = apply_added_edges(two_component_bike_graph, [_edge(2, 5, 200.0)])
        assert result.has_edge(2, 5)
        edge_data = result.get_edge_data(2, 5)
        # get_edge_data returns a dict keyed by key; grab the first
        first_key = next(iter(edge_data))
        d = edge_data[first_key]
        assert d.get("bike_lane") == "yes"
        assert d.get("length") == 200.0

    def test_apply_added_edges_skips_self_loop(
        self,
        two_component_bike_graph: nx.MultiDiGraph,
    ) -> None:
        """Self-loop edges are skipped (no new edge in result)."""
        before_count = two_component_bike_graph.number_of_edges()
        result = apply_added_edges(two_component_bike_graph, [_edge(5, 5, 10.0)])
        assert result.number_of_edges() == before_count
        # Also check that the original is untouched
        assert not result.has_edge(5, 5)
