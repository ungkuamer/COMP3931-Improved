"""Tests for graph metrics (connectivity, path efficiency, fragmentation, coverage).

See RECREATE_SPEC.md §3.6, §5.7, §6.2, §6.3, §6.7.
"""

from __future__ import annotations

import networkx as nx
import pytest

from bike_rl.config import Config
from bike_rl.metrics import (
    MetricsState,
    UnionFind,
    connectivity,
    coverage,
    fragmentation,
    path_efficiency,
)

# ── Local fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def triangle_graph() -> nx.MultiDiGraph:
    """3-node directed triangle, length 120 each, no bike_lane tags."""
    g = nx.MultiDiGraph()
    for n, (x, y) in {1: (0.0, 0.0), 2: (0.001, 0.0), 3: (0.0005, 0.001)}.items():
        g.add_node(n, x=x, y=y)
    g.add_edge(1, 2, length=120.0)
    g.add_edge(2, 3, length=120.0)
    g.add_edge(3, 1, length=120.0)
    return g


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


# ── connectivity tests ───────────────────────────────────────────────────


class TestConnectivity:
    """Tests for the pure ``connectivity`` function."""

    def test_connectivity_triangle_is_one(self, triangle_graph: nx.MultiDiGraph) -> None:
        """A triangle has transitivity 1.0."""
        assert connectivity(triangle_graph, Config()) == 1.0

    def test_connectivity_small_component_returns_zero(self) -> None:
        """Largest component with 2 nodes returns 0.0 (transitivity undefined)."""
        g = nx.MultiDiGraph()
        g.add_node(1, x=0.0, y=0.0)
        g.add_node(2, x=0.001, y=0.0)
        g.add_edge(1, 2, length=120.0)
        assert connectivity(g, Config()) == 0.0

    def test_connectivity_empty_graph_returns_zero(self) -> None:
        """Empty graph returns 0.0."""
        g = nx.MultiDiGraph()
        assert connectivity(g, Config()) == 0.0


# ── path_efficiency tests ────────────────────────────────────────────────


class TestPathEfficiency:
    """Tests for the pure ``path_efficiency`` function."""

    def test_path_efficiency_in_range(self, tiny_bike_graph: nx.MultiDiGraph) -> None:
        """Path efficiency is in (0,1] with a reasonable value."""
        val = path_efficiency(tiny_bike_graph, Config())
        assert 0.0 < val <= 1.0
        # tiny_bike_graph has a largest component of 2 nodes with
        # edge length 120 → efficiency ≈ 1/(1+0.12) ≈ 0.893
        assert val > 0.8
        assert val < 1.0

    def test_path_efficiency_deterministic(self, tiny_bike_graph: nx.MultiDiGraph) -> None:
        """Calling path_efficiency twice with the same config yields same result."""
        cfg = Config()
        v1 = path_efficiency(tiny_bike_graph, cfg)
        v2 = path_efficiency(tiny_bike_graph, cfg)
        assert v1 == v2

    def test_path_efficiency_single_node(self) -> None:
        """Single-node graph returns 1.0 (no pairs to measure)."""
        g = nx.MultiDiGraph()
        g.add_node(1, x=0.0, y=0.0)
        assert path_efficiency(g, Config()) == 1.0


# ── fragmentation tests ─────────────────────────────────────────────────


class TestFragmentation:
    """Tests for the pure ``fragmentation`` function."""

    def test_fragmentation_two_components(self, two_component_bike_graph: nx.MultiDiGraph) -> None:
        """Known-answer test: 2 components of size 2, 4 subgraph nodes.

        Expected: 0.3*min(1,1/10) + 0.4*(1-2/4) + 0.3*0 = 0.03 + 0.2 + 0 = 0.23.
        """
        val = fragmentation(two_component_bike_graph, Config())
        assert abs(val - 0.23) < 1e-9

    def test_fragmentation_empty_bike_lane_returns_zero(
        self, triangle_graph: nx.MultiDiGraph
    ) -> None:
        """No bike_lane='yes' edges → returns 0.0."""
        assert fragmentation(triangle_graph, Config()) == 0.0

    def test_fragmentation_single_chain(self) -> None:
        """Single chain of 3 bike-lane nodes: 1 component, largest=3, isolated=0.

        Expected: 0.3*min(1,0/10) + 0.4*(1-3/3) + 0.3*0/3 = 0.0.
        """
        g = nx.MultiDiGraph()
        for n, (x, y) in {1: (0.0, 0.0), 2: (0.001, 0.0), 3: (0.002, 0.0)}.items():
            g.add_node(n, x=x, y=y)
        g.add_edge(1, 2, length=100.0, bike_lane="yes")
        g.add_edge(2, 3, length=100.0, bike_lane="yes")
        assert fragmentation(g, Config()) == 0.0


# ── coverage tests (§5.7 fix) ───────────────────────────────────────────


class TestCoverage:
    """Tests for the pure ``coverage`` function — the §5.7 fix."""

    def test_coverage_is_sensitive_to_added_edge(
        self, two_component_bike_graph: nx.MultiDiGraph
    ) -> None:
        """§5.7 regression test: adding a bike-lane edge measurably increases coverage.

        With radius mode and 300m radius, all 5 nodes are within range of a
        bike-lane endpoint already (node 5 is ~249m from node 2).  Adding edge
        (2,5) with bike_lane='yes' grows the largest component from 2 to 3,
        which increases the largest_component_ratio term.
        """
        cfg = Config()  # coverage_mode='radius', coverage_radius_m=300
        before = coverage(two_component_bike_graph, cfg)
        # Add a bike-lane edge connecting node 2 to node 5
        g = two_component_bike_graph.copy()
        g.add_edge(2, 5, length=200.0, bike_lane="yes")
        after = coverage(g, cfg)
        assert after > before, "Coverage must increase after adding a bike-lane edge (§5.7)"
        assert before < 1.0
        assert after < 1.0

    def test_coverage_radius_mode_covers_nearby_nodes(
        self, two_component_bike_graph: nx.MultiDiGraph
    ) -> None:
        """With default radius=300m, all 5 nodes are covered by nearby bike-lane endpoints."""
        val = coverage(two_component_bike_graph, Config())
        # coverage_ratio = 5/5 = 1.0 (all within 300m), largest_ratio = 2/5 = 0.4
        # coverage = 0.7*1.0 + 0.3*0.4 = 0.82
        assert 0.7 < val < 1.0

    def test_coverage_component_mode(self, two_component_bike_graph: nx.MultiDiGraph) -> None:
        """Component mode: only nodes in same component as a bike-lane node count."""
        cfg = Config(coverage_mode="component")
        # Components in simplified undirected: {1,2}, {3,4}, {5}
        # Bike-lane components: {1,2} and {3,4} = 4 covered nodes
        # coverage_ratio = 4/5 = 0.8, largest_ratio = 2/5 = 0.4
        # coverage = 0.7*0.8 + 0.3*0.4 = 0.68
        val = coverage(two_component_bike_graph, cfg)
        assert 0.4 <= val < 1.0

    def test_coverage_no_bike_lane_returns_zero(self, triangle_graph: nx.MultiDiGraph) -> None:
        """No bike_lane='yes' edges → returns 0.0."""
        assert coverage(triangle_graph, Config()) == 0.0

    def test_coverage_empty_graph_returns_zero(self) -> None:
        """Empty graph returns 0.0."""
        g = nx.MultiDiGraph()
        assert coverage(g, Config()) == 0.0


# ── UnionFind tests ──────────────────────────────────────────────────────


class TestUnionFind:
    """Tests for the ``UnionFind`` incremental component tracker (§6.3)."""

    def test_union_find_basic(self) -> None:
        """Union operations correctly update component count, largest, and isolated."""
        uf = UnionFind([1, 2, 3, 4])
        assert uf.num_components == 4
        assert uf.largest_size == 1
        assert uf.isolated_count == 4

        uf.union(1, 2)
        assert uf.num_components == 3
        assert uf.largest_size == 2
        assert uf.isolated_count == 2  # {1,2} not isolated, {3}, {4} are

        uf.union(3, 4)
        assert uf.num_components == 2
        assert uf.largest_size == 2
        assert uf.isolated_count == 0

        uf.union(2, 3)
        assert uf.num_components == 1
        assert uf.largest_size == 4
        assert uf.isolated_count == 0

        # All nodes in same set
        assert uf.find(1) == uf.find(4)

    def test_union_find_add_node(self) -> None:
        """Add_node adds new nodes and is idempotent for existing ones."""
        uf = UnionFind([1])
        assert uf.num_components == 1
        uf.add_node(2)
        assert uf.num_components == 2
        assert uf.isolated_count == 2  # both singletons
        uf.add_node(2)  # idempotent
        assert uf.num_components == 2


# ── MetricsState tests ───────────────────────────────────────────────────


class TestMetricsState:
    """Tests for the ``MetricsState`` incremental cache (§6.2/§6.3/§5.14)."""

    def test_metrics_state_matches_pure(self, two_component_bike_graph: nx.MultiDiGraph) -> None:
        """MetricsState returns the same values as the pure functions."""
        cfg = Config()
        st = MetricsState(two_component_bike_graph, cfg)
        assert abs(st.fragmentation() - fragmentation(two_component_bike_graph, cfg)) < 1e-9
        assert abs(st.coverage() - coverage(two_component_bike_graph, cfg)) < 1e-9
        assert abs(st.connectivity() - connectivity(two_component_bike_graph, cfg)) < 1e-9
        assert abs(st.path_efficiency() - path_efficiency(two_component_bike_graph, cfg)) < 1e-9

    def test_metrics_state_incremental_after_add(
        self, two_component_bike_graph: nx.MultiDiGraph
    ) -> None:
        """After add_edge, MetricsState stays consistent with pure functions."""
        cfg = Config()
        st = MetricsState(two_component_bike_graph, cfg)
        # add to metrics state
        st.add_edge(2, 5, {"length": 200.0, "bike_lane": "yes"})
        # add to canonical graph for pure functions
        g = two_component_bike_graph.copy()
        g.add_edge(2, 5, length=200.0, bike_lane="yes")
        assert abs(st.fragmentation() - fragmentation(g, cfg)) < 1e-9
        assert abs(st.coverage() - coverage(g, cfg)) < 1e-9

    def test_metrics_state_cache_invalidates(
        self, two_component_bike_graph: nx.MultiDiGraph
    ) -> None:
        """Connectivity cache returns same value on repeated calls; version increments."""
        cfg = Config()
        st = MetricsState(two_component_bike_graph, cfg)
        # First call populates cache
        c0 = st.connectivity()
        # Second call with no change — should be identical
        c1 = st.connectivity()
        assert c0 == c1
        # Record version
        v0 = st._version
        # Add a non-bike-lane edge (changes whole graph)
        st.add_edge(2, 5, {"length": 200.0})
        assert st._version > v0  # version incremented

    def test_metrics_state_ignores_self_loop(
        self, two_component_bike_graph: nx.MultiDiGraph
    ) -> None:
        """Self-loops are ignored, matching _bike_lane_subgraph / _simplified_undirected."""
        cfg = Config()
        st = MetricsState(two_component_bike_graph, cfg)
        before = st.fragmentation()
        st.add_edge(5, 5, {"length": 10.0, "bike_lane": "yes"})
        assert st.fragmentation() == before

    def test_metrics_state_connectivity_and_efficiency_after_non_bike_edge(
        self, two_component_bike_graph: nx.MultiDiGraph
    ) -> None:
        """Adding a non-bike-lane edge still affects connectivity/efficiency."""
        cfg = Config()
        st = MetricsState(two_component_bike_graph, cfg)
        # Add a non-bike-lane edge connecting the two components
        st.add_edge(2, 3, {"length": 100.0})
        # The pure functions on an updated graph should match
        g = two_component_bike_graph.copy()
        g.add_edge(2, 3, length=100.0)
        assert abs(st.connectivity() - connectivity(g, cfg)) < 1e-9
        assert abs(st.path_efficiency() - path_efficiency(g, cfg)) < 1e-9
