"""Tests for ``bike_rl.candidates``.

Covers: length filter, road-priority ordering, no mutation of the source
graph, list-valued highway, connects flag, recompute_connects propagation
and freshness, candidate equality stability, candidate_cost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import networkx as nx
import pytest

from bike_rl.candidates import (
    Candidate,
    candidate_cost,
    extract_candidates,
    recompute_connects,
)
from bike_rl.config import Config

if TYPE_CHECKING:
    _NXGraph = nx.MultiDiGraph[Any, Any, Any]
else:
    _NXGraph = nx.MultiDiGraph


def _default_cfg() -> Config:
    return Config()


def test_extract_candidates_filters_by_length(
    tiny_bike_graph: _NXGraph,
    tiny_walk_graph: _NXGraph,
) -> None:
    """Candidates respect min/max length bounds and skip existing bike edges."""
    cfg = _default_cfg()
    candidates = extract_candidates(tiny_bike_graph, tiny_walk_graph, cfg)

    # No candidate should have length <= min or >= max
    for c in candidates:
        assert cfg.min_candidate_length < c.length < cfg.max_candidate_length, (
            f"Candidate {c.u}->{c.v} length {c.length} out of bounds"
        )

    # The (1,2) edge already in bike graph must not be a candidate
    for c in candidates:
        assert not (c.u == 1 and c.v == 2), "Edge (1,2) should be skipped"
        assert not (c.u == 3 and c.v == 4), "Edge (3,4) should be skipped"

    # The (1,5) edge (length 50, < 100) must be excluded
    for c in candidates:
        assert not (c.u == 1 and c.v == 5 and c.length == 50.0)

    # The (2,3) edge (length 1200, > 1000) must be excluded
    for c in candidates:
        assert not (c.u == 2 and c.v == 3 and c.length == 1200.0)


def test_extract_candidates_orders_by_priority_then_connects(
    tiny_bike_graph: _NXGraph,
    tiny_walk_graph: _NXGraph,
) -> None:
    """Candidates are sorted by (road_priority, connects) descending."""
    cfg = _default_cfg()
    candidates = extract_candidates(tiny_bike_graph, tiny_walk_graph, cfg)

    assert len(candidates) >= 2
    for i in range(len(candidates) - 1):
        key_i = (candidates[i].road_priority, candidates[i].connects_to_bike_path)
        key_next = (
            candidates[i + 1].road_priority,
            candidates[i + 1].connects_to_bike_path,
        )
        assert key_i >= key_next, f"Candidates not sorted at index {i}: {key_i} < {key_next}"

    # Primary (priority 5) comes before tertiary (priority 3)
    priorities = [c.road_priority for c in candidates]
    assert 5 in priorities
    first_primary_idx = priorities.index(5)
    if 3 in priorities:
        last_tertiary_idx = len(priorities) - 1 - priorities[::-1].index(3)
        assert first_primary_idx < last_tertiary_idx


def test_extract_candidates_does_not_mutate_source(
    tiny_bike_graph: _NXGraph,
    tiny_walk_graph: _NXGraph,
) -> None:
    """Source walk_graph is not mutated (§5.8)."""
    # Snapshot edge data dict ids before extraction
    before_ids: set[int] = set()
    for _u, _v, data in tiny_walk_graph.edges(data=True):
        before_ids.add(id(data))

    extract_candidates(tiny_bike_graph, tiny_walk_graph, _default_cfg())

    # After extraction, no edge dict should have new keys added
    for _u, _v, data in tiny_walk_graph.edges(data=True):
        assert id(data) in before_ids, "Edge data dict was replaced"
        assert "road_priority" not in data, "Source graph was mutated"
        assert "connects_to_bike_path" not in data, "Source graph was mutated"


def test_extract_candidates_handles_list_highway() -> None:
    """List-valued highway resolves to max priority."""
    bike: _NXGraph = nx.MultiDiGraph()
    bike.add_node(10, x=0.0, y=0.0)
    bike.add_node(11, x=0.001, y=0.0)

    walk: _NXGraph = nx.MultiDiGraph()
    walk.add_node(10, x=0.0, y=0.0)
    walk.add_node(11, x=0.001, y=0.0)
    walk.add_edge(10, 11, length=200.0, highway=["primary", "residential"])

    cfg = _default_cfg()
    candidates = extract_candidates(bike, walk, cfg)

    assert len(candidates) == 1
    assert candidates[0].road_priority == 5  # max of primary(5), residential(2)


def test_extract_candidates_connects_flag() -> None:
    """connects_to_bike_path is False when neither endpoint is in bike graph."""
    bike: _NXGraph = nx.MultiDiGraph()
    bike.add_node(1, x=0.0, y=0.0)

    walk: _NXGraph = nx.MultiDiGraph()
    walk.add_node(100, x=0.0, y=0.0)
    walk.add_node(101, x=0.001, y=0.0)
    walk.add_edge(100, 101, length=200.0, highway="residential")

    candidates = extract_candidates(bike, walk, _default_cfg())

    assert len(candidates) >= 1
    for c in candidates:
        if c.u == 100 and c.v == 101:
            assert c.connects_to_bike_path is False
            break
    else:
        pytest.fail("Candidate (100,101) not found")


def test_recompute_connects_propagates_after_add(
    tiny_bike_graph: _NXGraph,
    tiny_walk_graph: _NXGraph,
) -> None:
    """recompute_connects updates connects flag when node is added to bike network."""
    cfg = _default_cfg()
    candidates = extract_candidates(tiny_bike_graph, tiny_walk_graph, cfg)

    # The (5,5) self-loop candidate's endpoint 5 is not in bike nodes
    bike_nodes: set[int | str] = set(tiny_bike_graph.nodes())
    # Find a candidate that doesn't connect yet
    disconnected = [c for c in candidates if not c.connects_to_bike_path]
    # At minimum the self-loop (5,5) should be disconnected
    assert len(disconnected) >= 1, "Expected at least one disconnected candidate"

    # Add node 5 to bike network
    updated = recompute_connects(candidates, bike_nodes | {5})
    # Previously disconnected candidates should now connect
    for c in updated:
        if c.u == 5 or c.v == 5:
            assert c.connects_to_bike_path is True


def test_recompute_connects_returns_new_list(
    tiny_bike_graph: _NXGraph,
    tiny_walk_graph: _NXGraph,
) -> None:
    """recompute_connects creates a new list without mutating inputs."""
    cfg = _default_cfg()
    candidates = extract_candidates(tiny_bike_graph, tiny_walk_graph, cfg)
    bike_nodes: set[int | str] = set(tiny_bike_graph.nodes())

    # Capture original connects values
    original_connects = [c.connects_to_bike_path for c in candidates]

    updated = recompute_connects(candidates, bike_nodes | {5})

    assert updated is not candidates, "Should return a new list"
    # Input candidates should be unchanged
    for i, c in enumerate(candidates):
        assert c.connects_to_bike_path == original_connects[i], "Input candidates were mutated"


def test_candidate_equality_stable_across_recompute() -> None:
    """Candidates with same identity fields are equal despite different extra data.

    Candidates sharing ``(u, v, length, road_priority)`` are equal even when
    ``connects_to_bike_path`` or ``data`` differ (OPTIMIZER_SPEC §5).
    """
    c1 = Candidate(
        u=1,
        v=2,
        length=200.0,
        road_priority=5,
        connects_to_bike_path=True,
        data={"highway": "primary"},
    )
    c2 = Candidate(
        u=1,
        v=2,
        length=200.0,
        road_priority=5,
        connects_to_bike_path=False,
        data={"highway": "secondary"},
    )

    assert c1 == c2, "Candidates should be equal despite different connects/data"
    assert hash(c1) == hash(c2), "Hashes should match"

    # list.remove should work (this is what greedy solver relies on)
    lst = [c1]
    lst.remove(c2)  # Should not raise ValueError
    assert len(lst) == 0


def test_candidate_cost() -> None:
    """candidate_cost returns length * edge_cost_factor."""
    c = Candidate(
        u=1,
        v=2,
        length=200.0,
        road_priority=5,
        connects_to_bike_path=True,
    )
    cfg = _default_cfg()
    expected = 200.0 * cfg.edge_cost_factor  # 200 * 10 = 2000
    assert candidate_cost(c, cfg) == expected
