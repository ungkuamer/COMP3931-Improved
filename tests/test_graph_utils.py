"""Tests for ``bike_rl.graph_utils``.

Covers: nx_to_rx structural preservation + node_map; validate_graph fills
missing length; configure_osm_cache; load_city_graph / load_bbox_graph call
OSMnx with correct args; cache_graph / load_cached_graph round-trip.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx

from bike_rl.config import Config
from bike_rl.graph_utils import (
    cache_graph,
    configure_osm_cache,
    load_bbox_graph,
    load_cached_graph,
    load_city_graph,
    nx_to_rx,
    validate_graph,
)

if TYPE_CHECKING:
    _NXGraph = nx.MultiDiGraph[Any, Any, Any]
else:
    _NXGraph = nx.MultiDiGraph

_MockCalls = dict[str, list[dict[str, Any]]]


def test_nx_to_rx_preserves_nodes_and_edges(
    tiny_bike_graph: _NXGraph,
) -> None:
    """rx_graph node/edge count matches the source NetworkX graph."""
    rx_graph, node_map = nx_to_rx(tiny_bike_graph)
    assert rx_graph.num_nodes() == tiny_bike_graph.number_of_nodes()
    assert rx_graph.num_edges() == tiny_bike_graph.number_of_edges()


def test_nx_to_rx_node_map_is_bijection(
    tiny_bike_graph: _NXGraph,
) -> None:
    """node_map maps every original node id to a unique rx index."""
    rx_graph, node_map = nx_to_rx(tiny_bike_graph)
    num_nodes = tiny_bike_graph.number_of_nodes()
    # Every original node id is a key
    for orig_id in tiny_bike_graph.nodes():
        assert orig_id in node_map
    # Values are a permutation of 0..num_nodes-1
    assert set(node_map.values()) == set(range(num_nodes))
    # __networkx_node__ round-trips
    for orig_id, rx_idx in node_map.items():
        assert rx_graph[rx_idx]["__networkx_node__"] == orig_id


def test_nx_to_rx_preserves_edge_data() -> None:
    """nx_to_rx preserves edge attributes without defaulting missing length.

    With the ``weight_fn`` removed (per the adapted plan), ``nx_to_rx`` copies
    edge data as-is; missing-length defaulting is the responsibility of
    :func:`validate_graph`.
    """
    g: _NXGraph = nx.MultiDiGraph()
    g.add_node(1, x=0.0, y=0.0)
    g.add_node(2, x=1.0, y=0.0)
    g.add_node(3, x=2.0, y=0.0)
    g.add_edge(1, 2, length=42.0, highway="primary")
    g.add_edge(2, 3, highway="residential")  # no length attribute

    rx_graph, node_map = nx_to_rx(g)

    # Find edge (1,2) — should have length=42.0 preserved
    idx1, idx2 = node_map[1], node_map[2]
    edge_data_12 = rx_graph.get_edge_data(idx1, idx2)
    assert edge_data_12 is not None
    assert edge_data_12["length"] == 42.0
    assert edge_data_12["highway"] == "primary"

    # Find edge (2,3) — should have no length attribute (not defaulted)
    idx3 = node_map[3]
    edge_data_23 = rx_graph.get_edge_data(idx2, idx3)
    assert edge_data_23 is not None
    assert "length" not in edge_data_23
    assert edge_data_23["highway"] == "residential"


def test_validate_graph_fills_missing_length() -> None:
    """validate_graph sets missing length to default_edge_length."""
    g: _NXGraph = nx.MultiDiGraph()
    g.add_node(1, x=0.0, y=0.0)
    g.add_node(2, x=1.0, y=0.0)
    g.add_edge(1, 2, highway="residential")  # no length
    cfg = Config()

    validate_graph(g, cfg)

    for _u, _v, data in g.edges(data=True):
        assert data["length"] == cfg.default_edge_length


def test_validate_graph_preserves_existing_length() -> None:
    """validate_graph does not overwrite existing length values."""
    g: _NXGraph = nx.MultiDiGraph()
    g.add_node(1, x=0.0, y=0.0)
    g.add_node(2, x=1.0, y=0.0)
    g.add_edge(1, 2, length=42.0, highway="residential")
    cfg = Config()

    validate_graph(g, cfg)

    for _u, _v, data in g.edges(data=True):
        assert data["length"] == 42.0


def test_configure_osm_cache_enables_use_cache() -> None:
    """configure_osm_cache sets ox.settings.use_cache to True."""
    import osmnx as ox

    old_use_cache = ox.settings.use_cache
    try:
        configure_osm_cache(Config())
        assert ox.settings.use_cache is True
    finally:
        ox.settings.use_cache = old_use_cache


def test_load_city_graph_calls_graph_from_place(
    mock_osm: _MockCalls,
) -> None:
    """load_city_graph delegates to ox.graph_from_place with correct args."""
    load_city_graph("Otley, UK", Config(), network_type="bike")
    assert len(mock_osm["place"]) == 1
    assert mock_osm["place"][0]["name"] == "Otley, UK"
    assert mock_osm["place"][0]["network_type"] == "bike"


def test_load_city_graph_network_type_walk(
    mock_osm: _MockCalls,
) -> None:
    """load_city_graph passes network_type='walk' correctly."""
    load_city_graph("Otley, UK", Config(), network_type="walk")
    assert mock_osm["place"][0]["network_type"] == "walk"


def test_load_bbox_graph_uses_nsew_order(
    mock_osm: _MockCalls,
) -> None:
    """load_bbox_graph passes (N, S, E, W) in correct order (RECREATE_SPEC §3.1)."""
    load_bbox_graph(53.9, 53.8, -1.6, -1.7, Config())
    assert len(mock_osm["bbox"]) == 1
    call = mock_osm["bbox"][0]
    assert call["N"] == 53.9
    assert call["S"] == 53.8
    assert call["E"] == -1.6
    assert call["W"] == -1.7


def test_cache_graph_round_trip(
    tiny_bike_graph: _NXGraph,
    tmp_path: Path,
) -> None:
    """cache_graph / load_cached_graph round-trips correctly."""
    path = tmp_path / "g.pkl"
    cache_graph(tiny_bike_graph, path)
    g2 = load_cached_graph(path)
    assert g2.number_of_nodes() == tiny_bike_graph.number_of_nodes()
    assert g2.number_of_edges() == tiny_bike_graph.number_of_edges()
    # Check edge data preserved
    edge_list = list(g2.edges(data=True))
    found = False
    for u, v, d in edge_list:
        if u == 1 and v == 2:
            assert d["length"] == 120.0
            found = True
            break
    assert found, "Edge (1,2) not found in loaded graph"
