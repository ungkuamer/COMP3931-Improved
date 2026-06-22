"""Shared pytest fixtures."""

from __future__ import annotations

import networkx as nx
import pytest


@pytest.fixture
def tiny_bike_graph() -> nx.MultiDiGraph:
    """A 4-node synthetic bike graph for smoke tests (no network access)."""
    g = nx.MultiDiGraph()
    g.add_node(1, x=0.0, y=0.0)
    g.add_node(2, x=0.001, y=0.0)
    g.add_node(3, x=0.0, y=0.001)
    g.add_node(4, x=0.001, y=0.001)
    g.add_edge(1, 2, length=120.0, highway="residential", bike_lane="yes")
    g.add_edge(3, 4, length=130.0, highway="residential", bike_lane="yes")
    return g


@pytest.fixture
def tiny_walk_graph() -> nx.MultiDiGraph:
    """A 5-node synthetic walk graph with candidate edges of mixed priority.

    Nodes 1-4 overlap with ``tiny_bike_graph``; node 5 is new. Edges:
      - (1,2) length 120 residential — already in bike graph (must be skipped).
      - (2,5) length 200 primary — candidate, connects_to_bike_path True.
      - (3,5) length 150 secondary — candidate, connects True.
      - (4,5) length 600 tertiary — candidate, connects True.
      - (1,5) length 50 residential — too short (< min), skipped.
      - (2,3) length 1200 residential — too long (> max), skipped.
      - (5,5) length 200 unclassified self-loop — highway ok, length ok.
      - (3,4) length 300 highway=['tertiary','residential'] (list-valued) —
        already in bike graph (skipped via has_edge).
    """
    g = nx.MultiDiGraph()
    NODES = {
        1: (0.0, 0.0),
        2: (0.001, 0.0),
        3: (0.0, 0.001),
        4: (0.001, 0.001),
        5: (0.002, 0.002),
    }
    for n, (x, y) in NODES.items():
        g.add_node(n, x=x, y=y)
    g.add_edge(1, 2, length=120.0, highway="residential")
    g.add_edge(2, 5, length=200.0, highway="primary")
    g.add_edge(3, 5, length=150.0, highway="secondary")
    g.add_edge(4, 5, length=600.0, highway="tertiary")
    g.add_edge(1, 5, length=50.0, highway="residential")
    g.add_edge(2, 3, length=1200.0, highway="residential")
    g.add_edge(5, 5, length=200.0, highway="unclassified")
    g.add_edge(3, 4, length=300.0, highway=["tertiary", "residential"])
    return g


@pytest.fixture
def mock_osm(monkeypatch):
    """Monkeypatch osmnx loader functions to return synthetic graphs (no network access).

    Yields a dict recording call args so tests can assert correct calls.
    """
    import osmnx as ox

    calls: dict[str, list] = {"place": [], "bbox": []}

    def fake_place(name, network_type="bike", **kw):
        calls["place"].append({"name": name, "network_type": network_type, **kw})
        g = nx.MultiDiGraph()
        g.add_node(1, x=0.0, y=0.0)
        g.add_node(2, x=0.001, y=0.0)
        g.add_edge(1, 2, length=120.0, highway="residential")
        return g

    def fake_bbox(bbox, network_type="bike", **kw):
        calls["bbox"].append(
            {
                "N": bbox[0],
                "S": bbox[1],
                "E": bbox[2],
                "W": bbox[3],
                "network_type": network_type,
                **kw,
            }
        )
        g = nx.MultiDiGraph()
        g.add_node(1, x=0.0, y=0.0)
        g.add_node(2, x=0.001, y=0.0)
        g.add_edge(1, 2, length=120.0, highway="residential")
        return g

    monkeypatch.setattr(ox, "graph_from_place", fake_place)
    monkeypatch.setattr(ox, "graph_from_bbox", fake_bbox)
    return calls
