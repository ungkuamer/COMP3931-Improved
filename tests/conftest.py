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
