"""Graph conversion, loading, validation, and caching utilities.

See RECREATE_SPEC.md §3.3, §6.1, §6.5.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import networkx as nx
    import rustworkx as rx

    from bike_rl.config import Config

_NXGraph = nx.MultiDiGraph[Any, Any, Any]


def nx_to_rx(nx_graph: _NXGraph) -> tuple[rx.PyDiGraph, dict[str, int]]:
    """Convert a NetworkX MultiDiGraph to a rustworkx PyDiGraph.

    Builds a node map using ``__networkx_node__`` attribute.
    """
    raise NotImplementedError("Plan 002")


def load_city_graph(city_name: str, cfg: Config) -> _NXGraph:
    """Load a city graph by name using OSMnx."""
    raise NotImplementedError("Plan 002")


def load_bbox_graph(north: float, south: float, east: float, west: float, cfg: Config) -> _NXGraph:
    """Load a graph for a bounding box (N,S,E,W order per RECREATE_SPEC §3.1)."""
    raise NotImplementedError("Plan 002")


def validate_graph(graph: _NXGraph) -> None:
    """Validate graph: fill missing length, raise on invalid."""
    raise NotImplementedError("Plan 002")


def cache_graph(graph: _NXGraph, path: Path) -> None:
    """Cache graph to disk."""
    raise NotImplementedError("Plan 002")


def load_cached_graph(path: Path) -> _NXGraph:
    """Load cached graph from disk."""
    raise NotImplementedError("Plan 002")
