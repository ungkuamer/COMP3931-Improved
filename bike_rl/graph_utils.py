"""Graph conversion, loading, validation, and caching utilities.

See RECREATE_SPEC.md §3.3, §6.1, §6.5.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import networkx as nx
import rustworkx as rx

if TYPE_CHECKING:
    from bike_rl.config import Config

    _NXGraph = nx.MultiDiGraph[Any, Any, Any]
else:
    _NXGraph = nx.MultiDiGraph

logger = logging.getLogger(__name__)


def nx_to_rx(nx_graph: _NXGraph) -> tuple[rx.PyDiGraph, dict[int | str, int]]:
    """Convert a NetworkX MultiDiGraph to a rustworkx PyDiGraph.

    Uses ``rustworkx.networkx_converter`` with ``keep_attributes=True``.
    Missing-length defaulting is **not** done here; callers should run
    :func:`validate_graph` first if edges without a ``length`` attribute need
    to be filled (RECREATE_SPEC §6.5 — the O(V·E) defaulting loop is removed
    in favour of calling ``validate_graph`` separately).

    Args:
        nx_graph: The NetworkX MultiDiGraph to convert (e.g. an OSMnx graph).

    Returns:
        A ``(rx_graph, node_map)`` tuple where ``node_map`` maps each original
        NetworkX node id to its rustworkx node index, built from the
        ``"__networkx_node__"`` attribute set by ``networkx_converter``.
    """
    rx_graph: rx.PyDiGraph = rx.networkx_converter(nx_graph, keep_attributes=True)  # type: ignore[assignment]
    node_map: dict[int | str, int] = {}
    for rx_idx in rx_graph.node_indices():
        payload = rx_graph[rx_idx]
        node_map[payload["__networkx_node__"]] = rx_idx
    return rx_graph, node_map


def validate_graph(graph: _NXGraph, cfg: Config) -> None:
    """Validate a graph in-place: fill any missing edge ``length`` attribute.

    Missing lengths are set to ``cfg.default_edge_length`` (RECREATE_SPEC §3.3
    defaults to 1.0).  This keeps downstream code that reads
    ``data['length']`` (e.g. candidate extraction, cost computation) correct
    without each caller repeating the check.

    Args:
        graph: The graph to validate (mutated in-place for missing lengths).
        cfg: Config providing ``default_edge_length``.
    """
    for _u, _v, data in graph.edges(data=True):
        if "length" not in data:
            data["length"] = cfg.default_edge_length


def configure_osm_cache(cfg: Config) -> None:
    """Enable OSMnx on-disk caching so parallel workers don't re-download.

    Sets ``ox.settings.use_cache = True`` and points the cache folder at
    ``cfg.osm_cache_dir`` (RECREATE_SPEC §6.1). Safe to call repeatedly.

    Args:
        cfg: Config providing ``osm_cache_dir``.
    """
    import osmnx as ox

    ox.settings.use_cache = True
    ox.settings.log_console = False
    cache_dir = str(Path(cfg.osm_cache_dir).resolve())
    if hasattr(ox.settings, "cache_folder"):
        ox.settings.cache_folder = cache_dir


def load_city_graph(city_name: str, cfg: Config, network_type: str = "bike") -> _NXGraph:
    """Load a city street graph from OSMnx by place name.

    Args:
        city_name: e.g. ``"Otley, UK"``.
        cfg: Config (used to enable the OSM cache).
        network_type: OSMnx network type — ``"bike"`` for the existing bike
            network, ``"walk"`` for the candidate-source walk network
            (RECREATE_SPEC §3.1).

    Returns:
        The downloaded MultiDiGraph.
    """
    import osmnx as ox

    configure_osm_cache(cfg)
    return ox.graph_from_place(city_name, network_type=network_type)


def load_bbox_graph(
    north: float,
    south: float,
    east: float,
    west: float,
    cfg: Config,
    network_type: str = "bike",
) -> _NXGraph:
    """Load a street graph for a bounding box.

    Args:
        north: Northern boundary of the bounding box.
        south: Southern boundary of the bounding box.
        east: Eastern boundary of the bounding box.
        west: Western boundary of the bounding box (N,S,E,W order per RECREATE_SPEC §3.1).
        cfg: Config (used to enable the OSM cache).
        network_type: OSMnx network type (``"bike"`` or ``"walk"``).

    Returns:
        The downloaded MultiDiGraph.
    """
    import osmnx as ox

    configure_osm_cache(cfg)
    return ox.graph_from_bbox((north, south, east, west), network_type=network_type)


def cache_graph(graph: _NXGraph, path: Path) -> None:
    """Pickle a graph to ``path`` so it can be shared with worker processes.

    Used by ``training.py`` (plan 005) to avoid each ``SubprocVecEnv`` worker
    re-downloading OSM data (RECREATE_SPEC §6.1). Creates parent directories.

    Args:
        graph: The graph to serialise.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(graph, f)


def load_cached_graph(path: Path) -> _NXGraph:
    """Load a graph pickled by :func:`cache_graph`.

    Args:
        path: File path written by :func:`cache_graph`.

    Returns:
        The deserialised MultiDiGraph.
    """
    with path.open("rb") as f:
        return cast(_NXGraph, pickle.load(f))  # noqa: S301
