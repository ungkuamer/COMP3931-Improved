"""Graph metrics: connectivity, path efficiency, fragmentation, coverage.

See RECREATE_SPEC.md §3.6, §6.2, §6.3, §6.7.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import networkx as nx

    from bike_rl.config import Config

_NXGraph = nx.MultiDiGraph[Any, Any, Any]


def connectivity(graph: _NXGraph, cfg: Config) -> float:
    """Return connectivity metric (largest component / total nodes)."""
    raise NotImplementedError("Plan 003")


def path_efficiency(graph: _NXGraph, cfg: Config) -> float:
    """Return mean path efficiency (inverse shortest-path distance)."""
    raise NotImplementedError("Plan 003")


def fragmentation(graph: _NXGraph, cfg: Config) -> float:
    """Return fragmentation penalty (lowest among components)."""
    raise NotImplementedError("Plan 003")


def coverage(graph: _NXGraph, cfg: Config) -> float:
    """Return population coverage (nodes within radius of any bike lane)."""
    raise NotImplementedError("Plan 003")
