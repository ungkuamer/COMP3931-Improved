"""Candidate extraction for bike-network expansion.

See RECREATE_SPEC.md §3.2, §5.8.  Candidates are built fresh from the walk
graph without mutating it; ``connects_to_bike_path`` is re-derived from the
current bike-network node set at step time via :func:`recompute_connects`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import networkx as nx

if TYPE_CHECKING:
    from bike_rl.config import Config

    _NXGraph = nx.MultiDiGraph[Any, Any, Any]
else:
    _NXGraph = nx.MultiDiGraph


@dataclass(frozen=True)
class Candidate:
    """A candidate edge for bike-lane addition.

    Two candidates are equal iff ``(u, v, length, road_priority)`` match —
    ``connects_to_bike_path`` and ``data`` are excluded so that a recomputed
    candidate stays equal to its original for ``list.remove`` / set membership
    in the optimisers (OPTIMIZER_SPEC §5).

    Attributes:
        u: Source node id.
        v: Target node id.
        length: Edge length in metres.
        road_priority: Highway-class priority (5=primary … 1=unclassified).
        connects_to_bike_path: Whether an endpoint already touches the bike
            network (re-derived at step time; not part of identity).
        data: Copy of the original edge attribute dict (never the walk graph's
            own dict — RECREATE_SPEC §5.8; not part of identity).
    """

    u: int | str
    v: int | str
    length: float
    road_priority: int
    connects_to_bike_path: bool = field(compare=False)
    data: dict[str, Any] = field(default_factory=dict, compare=False)


def candidate_cost(c: Candidate, cfg: Config) -> float:
    """Return the construction cost of a candidate: ``length * edge_cost_factor``.

    Single source of truth for cost — used by the env (RECREATE_SPEC §3.4) and
    by the optimisers (OPTIMIZER_SPEC §4).

    Args:
        c: The candidate.
        cfg: Config providing ``edge_cost_factor``.

    Returns:
        ``c.length * cfg.edge_cost_factor``.
    """
    return c.length * cfg.edge_cost_factor


def _highway_priority(highway: object, priorities: dict[str, int]) -> int | None:
    """Resolve a (possibly list-valued) OSM ``highway`` tag to a priority.

    OSM ``highway`` can be a single string or a list of strings (e.g. when a
    way is tagged with multiple types). Returns the **max** priority among the
    listed types, or ``None`` if none are in ``priorities``.

    Args:
        highway: The ``highway`` value from an OSM edge.
        priorities: Mapping from highway type to priority (from ``Config``).

    Returns:
        The priority, or ``None`` to skip this edge.
    """
    if isinstance(highway, list):
        vals = [priorities[h] for h in highway if h in priorities]
        return max(vals) if vals else None
    if isinstance(highway, str):
        return priorities.get(highway)
    return None


def extract_candidates(
    bike_graph: _NXGraph,
    walk_graph: _NXGraph,
    cfg: Config,
) -> list[Candidate]:
    """Extract candidate bike-lane edges from the walk graph.

    A walk-graph edge is a candidate iff (RECREATE_SPEC §3.2):
      - it is **not** already in ``bike_graph`` (by ``(u, v)`` presence),
      - its ``highway`` type is in ``cfg.road_priorities``,
      - ``cfg.min_candidate_length < length < cfg.max_candidate_length``.

    Candidates are built as **fresh** dataclasses with a copied ``data`` dict;
    the source ``walk_graph`` is never mutated (§5.8). Results are sorted by
    ``(road_priority, connects_to_bike_path)`` descending.

    Args:
        bike_graph: The existing bike network.
        walk_graph: The walkable network candidates are drawn from.
        cfg: Config (length bounds, road priorities).

    Returns:
        Sorted list of candidates.
    """
    bike_nodes: set[int | str] = set(bike_graph.nodes())
    out: list[Candidate] = []
    for u, v, data in walk_graph.edges(data=True):
        if bike_graph.has_edge(u, v):
            continue
        priority = _highway_priority(data.get("highway"), cfg.road_priorities)
        if priority is None:
            continue
        length = data.get("length")
        if length is None or not (cfg.min_candidate_length < length < cfg.max_candidate_length):
            continue
        connects = (u in bike_nodes) or (v in bike_nodes)
        out.append(
            Candidate(
                u=u,
                v=v,
                length=float(length),
                road_priority=priority,
                connects_to_bike_path=connects,
                data=dict(data),
            )
        )
    out.sort(key=lambda c: (c.road_priority, c.connects_to_bike_path), reverse=True)
    return out


def recompute_connects(candidates: list[Candidate], bike_nodes: set[int | str]) -> list[Candidate]:
    """Re-derive ``connects_to_bike_path`` and re-sort candidates.

    Returns a **new list** of fresh ``Candidate`` objects; the inputs are not
    mutated. Equality is preserved on ``(u, v, length, road_priority)`` so the
    new objects can be used interchangeably with the originals in
    ``list.remove`` / sets.

    Args:
        candidates: Current candidate list.
        bike_nodes: Current set of nodes in the (growing) bike network.

    Returns:
        New sorted list with updated ``connects_to_bike_path`` flags.
    """
    out = [
        Candidate(
            u=c.u,
            v=c.v,
            length=c.length,
            road_priority=c.road_priority,
            connects_to_bike_path=(c.u in bike_nodes) or (c.v in bike_nodes),
            data=c.data,
        )
        for c in candidates
    ]
    out.sort(key=lambda c: (c.road_priority, c.connects_to_bike_path), reverse=True)
    return out
