"""Canonical scalar objective — the single source of truth for 'what we optimise'.

Both the RL reward (derived) and the direct optimisers (direct call) use this
module so RL and optimisers solve exactly the same problem. See
OPTIMIZER_SPEC.md §3 and RESEARCH_DIRECTION.md §4.1-4.2.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import networkx as nx

from bike_rl.metrics import connectivity, coverage, fragmentation

if TYPE_CHECKING:
    from bike_rl.config import Config

    _NXGraph = nx.MultiDiGraph[Any, Any, Any]
else:
    _NXGraph = nx.MultiDiGraph


@dataclass(frozen=True)
class ObjectiveWeights:
    """Weights for the canonical objective (OPTIMIZER_SPEC §3.1).

    Attributes:
        connectivity: Weight on graph connectivity in [0,1].
        coverage: Weight on bike-lane-reachable population (coverage) in [0,1].
        fragmentation: Weight on the fragmentation penalty in [0,1]; subtracted.
    """

    connectivity: float = 0.4
    coverage: float = 0.4
    fragmentation: float = 0.2


class Edge(Protocol):
    """Structural type for an addable bike-lane edge (satisfied by Candidate).

    Attributes:
        u: Source node id.
        v: Target node id.
        length: Edge length in metres.
        data: Copy of the original edge attribute dict.
    """

    @property
    def u(self) -> int | str: ...

    @property
    def v(self) -> int | str: ...

    @property
    def length(self) -> float: ...

    @property
    def data(self) -> dict[str, Any]: ...


def apply_added_edges(graph: _NXGraph, added_edges: Sequence[Edge]) -> _NXGraph:
    """Return a copy of ``graph`` with ``added_edges`` applied (bike_lane='yes').

    The input graph is not mutated. Each edge is added as a bike-lane edge
    following the same pattern as ``bike_rl/env.py:_apply_edge``. Self-loops
    are skipped (matching ``_bike_lane_subgraph`` and ``MetricsState.add_edge``).

    Args:
        graph: The base bike network graph.
        added_edges: Sequence of edges to add as bike lanes.

    Returns:
        A new ``nx.MultiDiGraph`` that is a copy of ``graph`` with the given
        edges tagged ``bike_lane='yes'``.
    """
    g = graph.copy()
    for e in added_edges:
        if e.u == e.v:
            continue
        data = dict(e.data)
        data["bike_lane"] = "yes"
        data["length"] = e.length
        g.add_edge(e.u, e.v, **data)
    return g


def objective(
    graph: _NXGraph,
    added_edges: Sequence[Edge],
    weights: ObjectiveWeights,
    cfg: Config,
) -> float:
    """Return the canonical objective value of ``graph`` with ``added_edges``.

    ``f(graph, added_edges) = w_conn * connectivity + w_cov * coverage
    - w_frag * fragmentation``.

    Higher is better. Pure and deterministic.

    Args:
        graph: The base bike network graph.
        added_edges: Sequence of candidate edges to add as bike lanes.
        weights: Weights for the three metric components.
        cfg: Config passed through to each metric function.

    Returns:
        The scalar objective value.
    """
    g = apply_added_edges(graph, added_edges)
    conn = connectivity(g, cfg)
    cov = coverage(g, cfg)
    frag = fragmentation(g, cfg)
    return weights.connectivity * conn + weights.coverage * cov - weights.fragmentation * frag


def objective_delta(
    graph: _NXGraph,
    added_edges: Sequence[Edge],
    new_edge: Edge,
    weights: ObjectiveWeights,
    cfg: Config,
) -> float:
    """Return the marginal objective change from adding ``new_edge``.

    ``objective_delta(g, S, e, w, cfg) = objective(g, S+[e], w, cfg) -
    objective(g, S, w, cfg)``.

    Currently implemented as a full-recompute difference (correctness-first).
    A faster ``MetricsState``-backed incremental version is deferred to a later
    performance plan.

    Args:
        graph: The base bike network graph.
        added_edges: Sequence of edges already added as bike lanes.
        new_edge: The candidate edge to evaluate for addition.
        weights: Weights for the three metric components.
        cfg: Config passed through to each metric function.

    Returns:
        The marginal change in objective (positive = improvement).
    """
    before = objective(graph, added_edges, weights, cfg)
    after = objective(graph, [*added_edges, new_edge], weights, cfg)
    return after - before
