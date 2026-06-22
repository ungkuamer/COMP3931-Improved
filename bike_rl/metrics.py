"""Graph metrics: connectivity, path efficiency, fragmentation, coverage.

See RECREATE_SPEC.md §3.6, §5.7, §6.2, §6.3, §6.7.  The four public functions
are pure and deterministic; :class:`MetricsState` provides an incremental
cache used by the RL env (plan 004) and the optimisers (OPTIMIZER_SPEC §3).
"""

from __future__ import annotations

import logging
import math
import random
from typing import TYPE_CHECKING, Any

import networkx as nx
import rustworkx as rx

if TYPE_CHECKING:
    from bike_rl.config import Config

    _NXGraph = nx.MultiDiGraph[Any, Any, Any]
else:
    _NXGraph = nx.MultiDiGraph

logger = logging.getLogger(__name__)


# ── Private helpers ──────────────────────────────────────────────────────


def _simplified_undirected(graph: _NXGraph) -> nx.Graph[Any, Any]:
    """Return a simple undirected ``nx.Graph`` view of ``graph``.

    Parallel edges are merged (kept), self-loops dropped. Edge ``length`` is
    preserved (min across parallel edges, falling back to ``1.0``). Node
    attributes (``x``, ``y``, …) are copied through.
    """
    g: nx.Graph[Any, Any] = nx.Graph()
    g.add_nodes_from(graph.nodes(data=True))
    for u, v, data in graph.edges(data=True):
        if u == v:
            continue
        length = data.get("length", 1.0)
        if g.has_edge(u, v):
            g[u][v]["length"] = min(g[u][v].get("length", math.inf), length)
        else:
            g.add_edge(u, v, length=length)
    return g


def _bike_lane_subgraph(graph: _NXGraph) -> nx.Graph[Any, Any]:
    """Return the undirected ``nx.Graph`` of edges tagged ``bike_lane='yes'``.

    This is the subgraph the §5.7 coverage and §3.6 fragmentation metrics are
    defined over. Nodes with no incident bike-lane edge are not included.
    """
    g: nx.Graph[Any, Any] = nx.Graph()
    for u, v, data in graph.edges(data=True):
        if data.get("bike_lane") != "yes":
            continue
        if u == v:
            continue
        if not g.has_node(u):
            g.add_node(u, **graph.nodes[u])
        if not g.has_node(v):
            g.add_node(v, **graph.nodes[v])
        length = data.get("length", 1.0)
        if g.has_edge(u, v):
            g[u][v]["length"] = min(g[u][v].get("length", math.inf), length)
        else:
            g.add_edge(u, v, length=length)
    return g


def _to_pygraph(nx_undirected: nx.Graph[Any, Any]) -> tuple[rx.PyGraph, dict[int | str, int]]:
    """Convert an undirected ``nx.Graph`` to a rustworkx ``PyGraph``.

    Returns ``(pygraph, node_map)`` where ``node_map`` maps each original node
    id to its rustworkx node index (via the ``__networkx_node__`` payload
    attribute set by ``networkx_converter``).
    """
    pyg: rx.PyGraph = rx.networkx_converter(nx_undirected, keep_attributes=True)  # type: ignore[assignment]
    node_map: dict[int | str, int] = {}
    for idx in pyg.node_indices():
        node_map[pyg[idx]["__networkx_node__"]] = idx
    return pyg, node_map


# ── Pure metric functions ────────────────────────────────────────────────


def connectivity(graph: _NXGraph, cfg: Config) -> float:
    """Transitivity of the largest connected component (RECREATE_SPEC §3.6).

    The graph is simplified to an undirected simple graph; its largest
    connected component is converted to a rustworkx ``PyGraph`` and scored
    with ``rx.transitivity``. Returns ``0.0`` when the largest component has
    fewer than 3 nodes (transitivity is undefined/trivial there).

    Note: the original spec mentions sampled ``local_clustering_coefficient``
    for large graphs; that function is **not available** in the installed
    rustworkx, so global transitivity is used regardless of graph size. The
    incremental :class:`MetricsState` caches this on a version counter
    (§5.14) so it is not recomputed every step.

    Args:
        graph: The bike network graph (may include added ``bike_lane='yes'``
            edges).
        cfg: Config (sampling thresholds are read for consistency but do not
            affect transitivity — see note above).

    Returns:
        Transitivity in ``[0, 1]``.
    """
    simp = _simplified_undirected(graph)
    if simp.number_of_nodes() == 0:
        return 0.0
    components = list(nx.connected_components(simp))
    largest = max(components, key=len)
    if len(largest) < 3:
        return 0.0
    sub = simp.subgraph(largest).copy()
    pyg, _ = _to_pygraph(sub)
    return float(rx.transitivity(pyg))


def _sample_sources(n: int, cfg: Config) -> int:
    """Return the number of Dijkstra source nodes to sample for ``n`` nodes.

    Tiers from ``cfg.sampling_thresholds`` (small/medium/large/huge). For
    ``n`` up to ``small`` use all ``n`` nodes; up to ``medium`` use
    ``sqrt(n)``; above that use ``log2(n) + 1`` (``cfg.path_sample_formula``
    selects ``"log"`` vs ``"sqrt"`` — ``"log"`` = log2, ``"sqrt"`` = sqrt).
    Always at least 1. Deterministic given ``cfg.seed``.
    """
    if n <= cfg.sampling_thresholds.get("small", 100):
        return n
    formula = cfg.path_sample_formula
    if n <= cfg.sampling_thresholds.get("medium", 1000):
        return max(1, int(math.isqrt(n))) if formula == "sqrt" else max(1, int(math.log2(n)) + 1)
    if n <= cfg.sampling_thresholds.get("large", 5000):
        return max(1, int(math.isqrt(n))) if formula == "sqrt" else max(1, int(math.log2(n)) + 1)
    return max(1, int(math.log2(n)) + 1) if formula == "log" else max(1, int(math.isqrt(n)))


def path_efficiency(graph: _NXGraph, cfg: Config) -> float:
    """``1 / (1 + avg_shortest_path_length_metres / 1000)`` (§3.6).

    Computed on the largest connected component of the simplified undirected
    graph. Source nodes are sampled for large components (see
    :func:`_sample_sources`); distances to all reachable targets are collected
    via ``rx.dijkstra_shortest_path_lengths``. Returns ``1.0`` when the
    component has fewer than 2 nodes (no pairs).

    Args:
        graph: The bike network graph.
        cfg: Config (sampling thresholds, ``path_sample_formula``, ``seed``).

    Returns:
        Path efficiency in ``(0, 1]``; higher = shorter average paths.
    """
    simp = _simplified_undirected(graph)
    if simp.number_of_nodes() < 2:
        return 1.0
    components = list(nx.connected_components(simp))
    largest = max(components, key=len)
    if len(largest) < 2:
        return 1.0
    sub = simp.subgraph(largest).copy()
    pyg, node_map = _to_pygraph(sub)
    n = pyg.num_nodes()
    k = min(n, _sample_sources(n, cfg))
    rng = random.Random(cfg.seed)
    sources = rng.sample(list(range(n)), k) if k < n else list(range(n))
    lengths: list[float] = []
    for src in sources:
        dists = rx.dijkstra_shortest_path_lengths(
            pyg, src, edge_cost_fn=lambda w: float(w.get("length", 1.0))
        )
        for tgt, d in dists.items():
            if tgt != src:
                lengths.append(float(d))
    if not lengths:
        return 1.0
    avg = sum(lengths) / len(lengths)
    return 1.0 / (1.0 + avg / 1000.0)


def fragmentation(graph: _NXGraph, cfg: Config) -> float:
    """Fragmentation of the bike-lane subgraph (RECREATE_SPEC §3.6).

    Computed only over edges tagged ``bike_lane='yes'``:
    ``0.3*min(1,(n_comp-1)/10) + 0.4*(1 - largest_cc_fraction) +
    0.3*isolation_factor`` where ``isolation_factor`` is the fraction of
    bike-lane-subgraph nodes that sit in singleton components. Returns ``0.0``
    when there are no bike-lane edges (no network → no fragmentation). Lower
    is better; range ``[0, 1]``.

    Args:
        graph: The bike network graph.
        cfg: Config (unused here but kept for a uniform signature; the
            incremental :class:`MetricsState` uses a union-find for this).

    Returns:
        Fragmentation in ``[0, 1]``.
    """
    sub = _bike_lane_subgraph(graph)
    if sub.number_of_nodes() == 0:
        return 0.0
    total = sub.number_of_nodes()
    components = list(nx.connected_components(sub))
    n_comp = len(components)
    largest = max(len(c) for c in components)
    isolated = sum(1 for c in components if len(c) == 1)
    return (
        0.3 * min(1.0, (n_comp - 1) / 10.0) + 0.4 * (1.0 - largest / total) + 0.3 * isolated / total
    )


def _metres_between(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Equirectangular distance in metres between two node dicts with x/y.

    Uses ``x`` (lon) / ``y`` (lat) in degrees. Good enough for city-scale
    coverage radii (§5.7 proxy). Nodes missing ``x``/``y`` return ``math.inf``
    so they are never counted as covered by radius.
    """
    if "x" not in a or "y" not in a or "x" not in b or "y" not in b:
        return math.inf
    lat = math.radians((float(a["y"]) + float(b["y"])) / 2.0)
    dx = (float(a["x"]) - float(b["x"])) * 111_320.0 * math.cos(lat)
    dy = (float(a["y"]) - float(b["y"])) * 111_320.0
    return math.hypot(dx, dy)


def coverage(graph: _NXGraph, cfg: Config) -> float:
    """Bike-lane-reachable population proxy (RECREATE_SPEC §5.7 fix).

    ``0.7 * coverage_ratio + 0.3 * largest_component_ratio`` where:
      - ``coverage_ratio`` = fraction of all graph nodes that are
        bike-lane-reachable. In ``coverage_mode == "radius"`` a node is
        reachable if it is within ``cfg.coverage_radius_m`` (Euclidean,
        equirectangular) of any endpoint of a ``bike_lane='yes'`` edge. In
        ``coverage_mode == "component"`` a node is reachable if it lies in a
        connected component of the whole (undirected) graph that contains at
        least one bike-lane node.
      - ``largest_component_ratio`` = size of the largest connected component
        of the bike-lane subgraph divided by the **total graph node count**
        (so the metric moves as the bike-lane network grows — §5.7).

    Returns ``0.0`` when the graph has no nodes or no bike-lane edges. This is
    the sensitive replacement for the original near-constant
    ``_calculate_population_served``; a test pins that adding a bike-lane edge
    measurably increases the value.

    Args:
        graph: The bike network graph.
        cfg: Config (``coverage_mode``, ``coverage_radius_m``).

    Returns:
        Coverage in ``[0, 1]``.
    """
    total = graph.number_of_nodes()
    if total == 0:
        return 0.0
    bike_nodes: set[int | str] = {
        n for u, v, d in graph.edges(data=True) if d.get("bike_lane") == "yes" for n in (u, v)
    }
    if not bike_nodes:
        return 0.0

    if cfg.coverage_mode == "component":
        simp = _simplified_undirected(graph)
        covered: set[int | str] = set()
        for comp in nx.connected_components(simp):
            if comp & bike_nodes:
                covered |= comp
    else:  # "radius" (default)
        covered = set()
        for n, ndata in graph.nodes(data=True):
            for bn in bike_nodes:
                if _metres_between(ndata, graph.nodes[bn]) <= cfg.coverage_radius_m:
                    covered.add(n)
                    break

    coverage_ratio = len(covered) / total
    sub = _bike_lane_subgraph(graph)
    if sub.number_of_nodes() == 0:
        largest_ratio = 0.0
    else:
        largest = max(len(c) for c in nx.connected_components(sub))
        largest_ratio = largest / total
    return 0.7 * coverage_ratio + 0.3 * largest_ratio


# ── Incremental data structures (§6.2 / §6.3 / §5.14) ────────────────────


class UnionFind:
    """Union-find over node ids with component-size tracking (§6.3).

    Supports O(α(N)) union and a constant-time view of component count, the
    largest component size, and the isolated (singleton) node count — exactly
    the quantities :func:`fragmentation` needs.
    """

    def __init__(self, nodes: list[int | str]) -> None:
        self._parent: dict[int | str, int | str] = {n: n for n in nodes}
        self._size: dict[int | str, int] = {n: 1 for n in nodes}
        self._n_comp = len(nodes)
        self._largest = 1 if nodes else 0
        self._isolated = len(nodes)

    def add_node(self, n: int | str) -> None:
        """Add a new node if not already present."""
        if n in self._parent:
            return
        self._parent[n] = n
        self._size[n] = 1
        self._n_comp += 1
        self._largest = max(self._largest, 1)
        self._isolated += 1

    def find(self, x: int | str) -> int | str:
        """Find the root of ``x`` with path compression."""
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: int | str, b: int | str) -> None:
        """Merge the components of ``a`` and ``b`` (no-op if already joined)."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # union by size
        if self._size[ra] < self._size[rb]:
            ra, rb = rb, ra
        # rb was a singleton? track isolated count
        if self._size[rb] == 1:
            self._isolated -= 1
        if self._size[ra] == 1:
            self._isolated -= 1
        self._parent[rb] = ra
        self._size[ra] += self._size[rb]
        self._n_comp -= 1
        self._largest = max(self._largest, self._size[ra])

    @property
    def num_components(self) -> int:
        return self._n_comp

    @property
    def largest_size(self) -> int:
        return self._largest

    @property
    def isolated_count(self) -> int:
        return self._isolated


class MetricsState:
    """Incremental metrics cache (RECREATE_SPEC §6.2, §6.3, §5.14).

    Wraps a growing bike graph (the env adds edges via :meth:`add_edge`) and
    exposes the same four scalars as the pure functions, but maintains a
    persistent rustworkx ``PyGraph`` and a union-find so per-step work is
    cheap. Connectivity and path-efficiency are cached on ``graph_version``
    and only recomputed when the graph changes; fragmentation and coverage are
    derived incrementally.

    Public accessors return values **equal** to the pure functions on the
    current graph (pinned by tests).
    """

    def __init__(self, graph: _NXGraph, cfg: Config) -> None:
        self._graph = graph.copy()
        self._cfg = cfg
        self._version = 0
        # Persistent undirected PyGraph of the simplified whole graph (§6.2).
        simp = _simplified_undirected(self._graph)
        self._pyg, self._node_map = _to_pygraph(simp)
        # Union-find over bike-lane-subgraph nodes (§6.3).
        sub = _bike_lane_subgraph(self._graph)
        self._uf = UnionFind(list(sub.nodes()))
        for u, v in sub.edges():
            self._uf.union(u, v)
        self._bike_nodes: set[int | str] = {
            n
            for u, v, d in self._graph.edges(data=True)
            if d.get("bike_lane") == "yes"
            for n in (u, v)
        }
        self._sub_node_count = sub.number_of_nodes()
        self._cache: dict[str, tuple[int, float]] = {}

    def add_edge(self, u: int | str, v: int | str, data: dict[str, Any]) -> None:
        """Record an added edge and invalidate cached metrics (§5.14).

        Args:
            u: First endpoint node id.
            v: Second endpoint node id.
            data: Edge attribute dict (must contain ``length``; ``bike_lane``
                is honoured if ``"yes"``).
        """
        if u == v:
            return
        self._graph.add_edge(u, v, **data)
        # update persistent PyGraph (whole-graph view)
        for n in (u, v):
            if n not in self._node_map:
                idx = self._pyg.add_node({**self._graph.nodes[n], "__networkx_node__": n})
                self._node_map[n] = idx
        length = float(data.get("length", 1.0))
        self._pyg.add_edge(self._node_map[u], self._node_map[v], length)
        # update bike-lane structures if this is a bike-lane edge
        if data.get("bike_lane") == "yes":
            for n in (u, v):
                self._uf.add_node(n)
                if n not in self._bike_nodes:
                    self._bike_nodes.add(n)
                    self._sub_node_count += 1
            self._uf.union(u, v)
        self._version += 1

    def connectivity(self) -> float:
        """Cached transitivity of the largest component (§5.14)."""
        cached = self._cache.get("conn")
        if cached is not None and cached[0] == self._version:
            return cached[1]
        val = connectivity(self._graph, self._cfg)
        self._cache["conn"] = (self._version, val)
        return val

    def path_efficiency(self) -> float:
        """Cached path efficiency (§5.14)."""
        cached = self._cache.get("pe")
        if cached is not None and cached[0] == self._version:
            return cached[1]
        val = path_efficiency(self._graph, self._cfg)
        self._cache["pe"] = (self._version, val)
        return val

    def fragmentation(self) -> float:
        """Incremental fragmentation from the union-find (§6.3)."""
        total = self._sub_node_count
        if total == 0:
            return 0.0
        n_comp = self._uf.num_components
        largest = self._uf.largest_size
        isolated = self._uf.isolated_count
        return (
            0.3 * min(1.0, (n_comp - 1) / 10.0)
            + 0.4 * (1.0 - largest / total)
            + 0.3 * isolated / total
        )

    def coverage(self) -> float:
        """Incremental coverage from the bike-lane node set (§5.7 fix)."""
        return coverage(self._graph, self._cfg)
