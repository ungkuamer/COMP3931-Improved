# Plan 003: Implement `metrics.py` (connectivity, path efficiency, fragmentation, coverage) with tests

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**:
> `git diff --stat a828293..HEAD -- bike_rl/metrics.py bike_rl/config.py tests/conftest.py tests/test_metrics.py`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: MED
- **Depends on**: plans/002-graph-utils-and-candidates.md (DONE — provides `graph_utils.nx_to_rx`, `Config` fields used here)
- **Category**: bug (§5.7 population-served signal is near-constant) + perf (§6.2 persistent rx graph, §6.3 union-find fragmentation, §6.7 configurable sampling) + tech-debt (modular rewrite of §3.6 metrics) + tests
- **Planned at**: commit `a828293`, 2026-06-22
- **Issue**: (not published)

## Why this matters

The original `nx-rx-simple-ur.py` computes four graph metrics every step, but
two of them are degenerate:

- **Population served is near-constant** (RECREATE_SPEC §5.7): it divides by
  `len(self.graph.nodes())` — the *whole* original bike graph plus a handful of
  added edges — so adding a bike lane barely moves the number. The RL agent
  therefore gets no learning signal for coverage, which is arguably the whole
  point of the project. The optimiser baseline (OPTIMIZER_SPEC) and the
  research comparison (RESEARCH_DIRECTION §4.3) both depend on a coverage
  metric that is *sensitive* to additions; without the fix, greedy/ILP/RL all
  report the same useless number and the §10 comparison table is meaningless.
- **Every metric is recomputed from scratch each step**: `_calculate_connectivity`
  rebuilds an `nx.Graph`, simplifies, and re-converts to rustworkx every call
  (§6.2); `_calculate_fragmentation` rebuilds a graph and runs
  `connected_components` every step, O(V+E) per step (§6.3). On a real city
  this is the training-loop bottleneck.

This plan builds `metrics.py` — the module both the RL env (plan 004) and the
direct optimisers (OPTIMIZER_SPEC §3, the `objective.py` plan) import. It
delivers: (1) **pure, deterministic** metric functions with the §5.7 coverage
fix, (2) an **incremental `MetricsState`** that maintains a persistent
rustworkx graph (§6.2) and a union-find (§6.3) with version-keyed caching
(§5.14) so the env/optimisers don't recompute from scratch, and (3)
`test_metrics.py` with known-answer fixtures and a sensitivity test pinning
the §5.7 fix. `objective.py` is **out of scope** here — it is the first
optimiser-side plan (OPTIMIZER_SPEC §11.1) and depends on this module's
function signatures being stable.

## Current state

The package was scaffolded (plan 001) and `graph_utils.py` / `candidates.py`
were implemented (plan 002). The relevant files:

- `bike_rl/metrics.py` — **stub**. Four functions all raise
  `NotImplementedError("Plan 003")`:
  ```python
  def connectivity(graph: _NXGraph, cfg: Config) -> float: ...
  def path_efficiency(graph: _NXGraph, cfg: Config) -> float: ...
  def fragmentation(graph: _NXGraph, cfg: Config) -> float: ...
  def coverage(graph: _NXGraph, cfg: Config) -> float: ...
  ```
  (`_NXGraph` is imported under `TYPE_CHECKING`; the stub uses
  `nx.MultiDiGraph[Any, Any, Any]`. Keep that convention — see
  `bike_rl/graph_utils.py` for the `else: _NXGraph = nx.MultiDiGraph` pattern
  that keeps mypy strict happy without a runtime import of the `Any` params.)
- `bike_rl/objective.py` — stub, raises `NotImplementedError`. **Do not
  implement it in this plan** (out of scope). It is listed here only because
  its future implementation will call `connectivity`, `coverage`,
  `fragmentation` from this module — so keep those signatures stable and pure.
- `bike_rl/config.py` — frozen `Config`, fully implemented. **Already has every
  field this plan needs** (verified): `coverage_radius_m: float = 300.0`,
  `coverage_mode: str = "radius"`, `sampling_thresholds: dict[str,int]` with
  keys `huge=10000, large=5000, medium=1000, small=100`,
  `clustering_sample_sizes: dict[str,int] = {"huge":40,"large":60}`,
  `path_sample_formula: str = "log"`, `seed: int = 0`. **Do not add fields.**
- `bike_rl/graph_utils.py` — implemented. `nx_to_rx(nx_graph) -> (PyDiGraph,
  node_map)` is available, but it returns a **directed** `PyDiGraph`. The
  undirected metrics below need a `PyGraph` (undirected) — build it with
  `rx.networkx_converter(nx_graph_undirected, keep_attributes=True)` (see
  "rustworkx API facts" below); do **not** reuse `nx_to_rx` for the undirected
  metrics.
- `tests/conftest.py` — has `tiny_bike_graph` (4 nodes, edges `(1,2)` and
  `(3,4)` both `bike_lane="yes"`, lengths 120/130, `x`/`y` coords ~0.001 apart),
  `tiny_walk_graph`, `mock_osm`. **Do not modify these existing fixtures.**
- `tests/test_smoke.py`, `tests/test_graph_utils.py`, `tests/test_candidates.py`
  — existing, 22 tests total, all green. Do not touch.

### Repo conventions to match (from plan 002, still in force)

- **Typing**: `from __future__ import annotations` at top of every module;
  `mypy --strict bike_rl` must stay clean. For `_NXGraph`, use the
  `if TYPE_CHECKING: _NXGraph = nx.MultiDiGraph[Any, Any, Any]` /
  `else: _NXGraph = nx.MultiDiGraph` split (see `bike_rl/candidates.py:14-19`).
  `pyproject.toml` already sets `ignore_missing_imports = true` for
  `rustworkx.*`, `osmnx.*`, so rustworkx calls won't block mypy.
- **Imports**: stdlib → third-party → local, blank-line separated (ruff `I`).
  Use `if TYPE_CHECKING:` for `Config` (metrics doesn't need it at runtime
  except for reading fields — `Config` is fine to import normally if needed,
  but follow `candidates.py` which guards it under `TYPE_CHECKING` and takes
  `cfg: Config` only in annotations).
- **Docstrings**: Google convention (ruff `D`, `convention = "google"`).
  Public functions/classes get docstrings. `D100/D102/D104/D105/D107` ignored.
- **No `print`**: use `logging` (`logger = logging.getLogger(__name__)`). No
  prints in `metrics.py`.
- **Frozen dataclasses** for value types. Plain `@dataclass` (not frozen) is
  fine for `MetricsState`/`UnionFind` since they are mutable by design.
- **Tests**: `pytest`, fixtures from `tests/conftest.py`, `monkeypatch` for
  mocks, no network access. Model test file structure on
  `tests/test_candidates.py` (top docstring, `from __future__ import
  annotations`, small focused functions with docstrings).
- **Commit style**: Conventional Commits with scope — `feat(metrics): …`,
  `test(metrics): …`, matching `git log` entries like
  `feat(graph): implement nx_to_rx, OSM loaders, cache`.

### rustworkx API facts (verified against the installed rustworkx — use exactly these)

The executor must use these exact call shapes; the API is picky:

- `rx.networkx_converter(nx_graph, keep_attributes=True)` returns a
  **`PyDiGraph`** if `nx_graph` is a `MultiDiGraph`/`DiGraph`, and a
  **`PyGraph`** (undirected) if `nx_graph` is a `Graph` (undirected). It does
  **not** accept a `weight_fn` argument (only `keep_attributes`). Node payloads
  become `{"__networkx_node__": <orig id>, ...attrs}`; edge payloads become
  the original edge data dict (so `payload.get("length", 1.0)` works).
- `rx.transitivity(graph)` → float in `[0,1]`. Works on both `PyGraph` and
  `PyDiGraph`; **use it on an undirected `PyGraph`** for the connectivity
  metric.
- `rx.connected_components(pygraph)` → `list[set[int]]` of node-index sets.
  **Requires a `PyGraph`** (raises `TypeError` on `PyDiGraph`).
- `rx.number_connected_components(pygraph)` → int.
- `rx.dijkstra_shortest_path_lengths(graph, node, edge_cost_fn, goal=None)`
  → `PathLengthMapping` `{target_idx: length}`. **The third positional
  argument is `edge_cost_fn`** (a callable taking the edge *weight/payload*,
  e.g. `lambda w: w.get("length", 1.0)`), NOT `weight_fn`. `node` is a
  rustworkx **node index** (int), not an OSM id — map via the node_map you
  build from `payload["__networkx_node__"]`.
- There is **no `rx.local_clustering_coefficient`** in this version. The
  original spec §3.6 mentions "sampled `local_clustering_coefficient` for large
  graphs" — that function does not exist. Use `rx.transitivity` on the largest
  connected component only; do not attempt sampled local clustering. Document
  this deviation in the `connectivity` docstring.

### Intent docs to honor (quoted so you don't have to re-read them)

From `RECREATE_SPEC.md` §3.6 (metric definitions):
> **Connectivity** = normalised clustering/transitivity of the largest
> connected component of the (simplified, undirected) graph; uses rustworkx
> `transitivity` or sampled `local_clustering_coefficient` for large graphs.
> Cached by edge count.
> **Path efficiency** = `1 / (1 + avg_path_length/1000)`, where avg path
> length is from sampled-source Dijkstra. Aggressive sampling by graph size.
> **Fragmentation** (NetworkX, recomputed every step) =
> `0.3*min(1,(n_comp-1)/10) + 0.4*(1 - largest_cc_fraction) + 0.3*isolation_factor`
> on the bike-lane-only subgraph.
> **Population served** = `0.7*coverage_ratio + 0.3*largest_component_ratio`
> (currently near-constant — §5.7).

From `RECREATE_SPEC.md` §5.7 (the bug this plan fixes):
> `_calculate_population_served` is near-constant … counts the **whole
> original** bike graph plus a few added edges, so coverage barely moves. The
> signal the agent needs is missing. **Fix**: base population served on
> **bike-lane-reachable** nodes (e.g. nodes within `X` metres of any bike
> lane, or nodes in the same connected component as a bike lane), not on total
> graph node count. Document the chosen proxy in `config.py`.

From `RECREATE_SPEC.md` §5.14 (cache versioning):
> use a monotonic `self.graph_version` counter incremented on every edge add;
> cache keyed on that.

From `RECREATE_SPEC.md` §6.2:
> keep one `rx_simple_graph` updated incrementally on edge add; recompute
> connected components on it directly.

From `RECREATE_SPEC.md` §6.3:
> maintain a union-find over bike-lane endpoints; component counts/largest-
> component size update in O(α(N)) per edge add.

From `RECREATE_SPEC.md` §6.7:
> Move [graph-size thresholds and sample counts] to `Config` with sane
> defaults and document the bias/variance tradeoff. (Already done in plan 001
> — just *read* them from `Config` here.)

From `OPTIMIZER_SPEC.md` §3.1 (the downstream consumer of these metrics):
> `objective(graph, added_edges, weights, cfg)` … Components come from
> metrics.py (shared with the RL env): connectivity(graph) in [0,1];
> coverage(graph) in [0,1] (the FIXED population-served, §5.7);
> fragmentation(graph) in [0,1] (lower is better).
> … provide an `objective_delta` … backed by the incremental data structures
> in `metrics.py` (union-find for fragmentation, persistent rx graph for
> connectivity).

From `RESEARCH_DIRECTION.md` §4.3:
> Redefine coverage as *bike-lane-reachable population* (nodes within `X`
> metres of a bike lane, or in the same connected component as one) so the
> metric is **sensitive** enough to distinguish policies. Validate
> sensitivity: on a tiny fixture, adding an edge must produce a measurable
> objective change.

## Commands you will need

| Purpose    | Command                                                                 | Expected on success |
|------------|-------------------------------------------------------------------------|---------------------|
| Install    | `pip install -e ".[dev]"`                                               | exit 0              |
| Ruff lint  | `ruff check bike_rl/metrics.py tests/test_metrics.py`                   | exit 0, no errors   |
| Ruff fmt   | `ruff format --check bike_rl tests`                                     | exit 0              |
| Mypy       | `mypy --strict bike_rl`                                                 | exit 0, no errors   |
| Tests      | `pytest -q tests/test_metrics.py`                                       | all pass            |
| Full suite | `pytest -q`                                                             | all pass (≥22 prior + new) |
| Coverage   | `pytest --cov=bike_rl --cov-report=term-missing tests/test_metrics.py`  | exit 0; `bike_rl/metrics.py` ≥80% line coverage |

## Scope

**In scope** (the only files you should modify or create):
- `bike_rl/metrics.py` — full implementation: four pure metric functions,
  `UnionFind`, `MetricsState`, and any private helpers.
- `tests/test_metrics.py` — create.

**Out of scope** (do NOT touch, even though they look related):
- `bike_rl/objective.py` — the optimiser-side plan implements it; it will
  *import* `connectivity`/`coverage`/`fragmentation` from `metrics.py`. Do
  not implement `objective`, `objective_delta`, or `apply_added_edges`. Do
  not change their stubs.
- `bike_rl/env.py`, `bike_rl/training.py`, `bike_rl/evaluation.py`,
  `bike_rl/plotting.py`, `bike_rl/cli.py`, anything under `bike_rl/optim/` —
  plans 004+. Do not wire `metrics` into them.
- `bike_rl/config.py` — **do not modify**. All needed fields already exist
  (verified above). If you believe a field is missing, STOP and report — do
  not add it.
- `bike_rl/graph_utils.py`, `bike_rl/candidates.py` — done in plan 002. Do
  not edit. You may *import* from `graph_utils` if useful, but the undirected
  `PyGraph` construction below is self-contained and probably won't need it.
- `tests/conftest.py` — do not modify existing fixtures. You may add new
  fixtures **inside `tests/test_metrics.py`** (local fixtures are fine and
  keep conftest stable).
- `pyproject.toml`, `requirements*.txt`, CI, `.gitignore`, other test files.

## Git workflow

- Branch: `advisor/003-metrics-and-tests`
- Commit per logical unit (suggested: one for pure functions, one for
  `UnionFind`+`MetricsState`, one for tests). Conventional Commits with scope:
  `feat(metrics): implement pure metric functions with §5.7 coverage fix`,
  `feat(metrics): add incremental MetricsState + UnionFind (§6.2/§6.3/§5.14)`,
  `test(metrics): add known-answer + sensitivity + incremental tests`.
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Pure helpers — simplify, bike-lane subgraph, undirected PyGraph

In `bike_rl/metrics.py`, add private helpers used by every metric. None of
these are exported (leading underscore); they exist to keep the four public
functions short and to centralise the "what is the bike-lane subgraph" /
"build an undirected PyGraph" logic.

Target shape (adapt imports to satisfy ruff `I` ordering):

```python
"""Graph metrics: connectivity, path efficiency, fragmentation, coverage.

See RECREATE_SPEC.md §3.6, §5.7, §6.2, §6.3, §6.7.  The four public functions
are pure and deterministic; :class:`MetricsState` provides an incremental
cache used by the RL env (plan 004) and the optimisers (OPTIMIZER_SPEC §3).
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import networkx as nx
import rustworkx as rx

if TYPE_CHECKING:
    from bike_rl.config import Config

    _NXGraph = nx.MultiDiGraph[Any, Any, Any]
else:
    _NXGraph = nx.MultiDiGraph

logger = logging.getLogger(__name__)


def _simplified_undirected(graph: _NXGraph) -> nx.Graph:
    """Return a simple undirected ``nx.Graph`` view of ``graph``.

    Parallel edges are merged (kept), self-loops dropped. Edge ``length`` is
    preserved (min across parallel edges, falling back to ``1.0``). Node
    attributes (``x``, ``y``, …) are copied through.
    """
    g = nx.Graph()
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


def _bike_lane_subgraph(graph: _NXGraph) -> nx.Graph:
    """Return the undirected ``nx.Graph`` of edges tagged ``bike_lane='yes'``.

    This is the subgraph the §5.7 coverage and §3.6 fragmentation metrics are
    defined over. Nodes with no incident bike-lane edge are not included.
    """
    g = nx.Graph()
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


def _to_pygraph(nx_undirected: nx.Graph) -> tuple[rx.PyGraph, dict[int | str, int]]:
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
```

**Verify**:
- `ruff check bike_rl/metrics.py` → exit 0
- `ruff format --check bike_rl/metrics.py` → exit 0 (run `ruff format bike_rl/metrics.py` if it reformats)
- `mypy --strict bike_rl` → exit 0
- `python -c "from bike_rl.metrics import _simplified_undirected, _bike_lane_subgraph, _to_pygraph; print('ok')"` → prints `ok`

### Step 2: `connectivity` and `path_efficiency` (pure)

Implement the two metrics that operate on the **whole** graph (simplified,
undirected, largest connected component).

```python
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
    import random as _random

    rng = _random.Random(cfg.seed)
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
```

Notes:
- `rx.dijkstra_shortest_path_lengths`'s third positional arg is `edge_cost_fn`
  (a callable on the edge payload). The payload is the edge data dict, so
  `lambda w: float(w.get("length", 1.0))` is correct. **Do not** name it
  `weight_fn` — that raises `TypeError`.
- `random.Random(cfg.seed)` keeps sampling deterministic (RESEARCH_DIRECTION
  §4.4: deterministic solvers; OPTIMIZER_SPEC §5 "Deterministic"). `random` is
  imported locally to avoid a top-level import that ruff might reorder oddly —
  a top-level `import random` is also fine; pick one and let ruff sort it.

**Verify**:
- `ruff check bike_rl/metrics.py` → exit 0
- `ruff format --check bike_rl/metrics.py` → exit 0
- `mypy --strict bike_rl` → exit 0
- `python -c "from bike_rl.metrics import connectivity, path_efficiency; from bike_rl.config import Config; import networkx as nx; g=nx.MultiDiGraph(); g.add_edge(1,2,length=120.0); g.add_edge(2,3,length=120.0); g.add_edge(3,1,length=120.0); print(round(connectivity(g, Config()),3), round(path_efficiency(g, Config()),3))"` → prints two floats in `[0,1]` (transitivity `1.0` for a triangle; efficiency `~0.89` for 120 m avg).

### Step 3: `fragmentation` and `coverage` (pure, with the §5.7 fix)

`fragmentation` is defined on the **bike-lane subgraph** (edges with
`bike_lane='yes'`). `coverage` is the §5.7 fix — sensitive to additions.

```python
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
        0.3 * min(1.0, (n_comp - 1) / 10.0)
        + 0.4 * (1.0 - largest / total)
        + 0.3 * isolated / total
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
```

Notes:
- `cfg` is unused in `fragmentation` — keep the parameter for signature
  uniformity (the env/optimisers call all four with `(graph, cfg)`). ruff
  won't complain about an unused parameter; if it does (it shouldn't), rename
  to `_cfg` only as a last resort — prefer keeping `cfg` for the uniform call
  site.
- The `set` comprehension for `bike_nodes` flattens `(u, v)` per edge; ruff
  may want it on fewer lines — run `ruff format` and accept its layout.
- `_metres_between`: 1° lat ≈ 111 320 m; lon scaled by `cos(lat)`. Synthetic
  fixtures use 0.001° deltas ≈ 111 m, well under the default 300 m radius, so
  coverage is non-trivial and *moves* when a new node is added (see tests).

**Verify**:
- `ruff check bike_rl/metrics.py` → exit 0
- `ruff format --check bike_rl/metrics.py` → exit 0
- `mypy --strict bike_rl` → exit 0
- `python -c "from bike_rl.metrics import fragmentation, coverage; from tests.conftest import *; import networkx as nx; g=nx.MultiDiGraph(); 
import networkx as nx
g=nx.MultiDiGraph()
g.add_node(1,x=0.0,y=0.0); g.add_node(2,x=0.001,y=0.0); g.add_node(3,x=0.0,y=0.001); g.add_node(4,x=0.001,y=0.001)
g.add_edge(1,2,length=120.0,bike_lane='yes'); g.add_edge(3,4,length=130.0,bike_lane='yes')
from bike_rl.metrics import fragmentation, coverage
from bike_rl.config import Config
print(round(fragmentation(g,Config()),3), round(coverage(g,Config()),3))
"` → prints two floats; `fragmentation` is `> 0` (two components of size 2) and `coverage` is `< 1` (largest bike-lane component is 2 of 4 nodes → `0.7*1.0 + 0.3*0.5 = 0.85`).

(If the one-liner above is awkward, write it to `/tmp/probe.py` and run `python /tmp/probe.py`; the expected output is the same.)

### Step 4: `UnionFind` + `MetricsState` (incremental, §6.2 / §6.3 / §5.14)

Add the incremental structures that the env (plan 004) and the optimisers'
`objective_delta` (OPTIMIZER_SPEC §3.1) will use. The contract: every
accessor on `MetricsState` returns a value **equal** to the corresponding pure
function applied to the current graph — tests pin this.

```python
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
```

Now `MetricsState`. It holds: (a) the working `nx.MultiDiGraph` (so the pure
functions can be re-run lazily if needed), (b) a persistent undirected
`rx.PyGraph` of the simplified whole graph for connectivity/efficiency
(§6.2), (c) a `UnionFind` over bike-lane endpoints for fragmentation (§6.3),
(d) the bike-lane node set for coverage, (e) a `graph_version` counter
(§5.14) that invalidates cached connectivity/efficiency. Fragmentation and
coverage are recomputed from the union-find / node set in O(1)-ish each call.

```python
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
            n for u, v, d in self._graph.edges(data=True)
            if d.get("bike_lane") == "yes" for n in (u, v)
        }
        self._sub_node_count = sub.number_of_nodes()
        self._cache: dict[str, tuple[int, float]] = {}

    def add_edge(self, u: int | str, v: int | str, data: dict[str, Any]) -> None:
        """Record an added edge and invalidate cached metrics (§5.14).

        Args:
            u, v: Endpoint node ids.
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
```

Notes for the executor:
- `MetricsState.connectivity`/`path_efficiency` still call the pure functions
  under the hood (transitivity and sampled Dijkstra are not cheap to maintain
  truly incrementally); the win is the **version cache** so they are not
  recomputed when the env reads the observation multiple times per step, and
  the persistent `PyGraph` avoids the rebuild the original did every call
  (§6.2). `fragmentation` and `coverage`'s largest-component term are the
  ones that become truly O(1)-ish via union-find.
- `add_edge` adds to the `PyGraph` by re-using `self._node_map`; new nodes get
  a payload matching what `networkx_converter` would have produced
  (`{"__networkx_node__": n, ...attrs}`). This keeps `_to_pygraph`-built and
  manually-added nodes consistent.
- `coverage` currently delegates to the pure function (it is cheap enough on
  the bike-lane node set; the radius scan is O(|all nodes| × |bike nodes|),
  acceptable for the env). If a later plan needs it faster, add a spatial
  index then — out of scope now.

**Verify**:
- `ruff check bike_rl/metrics.py` → exit 0
- `ruff format --check bike_rl/metrics.py` → exit 0
- `mypy --strict bike_rl` → exit 0
- `python -c "from bike_rl.metrics import UnionFind, MetricsState; print('ok')"` → prints `ok`
- `python -c "
import networkx as nx
from bike_rl.metrics import MetricsState, fragmentation, coverage
from bike_rl.config import Config
g=nx.MultiDiGraph()
for n,(x,y) in {1:(0.,0.),2:(0.001,0.),3:(0.,0.001),4:(0.001,0.001),5:(0.002,0.002)}.items():
    g.add_node(n,x=x,y=y)
g.add_edge(1,2,length=120.0,bike_lane='yes'); g.add_edge(3,4,length=130.0,bike_lane='yes')
st=MetricsState(g, Config())
assert abs(st.fragmentation()-fragmentation(g,Config()))<1e-9
assert abs(st.coverage()-coverage(g,Config()))<1e-9
st.add_edge(2,5,{'length':200.0,'bike_lane':'yes'})
assert abs(st.fragmentation()-fragmentation(g,Config()))<1e-9, (st.fragmentation(), fragmentation(g,Config()))
print('incremental matches pure')
"` → prints `incremental matches pure` (the assertion compares `MetricsState` against the pure functions after an `add_edge`). **Important**: `MetricsState.__init__` copies the graph, so to compare after `add_edge` you must apply the *same* edge to `g` first — the snippet above adds the edge to `g` *via* `st.add_edge`, which mutates `st`'s internal copy, not `g`. Fix the probe: after `st.add_edge(2,5,{...})`, also do `g.add_edge(2,5,length=200.0,bike_lane='yes')` before comparing. Apply that correction when you run the probe. The expected result is still `incremental matches pure`.

### Step 5: Write `tests/test_metrics.py`

Create `tests/test_metrics.py`. Model the structure on
`tests/test_candidates.py` (top docstring, `from __future__ import
annotations`, small focused functions with docstrings, fixtures from
`conftest.py` plus any local fixtures). Use `tiny_bike_graph` and
`tiny_walk_graph` from `conftest.py`; define additional small graphs **as
local fixtures** inside the test file (do not edit `conftest.py`).

Local fixtures to define in the test file:

```python
@pytest.fixture
def triangle_graph() -> nx.MultiDiGraph:
    """3-node directed triangle, length 120 each, no bike_lane tags."""
    g = nx.MultiDiGraph()
    for n, (x, y) in {1: (0.0, 0.0), 2: (0.001, 0.0), 3: (0.0005, 0.001)}.items():
        g.add_node(n, x=x, y=y)
    g.add_edge(1, 2, length=120.0)
    g.add_edge(2, 3, length=120.0)
    g.add_edge(3, 1, length=120.0)
    return g


@pytest.fixture
def two_component_bike_graph() -> nx.MultiDiGraph:
    """5 nodes: bike-lane edges (1,2) and (3,4); node 5 unconnected.
    bike_lane='yes' on both edges. Used for fragmentation/coverage known
    answers and the §5.7 sensitivity test.
    """
    g = nx.MultiDiGraph()
    for n, (x, y) in {
        1: (0.0, 0.0), 2: (0.001, 0.0), 3: (0.0, 0.001),
        4: (0.001, 0.001), 5: (0.002, 0.002),
    }.items():
        g.add_node(n, x=x, y=y)
    g.add_edge(1, 2, length=120.0, bike_lane="yes")
    g.add_edge(3, 4, length=130.0, bike_lane="yes")
    return g
```

Test cases (write all; each is a short function with a docstring):

1. `test_connectivity_triangle_is_one(triangle_graph)` —
   `connectivity(triangle_graph, Config()) == 1.0` (a triangle has
   transitivity 1.0).
2. `test_connectivity_small_component_returns_zero()` — a graph with a largest
   component of 2 nodes (single edge) returns `0.0` (transitivity undefined
   below 3 nodes).
3. `test_path_efficiency_in_range(tiny_bike_graph)` —
   `0.0 < path_efficiency(tiny_bike_graph, Config()) <= 1.0`; with 120 m
   edges the value is `1/(1+0.12)` ≈ `0.89` — assert it is `> 0.8` and `< 1.0`.
4. `test_path_efficiency_deterministic(tiny_bike_graph)` — call twice, assert
   equal (deterministic sampling via `cfg.seed`).
5. `test_fragmentation_two_components(two_component_bike_graph)` — two
   components of size 2 each, 5 total nodes (node 5 has no bike-lane edge so
   it is **not** in the bike-lane subgraph; subgraph has 4 nodes, 2
   components, largest 2, isolated 0). Expected:
   `0.3*min(1,1/10) + 0.4*(1-2/4) + 0.3*0 = 0.03 + 0.2 + 0 = 0.23`. Assert
   `abs(fragmentation(...) - 0.23) < 1e-9`.
6. `test_fragmentation_empty_bike_lane_returns_zero(triangle_graph)` — no
   `bike_lane='yes'` edges → returns `0.0`.
7. `test_coverage_is_sensitive_to_added_edge(two_component_bike_graph)` —
   **the §5.7 regression test**. Compute `before = coverage(g, Config())`.
   Add edge `(2, 5, length=200.0, bike_lane='yes')` to `g` (node 5 was
   uncovered). Compute `after = coverage(g, Config())`. Assert `after >
   before` (the metric moves — the whole point of the §5.7 fix). Also assert
   `before < 1.0` and `after < 1.0`. Document in the docstring that this pins
   RECREATE_SPEC §5.7 / RESEARCH_DIRECTION §4.3.
8. `test_coverage_radius_mode_covers_nearby_nodes(two_component_bike_graph)` —
   with `coverage_mode='radius'` (default) and `coverage_radius_m=300`,
   nodes 1–4 (within ~111 m of a bike-lane endpoint) are covered; node 5 is
   ~285 m from node 2/4 → also within 300 m, so before adding any edge node 5
   is covered too. Assert `coverage_ratio` portion yields a value `> 0.7`
   (i.e. all 5 nodes covered by radius → `0.7*1.0 + 0.3*0.4 = 0.82`). Adjust
   the exact expected value to match your implementation; the assertion that
   matters is "all nodes within 300 m are covered" and "value is `> 0` and
   `< 1.0`".
9. `test_coverage_component_mode(two_component_bike_graph)` — build a
   `Config(coverage_mode='component')`; in the simplified undirected graph,
   components are `{1,2}`, `{3,4}`, `{5}`; only components containing a
   bike-lane node count → `{1,2}` and `{3,4}` = 4 of 5 nodes covered.
   Assert `0.4 <= coverage(g, cfg_component) <= 0.6` (4/5 covered →
   `0.7*0.8 + 0.3*0.4 = 0.68`; adjust to your exact value; the key check is
   it differs from radius mode and is `< 1.0`).
10. `test_coverage_no_bike_lane_returns_zero(triangle_graph)` — no
    `bike_lane='yes'` edges → `coverage(...) == 0.0`.
11. `test_union_find_basic()` — `uf = UnionFind([1,2,3,4])`; assert
    `num_components == 4`, `largest_size == 1`, `isolated_count == 4`; `uf.union(1,2)`;
    assert `num_components == 3`, `largest_size == 2`, `isolated_count == 2`;
    `uf.union(3,4)`; `uf.union(2,3)`; assert `num_components == 1`,
    `largest_size == 4`, `isolated_count == 0`; `uf.find(1) == uf.find(4)`.
12. `test_union_find_add_node()` — start `UnionFind([1])`; `uf.add_node(2)`;
    assert `num_components == 2`; `uf.add_node(2)` again is a no-op
    (`num_components` unchanged).
13. `test_metrics_state_matches_pure(two_component_bike_graph)` —
    `st = MetricsState(g, Config())`; assert
    `abs(st.fragmentation() - fragmentation(g, Config())) < 1e-9` and
    `abs(st.coverage() - coverage(g, Config())) < 1e-9` and
    `abs(st.connectivity() - connectivity(g, Config())) < 1e-9` and
    `abs(st.path_efficiency() - path_efficiency(g, Config())) < 1e-9`.
14. `test_metrics_state_incremental_after_add(two_component_bike_graph)` —
    `st = MetricsState(g, Config())`; `st.add_edge(2, 5, {'length':200.0,
    'bike_lane':'yes'})`; also `g.add_edge(2, 5, length=200.0,
    bike_lane='yes')` (keep the pure-function input in sync). Assert
    `abs(st.fragmentation() - fragmentation(g, Config())) < 1e-9` and
    `abs(st.coverage() - coverage(g, Config())) < 1e-9`. (This is the load-
    bearing test that the incremental structures stay consistent with the
    pure definitions after an edge add — §6.2/§6.3.)
15. `test_metrics_state_cache_invalidates(two_component_bike_graph)` —
    `st = MetricsState(g, Config())`; `c0 = st.connectivity()`;
    `st.add_edge(2, 5, {'length':200.0})` (a **non**-bike-lane edge, so the
    whole-graph view changes but bike-lane structures don't); `c1 =
    st.connectivity()`; assert `c0 == c1` is **not required** (they may or
    may not differ on this tiny graph) — instead assert that calling
    `st.connectivity()` twice in a row with no intervening `add_edge`
    returns the **same** float (cache hit) and that `st._version` incremented
    after `add_edge`. (Use `st._version` only as a read check; do not make
    the test depend on its exact count.)
16. `test_metrics_state_ignores_self_loop(two_component_bike_graph)` —
    `before = st.fragmentation()`; `st.add_edge(5, 5, {'length':10.0,
    'bike_lane':'yes'})`; assert `st.fragmentation() == before` (self-loops
    are ignored, matching `_simplified_undirected`/`_bike_lane_subgraph`).

Structural pattern: `tests/test_candidates.py` (short functions, docstrings,
`from __future__ import annotations`, fixtures from `conftest.py`).

**Verify**:
- `ruff check tests/test_metrics.py` → exit 0
- `ruff format --check tests/test_metrics.py` → exit 0
- `mypy --strict bike_rl` → exit 0 (tests are not in the mypy scope, but keep them clean)
- `pytest -q tests/test_metrics.py` → all pass (16 tests)

### Step 6: Full gate + coverage

Run the full scoped gate and confirm coverage on the new module.

**Verify**:
- `ruff check bike_rl tests` → exit 0
- `ruff format --check bike_rl tests` → exit 0
- `mypy --strict bike_rl` → exit 0
- `pytest -q` → all pass (22 prior + 16 new = 38)
- `pytest --cov=bike_rl --cov-report=term-missing tests/test_metrics.py` → exit 0;
  **`bike_rl/metrics.py` shows ≥80% line coverage**. The uncovered lines
  should only be rare error/branch paths (e.g. the `math.inf` return in
  `_metres_between` for missing x/y — add a tiny test touching it if it
  pushes coverage below 80%). `objective.py` will still be 0% (it's a stub,
  out of scope) — that's fine; this gate measures `metrics.py` only via the
  `--cov=bike_rl` term-missing report, and `metrics.py` specifically must be
  ≥80%.

## Test plan

New test file `tests/test_metrics.py` (16 tests, listed in Step 5):
- Pure functions: known-answer connectivity (triangle=1.0, small=0.0),
  path-efficiency range + determinism, fragmentation known value (0.23) +
  empty case, coverage sensitivity (§5.7 regression), radius vs component
  modes, empty-bike-lane cases.
- `UnionFind`: basic union/size/isolated tracking, `add_node` idempotency.
- `MetricsState`: matches pure functions before and after `add_edge`
  (§6.2/§6.3 consistency), cache invalidation on version bump (§5.14),
  self-loop ignored.

Structural pattern: `tests/test_candidates.py`.

Verification: `pytest -q tests/test_metrics.py` → 16 pass; `pytest --cov`
shows `bike_rl/metrics.py` ≥80%.

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `ruff check bike_rl tests` exits 0
- [ ] `ruff format --check bike_rl tests` exits 0
- [ ] `mypy --strict bike_rl` exits 0
- [ ] `pytest -q` exits 0; 16 new tests in `tests/test_metrics.py` exist and
      pass; the 22 prior tests still pass
- [ ] `pytest --cov=bike_rl --cov-report=term-missing tests/test_metrics.py`
      shows ≥80% line coverage on `bike_rl/metrics.py`
- [ ] No `raise NotImplementedError("Plan 003")` remains in
      `bike_rl/metrics.py` (`grep -rn "Plan 003" bike_rl/metrics.py` returns
      nothing)
- [ ] `bike_rl/objective.py` is **unchanged** (still a stub) —
      `git diff --stat a828293..HEAD -- bike_rl/objective.py` shows no changes
- [ ] `bike_rl/config.py` is **unchanged** —
      `git diff --stat a828293..HEAD -- bike_rl/config.py` shows no changes
- [ ] No files outside the in-scope list are modified (`git status --short`
      lists only `bike_rl/metrics.py` and `tests/test_metrics.py`)
- [ ] `plans/README.md` status row for 003 updated (TODO → DONE) — unless a
      reviewer told you they maintain the index

## STOP conditions

Stop and report back (do not improvise) if:

- The code at the locations in "Current state" doesn't match the excerpts
  (e.g. `metrics.py` already has real bodies, or `Config` is missing
  `coverage_mode`/`coverage_radius_m`/`sampling_thresholds`/`path_sample_formula`/`seed`)
  — the codebase has drifted since this plan was written; re-baseline before
  continuing.
- The rustworkx API differs from "rustworkx API facts" above — specifically:
  `rx.networkx_converter` does not accept `keep_attributes=True`, or
  `rx.transitivity` / `rx.connected_components` / `rx.dijkstra_shortest_path_lengths`
  do not exist or have different signatures (e.g. `dijkstra_shortest_path_lengths`
  does not take `edge_cost_fn`). Report the installed `rustworkx.__version__`
  and the actual signatures; do not invent a different API path.
- `rx.transitivity` raises on a `PyGraph` with 3 nodes (some versions require
  a minimum size or a different graph type). Report the error and the version.
- You find that `coverage`'s `largest_component_ratio` or `fragmentation`'s
  `isolation_factor` definition is ambiguous and a reasonable reading produces
  a value that **does not move** when a bike-lane edge is added — i.e. the
  §5.7 sensitivity test (#7) cannot be made to pass with a sensible
  definition. Report the definitions you tried and the computed values; do
  not ship a metric that fails its own sensitivity test.
- A step's verification fails twice after a reasonable fix attempt.
- The fix appears to require touching an out-of-scope file (e.g. you find
  `objective.py` or `config.py` must be edited to make tests pass — they must
  not be).

## Maintenance notes

For the human/agent who owns this code after the change lands:

- **`coverage`'s definition is the §5.7 fix and is load-bearing.** It is the
  number the optimiser comparison (OPTIMIZER_SPEC §10) and the RL reward both
  report. If a future change alters `coverage_mode` semantics or the
  `0.7*coverage_ratio + 0.3*largest_component_ratio` blend, re-run the
  sensitivity test (`test_coverage_is_sensitive_to_added_edge`) and update
  the optimiser comparison table. The `largest_component_ratio` denominator
  is **total graph nodes** (not bike-lane-subgraph nodes) deliberately, so
  the term grows as the network grows — do not "fix" it to the subgraph
  count without re-checking sensitivity.
- **`MetricsState` mirrors the pure functions.** The env (plan 004) and the
  optimisers' `objective_delta` will use it for speed. If you add a new
  metric or change a formula, update **both** the pure function and
  `MetricsState` and add a `test_metrics_state_matches_pure`-style assertion.
  The matching tests (#13, #14) are the guardrail.
- **`isolation_factor` is defined as the fraction of bike-lane-subgraph nodes
  in singleton components.** The original spec §3.6 leaves "isolation_factor"
  unspecified; this definition is a documented choice. If a reviewer wants a
  different isolation measure (e.g. components smaller than some threshold),
  change it in **both** `fragmentation` and `MetricsState.fragmentation` and
  update the known-answer test (#5).
- **`local_clustering_coefficient` is not in this rustworkx version**;
  `connectivity` uses global `transitivity` regardless of graph size. If a
  future rustworkx adds sampled local clustering and you want the original
  large-graph behaviour, gate on `cfg.sampling_thresholds` and add a test.
- **`UnionFind` is the §6.3 structure.** The optimisers' `objective_delta`
  (OPTIMIZER_SPEC §3.1, §5) should use `MetricsState` rather than
  re-implementing union-find; if you see a second union-find appear in
  `optim/`, refactor to reuse this one.
- **What a reviewer should scrutinize**: (1) the §5.7 sensitivity test (#7)
  — this is the single most important assertion in the file; (2) the
  `MetricsState`-matches-pure tests (#13, #14) — incremental consistency;
  (3) the `dijkstra_shortest_path_lengths(..., edge_cost_fn=...)` call — the
  argument name is `edge_cost_fn`, not `weight_fn`; a wrong name throws at
  runtime, not at import; (4) the `largest_component_ratio` denominator
  choice in `coverage`.
- **Deferred out of this plan**: `objective.py` (the optimiser-side plan,
  OPTIMIZER_SPEC §11.1), wiring `MetricsState` into `env.py` (plan 004), and
  the `optim/` solvers — all depend on this module's signatures being stable.
