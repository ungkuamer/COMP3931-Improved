# Plan 002: Implement `graph_utils.py` + `candidates.py` with tests

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**:
> `git diff --stat 4f233ea..HEAD -- bike_rl/graph_utils.py bike_rl/candidates.py bike_rl/config.py tests/conftest.py tests/test_graph_utils.py tests/test_candidates.py`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: LOW
- **Depends on**: plans/001-scaffold-package.md (DONE — provides `Config`, `RunContext`, package skeleton, CI)
- **Category**: tech-debt (modular rewrite of the original 1,582-line script's graph/candidate layer) + bug (§5.8 source-graph mutation) + perf (§6.1 OSM cache, §6.5 remove O(V·E) loop) + tests
- **Planned at**: commit `4f233ea`, 2026-06-22
- **Issue**: (not published)

## Why this matters

The original `nx-rx-simple-ur.py` does all graph loading, NetworkX→rustworkx
conversion, and candidate extraction inline in one 1,582-line file, and it
**mutates the source `walk_graph`** by writing `road_priority` /
`connects_to_bike_path` onto the walk graph's own edge dicts (RECREATE_SPEC
§5.8) — those persist across episodes after `reset()` and corrupt results.
Every parallel `SubprocVecEnv` worker also re-downloads the same OSM graphs
from scratch (§6.1), which is rate-limited and slow. And `nx_to_rx` runs an
O(V·E) loop to default missing `length` even though rustworkx's `weight_fn`
already does that (§6.5).

This plan builds the two shared modules that **both the RL pipeline and the
direct optimisers depend on** (see `OPTIMIZER_SPEC.md` §2 and
`RESEARCH_DIRECTION.md`: "common candidate extraction (`candidates.py`) …
common `graph_utils.py`"). Nothing downstream — `metrics.py`, `env.py`,
`objective.py`, the optimisers — can be built or tested until these exist with
a stable, well-tested interface. Getting `Candidate` identity/equality right
here is load-bearing for the greedy/local-search solvers, which call
`remaining.remove(best)` and rely on candidate equality surviving
`connects_to_bike_path` recomputation.

## Current state

The package was scaffolded in plan 001. The relevant files are stubs raising
`NotImplementedError("Plan 002")`:

- `bike_rl/graph_utils.py` — stub with signatures for `nx_to_rx`,
  `load_city_graph`, `load_bbox_graph`, `validate_graph`, `cache_graph`,
  `load_cached_graph`. All bodies raise `NotImplementedError("Plan 002")`.
  Current `nx_to_rx` signature:
  ```python
  def nx_to_rx(nx_graph: _NXGraph) -> tuple[rx.PyDiGraph, dict[str, int]]:
  ```
  (`dict[str, int]` is wrong — OSM node ids are `int`; corrected below.)
- `bike_rl/candidates.py` — stub with a `Candidate` dataclass and
  `extract_candidates` / `recompute_connects` signatures, bodies raise
  `NotImplementedError`. Current `Candidate`:
  ```python
  @dataclass(frozen=True)
  class Candidate:
      u: int | str
      v: int | str
      length: float
      road_priority: int
      connects_to_bike_path: bool
      data: dict[str, object] = field(default_factory=dict)
  ```
  Note: `data` is currently part of equality — this must change (see Step 4).
- `bike_rl/config.py` — frozen `Config` dataclass, fully implemented in plan
  001. Has `edge_cost_factor: float = 10.0`, `min_candidate_length: float =
  100.0`, `max_candidate_length: float = 1000.0`, `road_priorities: dict[str,
  int]` (primary=5…unclassified=1), `coverage_radius_m`, `seed`, etc. **No**
  `default_edge_length` or `osm_cache_dir` field yet — this plan adds them.
- `tests/conftest.py` — has one fixture, `tiny_bike_graph` (a 4-node
  `nx.MultiDiGraph` with `x`/`y` node attrs and two `residential` edges,
  length 120/130, `bike_lane="yes"`). No walk-graph fixture, no OSM mocks yet.
- `tests/test_smoke.py` — existing smoke tests for `Config`/`RunContext`/
  `ObjectiveWeights`; do not touch.

### Repo conventions to match

- **Typing**: `from __future__ import annotations` at the top of every module;
  type hints on all public functions; `mypy --strict bike_rl` must stay clean.
  `pyproject.toml` already sets `ignore_missing_imports = true` for
  `rustworkx.*`, `osmnx.*`, so those imports won't block mypy.
- **Imports**: stdlib first, then third-party, then local (`bike_rl.…`),
  separated by blank lines (ruff `I` is enabled). Use `if TYPE_CHECKING:` for
  imports only needed for annotations (see existing `graph_utils.py` and
  `candidates.py` stubs for the pattern).
- **Docstrings**: Google convention (ruff `D` with `convention = "google"` in
  `pyproject.toml`). Every public function/class gets a docstring. `D100`/
  `D102`/`D104`/`D105`/`D107` are ignored, so module-level docstrings are
  optional but public *functions/classes* must be documented.
- **No `print`**: use `logging` (a module-level `logger = logging.getLogger(
  __name__)`). `print` is only allowed in `cli.py` (§8). Not relevant here
  except: do not add `print`.
- **Tests**: `pytest`, testpaths `["tests"]`, `addopts = "-ra --strict-
  markers"`. Fixtures live in `tests/conftest.py`. Use `monkeypatch` for
  mocking; no network access in tests.
- **Frozen dataclasses** for value types: `Config`, `RunContext`,
  `ObjectiveWeights`, `Candidate` are all `@dataclass(frozen=True)`. Match
  this.
- **Commit style**: Conventional Commits — `git log --oneline` shows
  `feat(scaffold): …`, `test(scaffold): …`, `ci(scaffold): …`. Use
  `feat(graph)`, `feat(candidates)`, `test(graph)`, `test(candidates)`, etc.

### Intent docs to honor (quoted so you don't have to re-read them)

From `RECREATE_SPEC.md` §3.2 (candidate edges):
> Edges from the walk graph that are not already in the bike graph, with
> `highway` type in `{primary, secondary, tertiary, residential, unclassified}`
> (priorities 5/4/3/2/1). Length filter: `100 < length < 1000` metres. Each
> candidate carries `road_priority` and a `connects_to_bike_path` flag (whether
> an endpoint already touches the bike network). Sorted by
> `(road_priority, connects_to_bike_path)` descending.

From `RECREATE_SPEC.md` §5.8:
> in `candidates.py`, build a **fresh** `Candidate` dataclass per edge with its
> own copied attributes; never mutate the source graph. Re-derive
> `connects_to_bike_path` from current `bike_nodes` at step time.

From `RECREATE_SPEC.md` §6.5:
> The loop that defaults `length` is unnecessary because `weight_fn` already
> does `edge_data.get('length', 1.0)`. **Remove the loop.**

From `RECREATE_SPEC.md` §6.1:
> in the parent process, download & simplify the graphs once; serialise
> (pickle) and pass to workers, **or** enable `ox.settings.use_cache=True` …
> and cache to `./osm_cache/` on disk.

From `RECREATE_SPEC.md` §3.3:
> `nx_to_rx(nx_graph)` converts NetworkX → rustworkx via
> `rx.networkx_converter`, builds a `node_map` (original node ID → rustworkx
> index) using the `"__networkx_node__"` attribute, and defaults missing
> `length` to 1.0.

From `OPTIMIZER_SPEC.md` §4 (the optimisers consume these candidates):
> candidate edges C = {e_1, ..., e_n} from the walk graph (candidates.py);
> cost(e) = length(e) * edge_cost_factor; budget B.

From `OPTIMIZER_SPEC.md` §5 (greedy uses `remaining.remove(best)` — candidate
equality must be stable across `connects_to_bike_path` recompute).

## Commands you will need

| Purpose    | Command                                      | Expected on success |
|------------|----------------------------------------------|---------------------|
| Install    | `pip install -e ".[dev]"`                    | exit 0              |
| Ruff lint  | `ruff check bike_rl/graph_utils.py bike_rl/candidates.py tests/test_graph_utils.py tests/test_candidates.py tests/conftest.py` | exit 0, no errors |
| Ruff fmt   | `ruff format --check bike_rl tests`          | exit 0              |
| Mypy       | `mypy --strict bike_rl`                      | exit 0, no errors   |
| Tests      | `pytest -q tests/test_graph_utils.py tests/test_candidates.py` | all pass |
| Coverage   | `pytest --cov=bike_rl --cov-report=term-missing tests/test_graph_utils.py tests/test_candidates.py` | exit 0; `graph_utils.py` and `candidates.py` each ≥80% line coverage |

The full CI gate (`ruff check .`, `ruff format --check .`, `mypy --strict
bike_rl`, `pytest -q`) is run in plan 008; here you only need the scoped
commands above, but running the full gate at the end is a good final check.

## Scope

**In scope** (the only files you should modify):
- `bike_rl/config.py` — **only** to add two fields: `default_edge_length:
  float = 1.0` and `osm_cache_dir: str = "osm_cache"`. Add them in the "Cost /
  budget" or a new "OSM / graph loading" section; update the class docstring
  with the two new attributes. Do not change anything else in `Config`.
- `bike_rl/graph_utils.py` — full implementation.
- `bike_rl/candidates.py` — full implementation (including adjusting the
  `Candidate` dataclass fields/equality and adding `candidate_cost`).
- `tests/conftest.py` — add a `tiny_walk_graph` fixture and OSM mock fixtures
  (see Step 5). Do not remove or change `tiny_bike_graph`.
- `tests/test_graph_utils.py` — create.
- `tests/test_candidates.py` — create.

**Out of scope** (do NOT touch, even though they look related):
- `bike_rl/metrics.py`, `bike_rl/objective.py`, `bike_rl/env.py`,
  `bike_rl/training.py`, `bike_rl/evaluation.py`, `bike_rl/plotting.py`,
  `bike_rl/cli.py`, and anything under `bike_rl/optim/` — these are plans
  003+ and depend on this plan's interfaces. Do not implement them, do not
  "fix" their stubs.
- Do not add `rx_to_nx` (back-conversion). RECREATE_SPEC §5.9 explicitly says
  delete it; if a future plan needs it, that plan will implement + unit-test
  it. The test "round-trip" below means *structural preservation + node_map
  invertibility*, not nx→rx→nx.
- Do not wire `graph_utils`/`candidates` into `env.py` or `training.py`. The
  loader cache functions (`cache_graph`/`load_cached_graph`) are implemented
  here but **called** by `training.py` in plan 005.
- Do not change `pyproject.toml`, `requirements.txt`, CI, `.gitignore`, or
  `tests/test_smoke.py`.

## Git workflow

- Branch: `advisor/002-graph-utils-and-candidates`
- Commit per logical unit (suggested: one commit for `config.py`+`graph_utils.py`,
  one for `candidates.py`, one for the tests). Message style: Conventional
  Commits with scope, e.g. `feat(graph): implement nx_to_rx, OSM loaders, cache`
  — mirroring `git log` entries like `feat(scaffold): …`.
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Add two `Config` fields

Edit `bike_rl/config.py`:

- Add a new attribute-group comment `# OSM / graph loading` (near the "Cost /
  budget" group is fine) with two fields:
  ```python
  # OSM / graph loading
  default_edge_length: float = 1.0
  osm_cache_dir: str = "osm_cache"
  ```
- Add the two attributes to the `Config` class docstring (Google-style:
  `default_edge_length: Length used when an edge has no ``length`` attribute.
  osm_cache_dir: Directory for the OSMnx on-disk cache (relative to CWD).`).
- Place them in a sensible position (e.g. right after `edge_cost_factor`).
  Keep the dataclass `frozen=True`.

**Verify**:
- `ruff check bike_rl/config.py` → exit 0
- `python -c "from bike_rl.config import Config; c=Config(); print(c.default_edge_length, c.osm_cache_dir)"` → prints `1.0 osm_cache`
- `pytest -q tests/test_smoke.py` → all pass (existing Config test still green)

### Step 2: Implement `nx_to_rx` and `validate_graph`

In `bike_rl/graph_utils.py`:

Replace the `nx_to_rx` stub with a real implementation. Use rustworkx's
`networkx_converter` with `keep_attributes=True` (default) and a `weight_fn`
that defaults missing lengths — this removes the O(V·E) defaulting loop
(§6.5). Build `node_map` from the `"__networkx_node__"` attribute that
`networkx_converter` puts on each node payload when `keep_attributes=True`.

Target shape:

```python
import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx
import rustworkx as rx

if TYPE_CHECKING:
    from bike_rl.config import Config

logger = logging.getLogger(__name__)

_NXGraph = nx.MultiDiGraph[Any, Any, Any]


def nx_to_rx(nx_graph: _NXGraph) -> tuple[rx.PyDiGraph, dict[int | str, int]]:
    """Convert a NetworkX MultiDiGraph to a rustworkx PyDiGraph.

    Uses ``rustworkx.networkx_converter`` with ``keep_attributes=True`` and a
    ``weight_fn`` that returns ``edge_data['length']`` or ``1.0`` when missing
    (so no separate O(V·E) defaulting loop is needed — RECREATE_SPEC §6.5).

    Args:
        nx_graph: The NetworkX MultiDiGraph to convert (e.g. an OSMnx graph).

    Returns:
        A ``(rx_graph, node_map)`` tuple where ``node_map`` maps each original
        NetworkX node id to its rustworkx node index, built from the
        ``"__networkx_node__"`` attribute set by ``networkx_converter``.
    """
    rx_graph = rx.networkx_converter(
        nx_graph,
        weight_fn=lambda edge_data: edge_data.get("length", 1.0),
    )
    node_map: dict[int | str, int] = {}
    for rx_idx in rx_graph.node_indices():
        payload = rx_graph[rx_idx]
        node_map[payload["__networkx_node__"]] = rx_idx
    return rx_graph, node_map
```

Then implement `validate_graph`:

```python
def validate_graph(graph: _NXGraph, cfg: Config) -> None:
    """Validate a graph in-place: fill any missing edge ``length`` attribute.

    Missing lengths are set to ``cfg.default_edge_length`` (RECREATE_SPEC §3.3
    defaults to 1.0). ``nx_to_rx`` also defaults via ``weight_fn``, but
    filling here keeps downstream code (candidates read ``data['length']``)
    correct without each caller repeating the check.

    Args:
        graph: The graph to validate (mutated in-place for missing lengths).
        cfg: Config providing ``default_edge_length``.
    """
    for _u, _v, data in graph.edges(data=True):
        if "length" not in data:
            data["length"] = cfg.default_edge_length
```

Note: the stub signature was `validate_graph(graph: _NXGraph) -> None` (no
`cfg`). **Add the `cfg: Config` parameter** — it's needed for
`default_edge_length`. Update the signature accordingly.

**Verify**:
- `ruff check bike_rl/graph_utils.py` → exit 0
- `ruff format --check bike_rl/graph_utils.py` → exit 0 (run `ruff format bike_rl/graph_utils.py` if it reformats)
- `mypy --strict bike_rl/graph_utils.py` (or `mypy --strict bike_rl`) → exit 0
  (If mypy complains that `rx.networkx_converter` is untyped, the
  `ignore_missing_imports = true` override for `rustworkx.*` in
  `pyproject.toml` should already cover it; if not, add a targeted
  `# type: ignore[...]` with a comment — but only if needed.)

### Step 3: Implement OSM loaders + cache functions

In `bike_rl/graph_utils.py`, implement `configure_osm_cache`,
`load_city_graph`, `load_bbox_graph`, `cache_graph`, `load_cached_graph`.

Target shape:

```python
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
    # osmnx version differences: the cache-folder setting has been renamed
    # across versions. Try the modern name, fall back gracefully.
    cache_dir = str(Path(cfg.osm_cache_dir).resolve())
    for attr in ("cache_folder", "cache_folder_name"):
        if hasattr(ox.settings, attr):
            setattr(ox.settings, attr, cache_dir)
            break


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
    north: float, south: float, east: float, west: float, cfg: Config, network_type: str = "bike"
) -> _NXGraph:
    """Load a street graph for a bounding box.

    Args:
        north, south, east, west: Bounding box in the ``(N, S, E, W)`` order
            required by OSMnx (RECREATE_SPEC §3.1).
        cfg: Config (used to enable the OSM cache).
        network_type: OSMnx network type (``"bike"`` or ``"walk"``).

    Returns:
        The downloaded MultiDiGraph.
    """
    import osmnx as ox

    configure_osm_cache(cfg)
    return ox.graph_from_bbox(north, south, east, west, network_type=network_type)


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
        return pickle.load(f)  # noqa: S301 (trusted cache file we wrote ourselves)
```

Notes for the executor:
- The stub signatures for `load_city_graph`/`load_bbox_graph` did not include
  `network_type`. **Add `network_type: str = "bike"`** to both — the env
  (plan 004) will call with `"walk"` to get the candidate source graph.
- `load_city_graph`/`load_bbox_graph` did not take `cfg` originally? They did
  — the stub already had `cfg: Config`. Keep it.
- `import osmnx as ox` is done *inside* the functions (not at module top)
  because osmnx is heavy and importing it at module load slows test
  collection; the tests mock these `ox.*` calls. Keep the local import.
- The `# noqa: S301` suppresses ruff's pickle-load warning (bandit rule).
  ruff's selected rules include `B` (flake8-bugbear) but `S` (bandit) is **not**
  in the selected set in `pyproject.toml` (`select = ["E","F","I","UP","B",
  "SIM","D"]`), so the noqa may be unnecessary. Run `ruff check` and remove the
  noqa if ruff reports it as unused (ruff flags unused noqa with `RUF100` only
  if `RUF` is selected — it isn't, so an unused noqa is harmless; keep it for
  documentation unless ruff errors).

**Verify**:
- `ruff check bike_rl/graph_utils.py` → exit 0
- `ruff format --check bike_rl/graph_utils.py` → exit 0
- `mypy --strict bike_rl` → exit 0
- `python -c "from bike_rl.graph_utils import nx_to_rx, validate_graph, configure_osm_cache, load_city_graph, load_bbox_graph, cache_graph, load_cached_graph; print('ok')"` → prints `ok`

### Step 4: Implement `Candidate` + `candidate_cost` + `extract_candidates` + `recompute_connects`

In `bike_rl/candidates.py`. The key design decision: **`connects_to_bike_path`
and `data` must be excluded from equality/hash**, so that a `Candidate`
returned by `recompute_connects` stays equal to the original for
`list.remove(best)` in the greedy solver (plan 003) and for set membership.
Identity = `(u, v, length, road_priority)`.

Target shape:

```python
"""Candidate extraction for bike-network expansion.

See RECREATE_SPEC.md §3.2, §5.8. Candidates are built fresh from the walk
graph without mutating it; ``connects_to_bike_path`` is re-derived from the
current bike-network node set at step time via :func:`recompute_connects`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import networkx as nx

    from bike_rl.config import Config


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
    connects_to_bike_path: bool
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
    bike_graph: nx.MultiDiGraph, walk_graph: nx.MultiDiGraph, cfg: Config
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


def recompute_connects(
    candidates: list[Candidate], bike_nodes: set[int | str]
) -> list[Candidate]:
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
```

Notes:
- The stub `extract_candidates` signature was
  `extract_candidates(bike_graph: object, walk_graph: object, cfg: Config)`.
  Tighten the type hints to `nx.MultiDiGraph` (import under `TYPE_CHECKING`).
  `recompute_connects` keeps its signature.
- `dict(data)` is a shallow copy — sufficient because we never mutate the
  values inside; it satisfies §5.8 (we never write to the walk graph's dict).
- Do not add `rx_to_nx` or any back-conversion.

**Verify**:
- `ruff check bike_rl/candidates.py` → exit 0
- `ruff format --check bike_rl/candidates.py` → exit 0
- `mypy --strict bike_rl` → exit 0
- `python -c "from bike_rl.candidates import Candidate, candidate_cost, extract_candidates, recompute_connects; from bike_rl.config import Config; print('ok')"` → prints `ok`

### Step 5: Extend `tests/conftest.py` with fixtures

Add a `tiny_walk_graph` fixture and OSM-mock fixtures. Keep the existing
`tiny_bike_graph` unchanged. Append to `tests/conftest.py`:

```python
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
    for n, (x, y) in {
        1: (0.0, 0.0),
        2: (0.001, 0.0),
        3: (0.0, 0.001),
        4: (0.001, y=0.001),
        5: (0.002, 0.002),
    }.items():
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
    """Monkeypatch ``osmnx.graph_from_place`` / ``graph_from_bbox`` to return
    a synthetic graph (no network access). Yields a dict recording call args.
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

    def fake_bbox(north, south, east, west, network_type="bike", **kw):
        calls["bbox"].append(
            {"N": north, "S": south, "E": east, "W": west, "network_type": network_type, **kw}
        )
        g = nx.MultiDiGraph()
        g.add_node(1, x=0.0, y=0.0)
        g.add_node(2, x=0.001, y=0.0)
        g.add_edge(1, 2, length=120.0, highway="residential")
        return g

    monkeypatch.setattr(ox, "graph_from_place", fake_place)
    monkeypatch.setattr(ox, "graph_from_bbox", fake_bbox)
    return calls
```

**Important**: in the `tiny_walk_graph` fixture above, the node-adding loop
has a bug on purpose for you to **fix while typing it** — the dict literal for
node 4 uses `y=0.001` as a keyword inside a dict display, which is a
`SyntaxError`. Write it as a plain dict: `{1: (0.0, 0.0), 2: (0.001, 0.0),
3: (0.0, 0.001), 4: (0.001, 0.001), 5: (0.002, 0.002)}` and iterate
`for n, (x, y) in NODES.items(): g.add_node(n, x=x, y=y)`. Do not copy the
broken line.

**Verify**:
- `ruff check tests/conftest.py` → exit 0
- `ruff format --check tests/conftest.py` → exit 0
- `pytest -q tests/test_smoke.py` → still all pass (existing fixtures intact)
- `python -c "import tests.conftest"` → no error (or `pytest --collect-only -q tests/conftest.py` → exit 0)

### Step 6: Write `tests/test_graph_utils.py`

Create `tests/test_graph_utils.py` covering: `nx_to_rx` structural
preservation + node_map correctness; `validate_graph` fills missing length;
`configure_osm_cache` enables `use_cache`; `load_city_graph` /
`load_bbox_graph` call OSMnx with the right args and network_type (via
`mock_osm`); `cache_graph` / `load_cached_graph` pickle round-trip.

Model the file structure on `tests/test_smoke.py` (top-of-file docstring,
`from __future__ import annotations`, small focused test functions with
docstrings). Use the `tiny_bike_graph`, `tiny_walk_graph`, and `mock_osm`
fixtures.

Test cases (write all):

1. `test_nx_to_rx_preserves_nodes_and_edges(tiny_bike_graph)` —
   `rx_graph, node_map = nx_to_rx(g)`; assert `rx_graph.num_nodes() ==
   g.number_of_nodes()` and `rx_graph.num_edges() == g.number_of_edges()`.
2. `test_nx_to_rx_node_map_is_bijection(tiny_bike_graph)` — every `g` node id
   is a key in `node_map`; `set(node_map.values()) == set(range(num_nodes))`;
   each `rx_graph[node_map[orig]]["__networkx_node__"] == orig`.
3. `test_nx_to_rx_weight_fn_defaults_missing_length()` — build a small
   `nx.MultiDiGraph` with one edge that has **no** `length` attribute and one
   edge with `length=42.0`; `nx_to_rx`; retrieve edge payloads and assert the
   missing-length edge weighs `1.0` and the other weighs `42.0`. (Access
   weights via `rx_graph.weighted_edge_list()` or `rx_graph.edges()` — use
   whichever rustworkx API returns the payload; if the payload is the dict,
   read `payload.get("length", 1.0)` to confirm. Keep this test robust: the
   point is "no exception, default applied" — assert that the edge with
   `length=42` yields weight `42.0` via `weight_fn`, and the missing one
   yields `1.0`.)
4. `test_validate_graph_fills_missing_length()` — build a graph with an edge
   missing `length`; call `validate_graph(g, Config())`; assert every edge now
   has `"length" == 1.0` (for the one that was missing) and existing lengths
   unchanged.
5. `test_configure_osm_cache_enables_use_cache()` — call
   `configure_osm_cache(Config())`; `import osmnx as ox; assert
   ox.settings.use_cache is True`. (Restore prior value via `monkeypatch` or a
   `try/finally` to avoid leaking state across tests — simplest: capture
   `old = ox.settings.use_cache` in the test and restore in `finally`.)
6. `test_load_city_graph_calls_graph_from_place(mock_osm)` —
   `load_city_graph("Otley, UK", Config(), network_type="bike")`; assert
   `mock_osm["place"]` has one entry with `name == "Otley, UK"` and
   `network_type == "bike"`.
7. `test_load_city_graph_network_type_walk(mock_osm)` — call with
   `network_type="walk"`; assert the recorded `network_type == "walk"`.
8. `test_load_bbox_graph_uses_nsew_order(mock_osm)` —
   `load_bbox_graph(53.9, 53.8, -1.6, -1.7, Config())`; assert the recorded
   call has `N==53.9, S==53.8, E==-1.6, W==-1.7` in that order (guards the
   §3.1 N,S,E,W requirement).
9. `test_cache_graph_round_trip(tiny_bike_graph, tmp_path)` —
   `cache_graph(g, tmp_path / "g.pkl")`; `g2 = load_cached_graph(tmp_path /
   "g.pkl")`; assert `g2.number_of_nodes() == g.number_of_nodes()` and
   `g2.number_of_edges() == g.number_of_edges()` and the length of edge
   `(1,2,0)` equals `120.0` in both.

**Verify**:
- `ruff check tests/test_graph_utils.py` → exit 0
- `ruff format --check tests/test_graph_utils.py` → exit 0
- `pytest -q tests/test_graph_utils.py` → all pass (9 tests)

### Step 7: Write `tests/test_candidates.py`

Create `tests/test_candidates.py` covering: length filter, road-priority
ordering, no mutation of the source graph, `connects_to_bike_path`
propagation after an add, list-valued `highway`, `candidate_cost`, and
candidate equality stability across `recompute_connects`.

Test cases (write all):

1. `test_extract_candidates_filters_by_length(tiny_bike_graph,
   tiny_walk_graph)` — `extract_candidates(...)`; assert no candidate has
   `length <= 100` or `>= 1000` (the `(1,5)` 50 m and `(2,3)` 1200 m edges are
   excluded). Assert the `(1,2)` edge (already in bike graph) is **not**
   present.
2. `test_extract_candidates_orders_by_priority_then_connects(tiny_bike_graph,
   tiny_walk_graph)` — the resulting list is sorted by
   `(road_priority, connects_to_bike_path)` descending; assert
   `candidates[0].road_priority >= candidates[-1].road_priority` and that the
   primary candidate (200 m, priority 5) comes before the tertiary (600 m,
   priority 3). (All candidates here connect, so the secondary sort key is
   not distinguished — that's fine; the road-priority ordering is the
   observable property.)
3. `test_extract_candidates_does_not_mutate_source(tiny_bike_graph,
   tiny_walk_graph)` — before calling, snapshot
   `walk_graph.edges(data=True)` (e.g. `before = dict(tiny_walk_graph.edges(
   keys=False, data=True))` or capture each edge's dict `id()`). Call
   `extract_candidates(...)`; assert no edge data dict on `tiny_walk_graph`
   now has a `"road_priority"` or `"connects_to_bike_path"` key, and that the
   edge data dicts are the same objects (`id()` unchanged) — i.e. the source
   graph was not mutated (§5.8).
4. `test_extract_candidates_handles_list_highway(tiny_bike_graph,
   tiny_walk_graph)` — the `(3,4)` edge has `highway=["tertiary",
   "residential"]` but is already in the bike graph, so it's excluded by
   `has_edge`. To test the list-valued path directly, build a tiny separate
   walk graph with one edge `(10, 11, length=200,
   highway=["primary","residential"])` and a bike graph with no `(10,11)`
   edge and node 10 in it; assert `extract_candidates` returns one candidate
   with `road_priority == 5` (max of the listed types).
5. `test_extract_candidates_connects_flag(tiny_bike_graph, tiny_walk_graph)` —
   every returned candidate has `connects_to_bike_path == True` here because
   all candidate endpoints (2,3,4) are in `tiny_bike_graph`. To test the
   `False` case, build a walk graph with an edge between two nodes **not** in
   the bike graph (e.g. nodes 100, 101, length 200, highway="residential")
   and assert the resulting candidate has `connects_to_bike_path is False`.
6. `test_recompute_connects_propagates_after_add(tiny_bike_graph,
   tiny_walk_graph)` — get candidates; pick the candidate whose `u`/`v` is
   **not** yet connected (use the synthetic from case 5, or simpler: take the
   `(5,5)` self-loop candidate whose endpoint 5 is not in bike_nodes →
   `connects_to_bike_path is False`). Then simulate "node 5 is now in the bike
   network" by calling `recompute_connects(candidates, bike_nodes | {5})`;
   assert that candidate now has `connects_to_bike_path is True`.
7. `test_recompute_connects_returns_new_list(tiny_bike_graph,
   tiny_walk_graph)` — `new = recompute_connects(cands, bike_nodes)`; assert
   `new is not cands` and that input `cands` objects' `connects_to_bike_path`
   values are unchanged (inputs not mutated).
8. `test_candidate_equality_stable_across_recompute()` — build two candidates
   with same `(u, v, length, road_priority)` but different
   `connects_to_bike_path` and different `data`; assert they are `==` and
   `hash(c1) == hash(c2)`. Then put one in a list and `list.remove(the_other)`
   — assert it removes without `ValueError` (this is the property the greedy
   solver relies on).
9. `test_candidate_cost()` — `c = Candidate(1, 2, length=200.0,
   road_priority=5, connects_to_bike_path=True)`; `assert candidate_cost(c,
   Config()) == 2000.0` (200 * 10).

**Verify**:
- `ruff check tests/test_candidates.py` → exit 0
- `ruff format --check tests/test_candidates.py` → exit 0
- `pytest -q tests/test_candidates.py` → all pass (9 tests)

### Step 8: Full gate + coverage

Run the full scoped gate and confirm coverage on the two new modules.

**Verify**:
- `ruff check bike_rl tests` → exit 0
- `ruff format --check bike_rl tests` → exit 0
- `mypy --strict bike_rl` → exit 0
- `pytest -q` → all pass (existing smoke tests + 18 new tests)
- `pytest --cov=bike_rl --cov-report=term-missing tests/test_graph_utils.py tests/test_candidates.py` → exit 0; **`bike_rl/graph_utils.py` and `bike_rl/candidates.py` each show ≥80% line coverage** (the stubs' `raise NotImplementedError` lines should now be gone, so uncovered lines should only be minor error branches). If coverage on either file is below 80%, add tests for the uncovered branches before finishing.

## Test plan

New test files (all listed in Steps 6–7):
- `tests/test_graph_utils.py` — 9 tests covering `nx_to_rx`, `validate_graph`,
  `configure_osm_cache`, `load_city_graph`, `load_bbox_graph`,
  `cache_graph`/`load_cached_graph`.
- `tests/test_candidates.py` — 9 tests covering length filter, ordering,
  no-mutation (§5.8 regression), list-valued `highway`, `connects` flag and
  propagation, `recompute_connects` freshness, equality/hash stability, and
  `candidate_cost`.

Structural pattern to model on: `tests/test_smoke.py` (short functions,
docstrings, `from __future__ import annotations`, fixtures from
`conftest.py`).

Verification: `pytest -q tests/test_graph_utils.py tests/test_candidates.py`
→ all 18 pass; plus `pytest --cov` shows ≥80% on both modules.

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `ruff check bike_rl tests` exits 0
- [ ] `ruff format --check bike_rl tests` exits 0
- [ ] `mypy --strict bike_rl` exits 0
- [ ] `pytest -q` exits 0; 18 new tests (9 in `test_graph_utils.py`, 9 in
      `test_candidates.py`) exist and pass; existing `test_smoke.py` still
      passes
- [ ] `pytest --cov=bike_rl --cov-report=term-missing tests/test_graph_utils.py tests/test_candidates.py` shows ≥80% line coverage on both `bike_rl/graph_utils.py` and `bike_rl/candidates.py`
- [ ] No `raise NotImplementedError("Plan 002")` remains in
      `bike_rl/graph_utils.py` or `bike_rl/candidates.py`
      (`grep -rn "Plan 002" bike_rl/` returns no matches in those two files)
- [ ] No files outside the in-scope list are modified (`git status --short`
      lists only `bike_rl/config.py`, `bike_rl/graph_utils.py`,
      `bike_rl/candidates.py`, `tests/conftest.py`,
      `tests/test_graph_utils.py`, `tests/test_candidates.py`)
- [ ] `plans/README.md` status row for 002 updated (TODO → DONE, or
      IN PROGRESS → DONE) — unless a reviewer told you they maintain the index

## STOP conditions

Stop and report back (do not improvise) if:

- The code at the locations in "Current state" doesn't match the excerpts
  (e.g. `Candidate` already has `compare=False` on `data`, or `Config` already
  has `default_edge_length`/`osm_cache_dir`) — the codebase has drifted since
  this plan was written; re-baseline before continuing.
- `rustworkx.networkx_converter` does not accept a `weight_fn` argument, or
  does not set `"__networkx_node__"` on node payloads with
  `keep_attributes=True` (the API changed). Report the installed rustworkx
  version and the actual API; do not invent a different conversion strategy.
- The osmnx settings API does not expose `use_cache` or any of
  `cache_folder`/`cache_folder_name` (osmnx version mismatch). Report the
  installed osmnx version and the available settings attributes; keep the
  `ox.settings.use_cache = True` line if it exists, otherwise skip cache-folder
  configuration and note it.
- A step's verification fails twice after a reasonable fix attempt.
- The fix appears to require touching an out-of-scope file (e.g. you find
  `metrics.py` or `env.py` must be edited to make tests pass — they must not
  be).
- You discover that `extract_candidates`'s "already in the bike graph" check
  needs the OSM edge **key** (not just `(u, v)`) to be correct for parallel
  multi-edges. If `bike_graph.has_edge(u, v)` is insufficient for the
  test fixtures, report it; do not silently change the semantics — the
  downstream env/optimiser plans assume `(u, v)`-level exclusion.

## Maintenance notes

For the human/agent who owns this code after the change lands:

- **`Candidate` equality is `(u, v, length, road_priority)`** — intentionally
  excluding `connects_to_bike_path` and `data`. If a future change makes two
  distinct real-world edges share those four fields (e.g. parallel edges with
  identical length between the same pair), the equality will collide. If that
  ever matters, add a `key` field (the OSM edge key) to the identity set.
  Review the greedy/local-search solvers (plan 003+) when they land to
  confirm they rely on this equality contract.
- **`candidate_cost` is the single source of cost.** The env (plan 004) and
  the optimisers (plan 003+) should both call it rather than recomputing
  `length * edge_cost_factor`; if you see them inline the formula, refactor to
  call `candidate_cost`.
- **OSM cache**: `configure_osm_cache` is called by both loaders. If a future
  plan adds a `--no_cache` CLI flag, gate the `configure_osm_cache` call on a
  `Config` boolean (add the field) rather than bypassing it ad hoc.
- **`cache_graph`/`load_cached_graph` use pickle.** They are intended for
  graphs we downloaded ourselves (trusted). Do not load `.pkl` files from
  untrusted sources with `load_cached_graph`. Plan 005 wires these into
  `SubprocVecEnv` worker setup.
- **What a reviewer should scrutinize**: (1) the `compare=False` on
  `Candidate.data` and `connects_to_bike_path` — this is load-bearing for the
  optimisers; (2) the `(N, S, E, W)` ordering in `test_load_bbox_graph_uses_
  nsew_order` — this guards the original §5.1-class bbox bug; (3) the
  no-mutation test (`test_extract_candidates_does_not_mutate_source`) — this
  guards the §5.8 regression.
- **Deferred out of this plan**: wiring `graph_utils`/`candidates` into
  `env.py`, `training.py`, `objective.py`, and the optimisers — those are
  plans 003–005+ and depend on the interfaces fixed here.
