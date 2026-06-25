# Plan 013: Implement the ILP coverage-only oracle (`optim/ilp.py`) + `test_ilp.py`

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.

> **Drift check (run first)**: `git diff --stat bc58ad2..HEAD -- bike_rl/optim/ilp.py bike_rl/metrics.py bike_rl/objective.py bike_rl/optim/greedy.py bike_rl/optim/budget.py bike_rl/candidates.py bike_rl/config.py tests/conftest.py`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts below against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: MED
- **Depends on**: plans/010-objective-and-tests.md (DONE), plans/011-greedy-and-tests.md (DONE) — uses `objective`, `ObjectiveWeights`, `Solution`, `cost`, `Candidate`
- **Category**: tests (new solver + tests; extends the optimiser baseline)
- **Planned at**: commit `bc58ad2`, 2026-06-25
- **Issue**: (not published)

## Why this matters

The ILP is the **ground-truth ceiling** for the optimiser comparison
(`OPTIMIZER_SPEC.md` §1, §7, §10). Without it, greedy / local-search / RL
numbers float without a reference optimum — you cannot report an *optimality
gap*. This plan delivers the spec's first oracle: the **coverage-only ILP**
("maximise bike-lane-reachable population under budget", §7), which is the
clean classic **budgeted maximum-coverage** problem and is exactly
linearisable in radius coverage mode. It reuses the shared `objective`/`Solution`
machinery so its result row is directly comparable to the greedy / local-search
/ (later) RL rows (§8). The full-objective ILP (connectivity via flow/Steiner,
§7 "Full-objective ILP") is intentionally **deferred** to a later plan — this
one ships the scalable exact oracle the comparison table needs today.

## Current state

### Files and their roles
- `bike_rl/optim/ilp.py` — **currently a stub** raising `NotImplementedError("Optimiser plan")` in both `__init__` and `solve`. This plan replaces it entirely.
- `bike_rl/optim/greedy.py` — defines the `Solution` dataclass this plan **reuses unchanged** (see excerpt below). Do NOT modify it.
- `bike_rl/optim/budget.py` — `cost(edge, cfg)` delegate to `candidate_cost`; reuse unchanged.
- `bike_rl/objective.py` — `objective(graph, added_edges, weights, cfg)`; reuse to score the ILP's chosen edge set. Do NOT modify.
- `bike_rl/metrics.py` — `coverage(graph, cfg)` (the radius branch is what the ILP linearises) and the **private** `_metres_between(a, b)` distance helper (used by `coverage`). The ILP's coverage surrogate MUST agree with `coverage`'s radius branch, so this plan imports `_metres_between` from `bike_rl.metrics`. Do NOT modify `metrics.py`.
- `bike_rl/candidates.py` — `Candidate` (frozen dataclass; identity = `(u, v, length, road_priority)`) and `candidate_cost(c, cfg) = c.length * cfg.edge_cost_factor`.
- `bike_rl/config.py` — already has the fields this plan needs: `ilp_time_limit_s: float = 300.0`, `ilp_default_solver: str = "ortools"`, `max_candidates_for_ilp: int = 300`, `coverage_radius_m: float = 300.0`, `coverage_mode: str = "radius"`, `edge_cost_factor: float = 10.0`. **No Config change is required.** Do NOT modify `config.py`.
- `tests/conftest.py` — `tiny_bike_graph` / `tiny_walk_graph` fixtures. **These saturate coverage at 1.0** (all nodes already within 300 m of a base bike-lane edge), so they are useless for ILP-vs-brute-force. This plan defines its own fixture (see Step 1).
- `requirements.txt` / `pyproject.toml` — `ortools>=9.10` is already a dependency and is installed in `.venv` (verified: `from ortools.sat.python import cp_model` works). **No dependency change.** OR-Tools CP-SAT is the only engine this plan supports; PuLP/CBC and gurobipy fallbacks are **out of scope** (deferred).

### Excerpt — `Solution` (reused unchanged, `bike_rl/optim/greedy.py`)
```python
@dataclass
class Solution:
    edges: list[Any] = field(default_factory=list)
    objective: float = 0.0
    spent: float = 0.0
    runtime_s: float = 0.0
    solver: str = ""
    extra: dict[str, object] = field(default_factory=dict)
```
Import it: `from bike_rl.optim.greedy import Solution`. Keep field names and types identical so later `evaluate.py` (§11 item 5) can consume it.

### Excerpt — `coverage` radius branch, the definition the ILP must match (`bike_rl/metrics.py`)
```python
bike_nodes = {n for u, v, d in graph.edges(data=True)
              if d.get("bike_lane") == "yes" for n in (u, v)}
# radius mode:
covered = set()
for n, ndata in graph.nodes(data=True):
    for bn in bike_nodes:
        if _metres_between(ndata, graph.nodes[bn]) <= cfg.coverage_radius_m:
            covered.add(n); break
coverage_ratio = len(covered) / graph.number_of_nodes()
```
`_metres_between` (private, `bike_rl/metrics.py`) is the equirectangular metres
helper. A node is "reachable" iff it is within `cfg.coverage_radius_m` of **any
bike-lane edge endpoint** (existing base OR added). The ILP linearises exactly
this: a selected candidate `e` contributes its two endpoints `{e.u, e.v}` to the
`bike_nodes` set; the reachable set is the nodes within radius of any such
endpoint (base ones are free).

### Excerpt — `Candidate` cost and `cost()` (`bike_rl/optim/budget.py`, `bike_rl/candidates.py`)
```python
def cost(edge: Candidate, cfg: Config) -> float:
    return candidate_cost(edge, cfg)        # = edge.length * cfg.edge_cost_factor
```

### Conventions to match
- Style: `from __future__ import annotations`; `TYPE_CHECKING` guard for `nx.MultiDiGraph[Any, Any, Any]` aliasing (see `greedy.py` lines 8–14). Match exactly.
- Docstrings: Google style, module docstring first line `"""<module purpose>. See OPTIMIZER_SPEC.md §7."""` (see `greedy.py`, `local_search.py`).
- Public class docstring explains determinism and the spec reference (mirror `GreedySolver` / `LocalSearchSolver` docstrings).
- Tests: mirror `tests/test_local_search.py` structure — module docstring, local `_edge(...)` helper, `default_cfg` / `default_weights` fixtures, one `TestILPSolver` class. `pytest` style, `pytest.approx` for floats.
- ruff (`E,F,I,UP,B,SIM,D`; line-length 100) and `mypy --strict` are enforced (plan 008). CP-SAT APIs sometimes need `# type: ignore[...]` — keep them narrow and justified; `pyproject.toml` already sets `ignore_missing_imports = true` for `ortools.*`.

## Commands you will need

Run all commands from the repo root with the venv active.

| Purpose | Command | Expected on success |
|---|---|---|
| Activate venv | `source .venv/bin/activate` | shell prompt changes |
| Run only the new tests | `pytest -q tests/test_ilp.py` | all pass (no `--cov` interaction: `addopts` already adds coverage; whole-`bike_rl` gate stays ≥80%) |
| Full test suite | `pytest -q` | all pass, coverage ≥80% |
| Lint | `ruff check .` | exit 0 |
| Format check | `ruff format --check .` | exit 0 (run `ruff format .` if it reports diffs) |
| Type check | `mypy --strict bike_rl` | exit 0 |
| OR-Tools import sanity | `python -c "from ortools.sat.python import cp_model; cp_model.CpModel(); cp_model.CpSolver()"` | exit 0, no output |

## Scope

**In scope** (the only files you should modify):
- `bike_rl/optim/ilp.py` — replace stub with the full `ILPSolver`.
- `tests/test_ilp.py` — create new test file.

**Out of scope** (do NOT touch, even though they look related):
- `bike_rl/optim/greedy.py`, `local_search.py`, `budget.py`, `evaluate.py` — other solvers / harness; `evaluate.py` is the *next* optimiser plan (§11 item 5).
- `bike_rl/objective.py`, `bike_rl/metrics.py`, `bike_rl/config.py`, `bike_rl/candidates.py` — reuse only; this plan adds no `Config` field and no metric.
- The **full-objective ILP** (connectivity via flow/Steiner auxiliary vars, §7 "Full-objective ILP", and the `coverage_mode == "component"` linearisation, which needs connected-component auxiliary variables) — **deferred** to a follow-up plan. This plan raises a clear, documented error for `coverage_mode != "radius"` rather than silently approximating.
- PuLP/CBC / gurobipy engine support — deferred. Only OR-Tools CP-SAT here.
- Filtering candidates down to `cfg.max_candidates_for_ilp` — that threshold is enforced by the future `evaluate.py` harness, **not** by `ILPSolver`. `ILPSolver` solves exactly what it is handed.
- `__init__.py` re-exports — leave the existing `optim/__init__.py` alone.

## Git workflow

- Branch: `advisor/013-ilp-and-tests` (matches the `advisor/NNN-<slug>` convention used by prior plans).
- Commit per logical unit (e.g. one commit for the solver, one for the tests). Message style: conventional commits — e.g. `feat(optim): add ILP coverage-only oracle (OPTIMIZER_SPEC §7)` and `test(optim): add test_ilp.py brute-force parity`. See `git log --oneline -10` for the repo's existing style.
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Define the brute-force fixture in `tests/test_ilp.py` (tests first)

Create `tests/test_ilp.py`. Start with the module docstring, imports, the local
`_edge(...)` helper, the `default_cfg` / `default_weights` fixtures, and a
`max_coverage_instance` fixture (a function-scoped `pytest.fixture` returning
`(graph, candidates, budget_by_case)`). Use **coverage-only weights**
`ObjectiveWeights(connectivity=0.0, coverage=1.0, fragmentation=0.0)` only where
a test needs `objective()`-consistency, and the default `ObjectiveWeights()`
elsewhere (the ILP optimises the surrogate regardless of weights).

The fixture (coordinates in degrees; far nodes are **0.005° apart ≈ 556 m >
300 m radius**, so radius never bleeds between them — each candidate edge covers
exactly its two endpoints, making this a clean budgeted maximum-coverage
instance):

```python
import itertools
import math

import networkx as nx
import pytest

from bike_rl.candidates import Candidate, candidate_cost
from bike_rl.config import Config
from bike_rl.metrics import _metres_between, coverage
from bike_rl.objective import ObjectiveWeights, objective
from bike_rl.optim.budget import cost
from bike_rl.optim.greedy import Solution
from bike_rl.optim.ilp import ILPSolver


def _edge(u: int | str, v: int | str, length: float, road_priority: int = 1) -> Candidate:
    """Build a Candidate satisfying the Edge protocol."""
    return Candidate(
        u=u,
        v=v,
        length=float(length),
        road_priority=road_priority,
        connects_to_bike_path=False,
        data={"length": float(length), "highway": "residential"},
    )


@pytest.fixture
def default_cfg() -> Config:
    return Config()


@pytest.fixture
def default_weights() -> ObjectiveWeights:
    return ObjectiveWeights()


@pytest.fixture
def cov_only_weights() -> ObjectiveWeights:
    """Coverage-only weights: connectivity=0, fragmentation=0, coverage=1."""
    return ObjectiveWeights(connectivity=0.0, coverage=1.0, fragmentation=0.0)


@pytest.fixture
def max_coverage_instance() -> tuple[nx.MultiDiGraph, list[Candidate]]:
    """Clean budgeted max-coverage fixture (radius does not bleed).

    9 nodes. Base bike-lane edges among {1,2,3} cover only {1,2,3}
    (3/9). Far nodes 10..15 are mutually >300 m apart, so each candidate
    edge covers exactly its two endpoints. Base coverage_ratio = 3/9.

    Candidates (length -> cost = length * edge_cost_factor=10):
      (1,10)  len 100 -> 1000 : adds node 10
      (10,11) len  60 ->  600  : adds 10,11
      (11,12) len  60 ->  600  : adds 11,12
      (12,13) len  60 ->  600  : adds 12,13
      (13,14) len  60 ->  600  : adds 13,14
      (14,15) len  60 ->  600  : adds 14,15
      (2,15)  len 100 -> 1000 : adds node 15
      (1,11)  len 120 -> 1200 : adds node 11

    Brute-force optima (covered-count / 9), VERIFIED during planning:
      budget  600 -> {(10,11)}            covered=5  cr=5/9  spent=600
      budget 1000 -> {(10,11)}            covered=5  cr=5/9  spent=600
      budget 1200 -> any 2 disjoint        covered=7  cr=7/9  spent=1200
      budget 1800 -> {(10,11),(12,13),(14,15)} covered=9 cr=1.0  spent=1800
    """
    coords = {
        1:  (0.000, 0.000), 2:  (0.001, 0.000), 3:  (0.0005, 0.001),
        10: (0.010, 0.000), 11: (0.015, 0.000), 12: (0.020, 0.000),
        13: (0.025, 0.000), 14: (0.030, 0.000), 15: (0.035, 0.000),
    }
    g = nx.MultiDiGraph()
    for n, (x, y) in coords.items():
        g.add_node(n, x=x, y=y)
    g.add_edge(1, 2, length=120.0, highway="residential", bike_lane="yes")
    g.add_edge(2, 3, length=120.0, highway="residential", bike_lane="yes")
    candidates = [
        _edge(1, 10, 100.0, road_priority=5),
        _edge(10, 11, 60.0, road_priority=4),
        _edge(11, 12, 60.0, road_priority=4),
        _edge(12, 13, 60.0, road_priority=4),
        _edge(13, 14, 60.0, road_priority=3),
        _edge(14, 15, 60.0, road_priority=3),
        _edge(2, 15, 100.0, road_priority=5),
        _edge(1, 11, 120.0, road_priority=2),
    ]
    return g, candidates
```

Also add a brute-force helper in the test module for parity:
```python
def _reachable_count(graph: nx.MultiDiGraph, chosen: list[Candidate], cfg: Config) -> int:
    """Number of graph nodes within cfg.coverage_radius_m of any bike-lane endpoint.

    Mirrors the radius branch of bike_rl.metrics.coverage; uses the SAME
    _metres_between so it cannot drift from coverage().
    """
    bike = {n for u, v, d in graph.edges(data=True)
            if d.get("bike_lane") == "yes" for n in (u, v)}
    for e in chosen:
        bike.add(e.u); bike.add(e.v)
    covered = 0
    for n, ndata in graph.nodes(data=True):
        for bn in bike:
            if _metres_between(ndata, graph.nodes[bn]) <= cfg.coverage_radius_m:
                covered += 1
                break
    return covered


def _brute_best(graph, candidates, cfg, budget):
    """Return (best_reachable_count, best_chosen_set) over all affordable subsets."""
    best = -1
    best_set: list[Candidate] = []
    for r in range(len(candidates) + 1):
        for S in itertools.combinations(candidates, r):
            spent = sum(candidate_cost(c, cfg) for c in S)
            if spent > budget + 1e-9:
                continue
            cnt = _reachable_count(graph, list(S), cfg)
            if cnt > best + 1e-12 or (abs(cnt - best) <= 1e-12 and len(S) < len(best_set)):
                best = cnt
                best_set = list(S)
    return best, best_set
```

**Verify**: `python -c "import tests.test_ilp as t; print('ok')"` → no import
error (this requires Step 2 to exist; if you prefer, write Step 2 first and
Step 1 second — order of the two is interchangeable, but both must land).

### Step 2: Implement `ILPSolver` in `bike_rl/optim/ilp.py`

Replace the entire stub file. Structure (match `greedy.py` / `local_search.py`):

```python
"""ILP coverage-only oracle using OR-Tools CP-SAT. See OPTIMIZER_SPEC.md §7."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import networkx as nx
from ortools.sat.python import cp_model

from bike_rl.candidates import Candidate
from bike_rl.objective import ObjectiveWeights, objective
from bike_rl.optim.budget import cost
from bike_rl.optim.greedy import Solution

if TYPE_CHECKING:
    from bike_rl.config import Config

    _NXGraph = nx.MultiDiGraph[Any, Any, Any]
else:
    _NXGraph = nx.MultiDiGraph


_STATUS_NAMES = {
    cp_model.OPTIMAL: "OPTIMAL",
    cp_model.FEASIBLE: "FEASIBLE",
    cp_model.INFEASIBLE: "INFEASIBLE",
    cp_model.MODEL_INVALID: "MODEL_INVALID",
    cp_model.UNKNOWN: "UNKNOWN",
}

# CP-SAT linear coefficients must be integers. Scale float costs/budget by
# this and round; spent is recomputed from Candidate costs (float) for output.
_COST_SCALE = 1000.0


class ILPSolver:
    """Coverage-only ILP oracle (OPTIMIZER_SPEC §7).

    Solves the budgeted maximum-coverage problem: choose ``S ⊆ candidates``
    maximising the number of bike-lane-reachable graph nodes (the
    ``coverage_ratio`` of ``bike_rl.metrics.coverage`` in the radius branch)
    subject to ``Σ cost(e) ≤ budget``. This is the exact linearisable oracle
    the spec calls "maximise bike-lane-reachable population under budget".

    Reported ``Solution.objective`` is the shared canonical
    :func:`bike_rl.objective.objective` of the chosen ``S`` (OPTIMIZER_SPEC
    §8), so the ILP row is directly comparable to greedy / local-search / RL
    rows. The optimality *guarantee*, however, is over the **coverage_ratio
    surrogate** (``Solution.extra["surrogate"]`` / ``surrogate_count``) — not
    over the full objective. To use the ILP as an optimality ceiling in the
    §10 table, run it AND the heuristic solvers with coverage-only weights
    ``ObjectiveWeights(connectivity=0.0, coverage=1.0, fragmentation=0.0)``
    and compare the reachable-node count; report the optimality gap against
    the ILP.

    Deterministic — no seed. Only ``coverage_mode == "radius"`` is supported
    here (the component branch and the full-objective ILP with connectivity
    via flow/Steiner vars are deferred to a follow-up plan per §7).

    Engine: OR-Tools CP-SAT only. ``solver="ortools"`` (or ``None``) selects
    it; any other name raises ``ValueError``. PuLP/CBC and gurobipy fallbacks
    are deferred.
    """

    def __init__(
        self,
        cfg: Config,
        weights: ObjectiveWeights,
        solver: str | None = None,
        time_limit_s: float | None = None,
    ) -> None:
        engine = solver if solver is not None else cfg.ilp_default_solver
        if engine != "ortools":
            raise ValueError(
                f"ILPSolver: unsupported engine {engine!r}; only 'ortools' "
                "is implemented (PuLP/gurobipy deferred)."
            )
        self.cfg = cfg
        self.weights = weights
        self.engine = engine
        self.time_limit_s = (
            cfg.ilp_time_limit_s if time_limit_s is None else time_limit_s
        )

    def solve(
        self,
        graph: _NXGraph,
        candidates: list[Candidate],
        budget: float,
    ) -> Solution:
        """Solve the coverage-only ILP and return a :class:`Solution`.

        Args:
            graph: The base bike network graph (not mutated).
            candidates: Candidate edges to consider (input is not mutated).
            budget: Total budget; never exceeded.

        Returns:
            A ``Solution`` with ``solver="ilp"`` and ``extra`` carrying
            ``status`` (OPTIMAL/FEASIBLE/INFEASIBLE/UNKNOWN), ``objective_kind``
            (``"coverage_only"``), ``surrogate`` (coverage_ratio of chosen S),
            ``surrogate_count`` (integer reachable-node count), ``n_candidates``,
            ``n_selected``, ``solver_engine``, ``coverage_mode``, and
            ``opt_gap`` (0.0 when status is OPTIMAL).
        """
        ...  # see Step 2 body below
```

Body of `solve` (implement exactly this logic):

1. Record `start = time.perf_counter()`.
2. If `self.cfg.coverage_mode != "radius"`: raise `NotImplementedError` with a message naming the mode and pointing at the deferred full-objective ILP (do NOT silently fall back).
3. `total = graph.number_of_nodes()`. If `total == 0` or `not candidates` or `budget <= 0`: return an empty `Solution` (`solver="ilp"`, `extra` with `status="OPTIMAL"`, `objective_kind="coverage_only"`, `surrogate=0.0` if `total==0` else `base_reachable/total`, `surrogate_count=base_reachable`, `n_candidates=len(candidates)`, `n_selected=0`, `solver_engine="ortools"`, `coverage_mode=self.cfg.coverage_mode`, `opt_gap=0.0`). Compute `base_reachable` once via the same helper used in the model (Step 2.4).
4. Precompute coverage relation (the linearisable radius structure):
   - `base_bike = {n for u, v, d in graph.edges(data=True) if d.get("bike_lane") == "yes" for n in (u, v)}`.
   - For each node `n` in `graph.nodes(data=True)`: `free = any(_metres_between(ndata, graph.nodes[bn]) <= self.cfg.coverage_radius_m for bn in base_bike)`.
   - For each node `n` and each candidate index `j`: `covers` = `_metres_between(ndata, graph.nodes[c.u]) <= R` OR `_metres_between(ndata, graph.nodes[c.v]) <= R`. Collect `near[n] = [j, ...]`. (Import `_metres_between`: `from bike_rl.metrics import _metres_between`.)
5. Build the CP-SAT model:
   - `model = cp_model.CpModel()`
   - `x = [model.NewBoolVar(f"x{j}") for j in range(len(candidates))]`
   - `y = {n: model.NewBoolVar(f"y{n}") for n in graph.nodes()}` (use the node id; if node ids are not strings, format safely — `f"y{repr(n)}"` or keep an index dict `node_idx`).
   - Coverage constraints (max-coverage LP):
     - for a node `n` with `free == True`: `model.Add(y[n] == 1)`.
     - for a node `n` with `free == False`: `model.Add(y[n] <= sum(x[j] for j in near[n]))`. (When `near[n]` is empty, this is `y[n] <= 0`, i.e. never covered — correct.)
   - Budget constraint (integer-scaled): `costs_int = [int(round(cost(c, self.cfg) * _COST_SCALE)) for c in candidates]`; `budget_int = int(round(budget * _COST_SCALE))`; `model.Add(sum(costs_int[j] * x[j] for j in range(len(candidates))) <= budget_int)`.
   - Objective: `model.Maximize(sum(y[n].values()))` (maximise the number of reachable nodes). Add a **tiny secondary term to break ties deterministically** in favour of cheaper / higher-road-priority selections, so the chosen edge SET (not just the count) is deterministic across runs/specs:
     - `model.Maximize(sum(y.values()) * 100_000 - sum(costs_int[j] // 100 for j in range(len(candidates))) * x[j] - sum(x[j]))` — i.e. lex-maximise (reachable count) ≫ (lower cost) ≫ (fewer edges). Keep the secondary terms small enough (factor `100_000`) that they never flip the primary count by one. **Document this tie-break in a comment.** (If implementing the combined objective is awkward, maximise `sum(y.values())` only and instead enforce determinism by fixing lexicographic preferences via search hints — but the weighted scalar is simpler and recommended.)
   - Set the time limit: `model.parameters.max_time_in_seconds = self.time_limit_s`. Also set `model.parameters.num_search_workers = 1` for full determinism (CP-SAT is otherwise nondeterministic across thread counts).
6. Solve: `solver = cp_model.CpSolver(); status_code = solver.Solve(model)`. Wrap the call in a broad try/except that catches `RuntimeError` from OR-Tools and re-raises as `RuntimeError("ILPSolver: OR-Tools failed: <msg>")` — do NOT swallow.
7. Read results:
   - `status = _STATUS_NAMES.get(status_code, "UNKNOWN")`.
   - `chosen_idx = [j for j in range(len(candidates)) if solver.Value(x[j]) == 1]`.
   - `chosen = [candidates[j] for j in chosen_idx]`.
   - `spent = sum(cost(c, self.cfg) for c in chosen)` (float, from the real costs — NOT the scaled ints).
   - `surrogate_count = _reachable_count(graph, chosen, self.cfg)` where `_reachable_count` is a module-local helper with the SAME body as the test's `_reachable_count` (Step 1) — factor it once here and import/reuse in the test if you prefer, but keep it private. `surrogate = surrogate_count / total` (guard `total > 0`).
   - `opt_gap`: if `status == "OPTIMAL"` → `0.0`; elif `status == "FEASIBLE"` and `solver.ObjectiveValue() > 0` → `abs(solver.BestObjectiveBound() - solver.ObjectiveValue()) / abs(solver.ObjectiveValue())`; else `1.0` (or `float("nan")` for INFEASIBLE/UNKNOWN — pick one and document; recommended: `1.0` for INFEASIBLE, `float("nan")` for UNKNOWN).
   - Note: with the weighted scalar objective in step 5, `solver.ObjectiveValue()` is the weighted sum, not the raw count — so compute `opt_gap` from the **bound on the primary count** if available, else report `0.0` on OPTIMAL and `nan`/`1.0` otherwise. Keep `opt_gap` semantics documented; tests only assert `opt_gap == 0.0` when `status == "OPTIMAL"`.
8. Return:
```python
return Solution(
    edges=list(chosen),
    objective=objective(graph, list(chosen), self.weights, self.cfg),
    spent=spent,
    runtime_s=time.perf_counter() - start,
    solver="ilp",
    extra={
        "status": status,
        "objective_kind": "coverage_only",
        "surrogate": surrogate,
        "surrogate_count": surrogate_count,
        "n_candidates": len(candidates),
        "n_selected": len(chosen),
        "solver_engine": self.engine,
        "coverage_mode": self.cfg.coverage_mode,
        "opt_gap": opt_gap,
    },
)
```

Define the module-local `_reachable_count(graph, chosen, cfg)` helper at module level (private, leading underscore) — it is the single source of truth the ILP and the test both use; the test MAY import it (`from bike_rl.optim.ilp import _reachable_count`) instead of duplicating, to guarantee parity. If you do that, the test's own `_reachable_count` becomes a thin re-export. **Recommended**: define it once in `ilp.py` and import it in the test.

**Verify**:
- `ruff check bike_rl/optim/ilp.py` → exit 0
- `ruff format --check bike_rl/optim/ilp.py` → exit 0 (run `ruff format` if needed)
- `mypy --strict bike_rl` → exit 0 (add narrow `# type: ignore[...]` only where OR-Tools typing genuinely fails; justify each in a comment)

### Step 3: Write the `test_ilp.py` test bodies

Add the `TestILPSolver` class to `tests/test_ilp.py` (after the fixtures/helpers from Step 1). Tests (one per method, mirror `test_local_search.py` naming):

1. `test_matches_brute_force_on_tiny_instance` — **the core §9/§12 oracle test**. Parametrise over `budget in (600.0, 1000.0, 1200.0, 1800.0)` and the corresponding verified brute-force reachable counts `expected_count in (5, 5, 7, 9)`. For each budget: solve with `ILPSolver(default_cfg, default_weights)`; compute `best, _ = _brute_best(graph, candidates, default_cfg, budget)`; assert `sol.extra["surrogate_count"] == best == expected_count`; assert `sol.spent <= budget + 1e-9`. **Do NOT** assert a specific edge set at budget 1200 (multiple disjoint pairs are optimal — assert only the count). Use:
   ```python
   @pytest.mark.parametrize("budget, expected", [(600.0, 5), (1000.0, 5), (1200.0, 7), (1800.0, 9)])
   def test_matches_brute_force_on_tiny_instance(self, max_coverage_instance, default_cfg, default_weights, budget, expected):
       graph, candidates = max_coverage_instance
       sol = ILPSolver(default_cfg, default_weights).solve(graph, candidates, budget)
       best, _ = _brute_best(graph, candidates, default_cfg, budget)
       assert sol.extra["surrogate_count"] == best == expected
       assert sol.spent <= budget + 1e-9
   ```
2. `test_disjoint_optimum_at_budget_1200` — assert that at budget 1200 the ILP's chosen set is not a contiguous chain: the reachable count is 7 AND the chosen edges are pairwise disjoint (no shared endpoint) — i.e. it picked a max-coverage-with-disjoint-endpoints global solution, not a greedy-style chain. (Two disjoint 600-cost edges each covering 2 new far nodes.)
3. `test_respects_budget_never_overspends` — for budgets `(1.0, 600.0, 1200.0, 1e9)`: `sol.spent <= budget + 1e-9`.
4. `test_respects_time_limit_zero` — `ILPSolver(cfg, weights, time_limit_s=0.0)`: returns a `Solution` (status may be UNKNOWN or FEASIBLE; assert it does NOT crash and `runtime_s >= 0.0`). Do NOT assert optimality at time limit 0.
5. `test_optimal_status_and_zero_gap_on_small_instance` — `ILPSolver(default_cfg, default_weights, time_limit_s=10.0).solve(graph, candidates, 1800.0)`; assert `sol.extra["status"] == "OPTIMAL"`; assert `sol.extra["opt_gap"] == 0.0`; assert `sol.solver == "ilp"`.
6. `test_deterministic_across_runs` — solve twice at budget 1200; assert identical `edges`, `spent`, `extra["surrogate_count"]`, `extra["status"]`. (Determinism comes from `num_search_workers = 1` + the weighted tie-break.)
7. `test_does_not_mutate_input_candidates` — snapshot the list; solve; assert unchanged.
8. `test_solution_record_is_well_formed` — assert `isinstance(sol, Solution)`, `sol.solver == "ilp"`, `sol.runtime_s >= 0.0`, `extra` has all documented keys with correct types (`status` str, `surrogate_count` int, `n_candidates == 8`, `n_selected == len(sol.edges)`, `solver_engine == "ortools"`, `coverage_mode == "radius"`).
9. `test_objective_consistency_with_shared_scorer` — solve with `cov_only_weights`; assert `sol.objective == pytest.approx(objective(graph, sol.edges, cov_only_weights, default_cfg))`. (The ILP's reported `Solution.objective` equals a fresh `objective()` recompute on its edges — the §8 comparability guarantee.)
10. `test_empty_when_no_affordable_edge` — budget `1.0` with the costly-but-also-present candidates: actually use candidates whose every cost `> 1e9`? No — use budget `0.0` (or `1.0`) so nothing is affordable; assert `sol.edges == []`, `sol.spent == 0.0`, `extra["surrogate_count"] == 3` (only the base {1,2,3} reachable), `extra["status"] == "OPTIMAL"`.
11. `test_empty_candidates_returns_empty` — `ILPSolver(...).solve(graph, [], 1e9)`; assert `sol.edges == []`, `sol.extra["n_candidates"] == 0`, `sol.extra["status"] == "OPTIMAL"`, `sol.objective == pytest.approx(objective(graph, [], default_weights, default_cfg))`.
12. `test_surrogate_matches_metrics_coverage_ratio` — for the ILP's chosen S at budget 1800, recompute the reachable fraction with `_metres_between` directly in the test and assert it equals `sol.extra["surrogate"]` within `1e-9` AND that the ILP's reachable fraction ≥ the base `coverage_ratio` (sanity that adding edges never decreases reachable population). This pins the ILP surrogate to `metrics.coverage`'s radius definition.
13. `test_unsupported_engine_raises` — `ILPSolver(default_cfg, default_weights, solver="gurobi")` raises `ValueError`. (`solver="ortools"` and `solver=None` must NOT raise — covered by every other test implicitly.)
14. `test_non_radius_mode_raises` — build a `Config` with `coverage_mode="component"` (use `dataclasses.replace(default_cfg, coverage_mode="component")`; import `dataclasses.replace`); `ILPSolver(cfg, default_weights).solve(graph, candidates, 1800.0)` raises `NotImplementedError`. (Confirms the deferred-mode guard fires loudly, not silently.)

**Verify**:
- `pytest -q tests/test_ilp.py` → all pass (14 tests; counts may split with parametrisation — `test_matches_brute_force_on_tiny_instance` expands to 4).
- `pytest -q` → full suite green, coverage line shows `bike_rl/optim/ilp.py` at ≥80% (it will be ~100%; the only uncovered branches should be the `MODEL_INVALID`/`UNKNOWN` status arms which are hard to hit — if coverage of `ilp.py` specifically is <80% and that drags the **whole-package** gate below 80%, add a `# pragma: no cover` to the genuinely-unreachable status arms and document why; the whole-package gate is `--cov-fail-under=80` on `bike_rl`, currently ~89%, so a small stub addition will not break it).

## Test plan

- New file: `tests/test_ilp.py` — the 14 tests above (with parametrisation, ~17 test cases).
- Structural pattern to follow: `tests/test_local_search.py` (module docstring, scoped fixtures, one `TestX` class, `pytest.approx` for floats, `assert ... - 1e-9` tolerances).
- The brute-force parity test (test 1) is the §12-DoD gate ("ILP matches brute force on ≤12-candidate instances"). The fixture has 8 candidates — well within the ≤12 budget.
- Determinism (test 6) relies on `num_search_workers = 1` and the weighted lex tie-break from Step 2.5; if it ever fails, that is a real determinism regression in the solver — do NOT weaken the assertion.

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `pytest -q tests/test_ilp.py` exits 0; new tests pass.
- [ ] `pytest -q` exits 0; whole-package coverage ≥80% (the CI `addopts` gate).
- [ ] `ruff check .` exits 0.
- [ ] `ruff format --check .` exits 0.
- [ ] `mypy --strict bike_rl` exits 0.
- [ ] `grep -n "NotImplementedError(\"Optimiser plan\")" bike_rl/optim/ilp.py` returns no matches.
- [ ] No files outside `bike_rl/optim/ilp.py` and `tests/test_ilp.py` are modified (`git status --short`).
- [ ] `plans/README.md` status row for plan 013 updated (TODO → DONE).

## STOP conditions

Stop and report back (do not improvise) if:

- The code at the locations in "Current state" doesn't match the excerpts (the codebase has drifted since `bc58ad2`).
- `from ortools.sat.python import cp_model` no longer imports (OR-Tools removed/uninstalled) — the engine assumption is void; do NOT swap in PuLP silently (it's not a dependency).
- `from bike_rl.metrics import _metres_between` raises `ImportError` (the private helper was renamed/removed in `metrics.py`). This means the ILP surrogate and `metrics.coverage` can no longer be guaranteed to share a definition — STOP and report so the plan can be realigned (do NOT duplicate the formula and risk drift).
- `bike_rl/optim/greedy.Solution`'s field set or constructor no longer matches the excerpt (later `evaluate.py` plan depends on it).
- CP-SAT reports `MODEL_INVALID` on the tiny fixture — indicates the integer scaling or a constraint is malformed in a way the plan didn't anticipate; report the solver status and the model.
- The weighted lex tie-break in Step 2.5 is not enough to make `test_deterministic_across_runs` pass across two OR-Tools versions — report; do NOT make the ILP nondeterministic to "fix" it.
- You find that matching brute force requires optimising something other than the reachable-node count (i.e. the spec's coverage-only objective is *not* linearisable as modelled here) — STOP; the design assumption is false.

## Maintenance notes

For the human/agent owning this code after it lands:

- **Later optimiser plan (§11 item 5: `evaluate.py`)** will call `ILPSolver.solve(...)` and place its `Solution` in the §10 table alongside greedy / local-search / RL. It will enforce `cfg.max_candidates_for_ilp` (filtering / refusing to call the ILP above the threshold) — that threshold is **not** this solver's concern. `evaluate.py` will also wire `evaluate_rl_policy` to score a trained PPO policy with the same `objective`.
- **Full-objective ILP follow-up** (§7 "Full-objective ILP") will extend `ILPSolver` (or a sibling `FullObjectiveILPSolver`) with connectivity via flow/Steiner aux variables and the `coverage_mode == "component"` linearisation, gated to small instances. When that lands, **remove** the `NotImplementedError` guard added in Step 2.2 (don't weaken it now).
- **Perf**: warm-starting CP-SAT with the greedy solution (`model.AddHint(x[j], 1)` for greedy-chosen edges) is a natural later optimisation for larger instances; not needed for the tiny fixtures here.
- **Determinism**: kept via `num_search_workers = 1` + the weighted lex objective. If OR-Tools is upgraded and determinism regresses on a platform, the lex-objective weights (`* 100_000`) are the knob — raise the primary multiplier first.
- **Reviewer focus**: (a) the budget integer-scaling (no rounding can flip feasibility — `_COST_SCALE=1000` and `round` suffice for metres; confirm no candidate cost has sub-millimetre precision that would tie-break wrong); (b) the `opt_gap` semantics (documented; tests only assert `0.0` on OPTIMAL); (c) the `coverage_mode != "radius"` guard raising loudly rather than silently approximating.
- **Known upstream caveat** (from plan 012): `GreedySolver.solve` raises `TypeError` on exact `(delta/cost, road_priority, -cost)` ties (`max(scored)` falls through to comparing `Candidate`, which is not orderable). The `max_coverage_instance` fixture here deliberately creates such ties, so **do NOT** add an ILP-vs-greedy comparison test on this fixture — it would crash greedy. A one-line greedy tie-break fix is recommended as a separate small follow-up. The ILP-vs-brute-force tests here are solver-agnostic and safe.