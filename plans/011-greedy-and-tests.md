# Plan 011: Implement `optim/greedy.py` (GreedySolver) + `optim/budget.py` + `test_greedy.py`

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**:
> `git diff --stat 54dc820..HEAD -- bike_rl/optim/greedy.py bike_rl/optim/budget.py bike_rl/optim/__init__.py bike_rl/objective.py bike_rl/candidates.py bike_rl/config.py bike_rl/metrics.py tests/conftest.py`
> If any in-scope or dependency file changed since this plan was written,
> compare the "Current state" excerpts against the live code before
> proceeding; on a mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: LOW
- **Depends on**: plans/010-objective-and-tests.md (DONE — `objective.py`
  provides `objective`/`objective_delta`/`apply_added_edges`/`ObjectiveWeights`/
  `Edge`, which greedy calls directly), plans/002-graph-utils-and-candidates.md
  (DONE — `Candidate` dataclass + `candidate_cost`, the cost single source of
  truth), plans/003-metrics-and-tests.md (DONE — `metrics.py` used transitively
  by `objective`)
- **Category**: tech-debt (implements the first real direct-optimiser solver,
  unblocking the optimiser baseline and the fair RL-vs-optimiser comparison)
- **Planned at**: commit `54dc820`, 2026-06-24
- **Issue**: (not published)

## Why this matters

`OPTIMIZER_SPEC.md` §5 defines the **greedy-by-marginal-gain-per-cost** solver
as the performance floor that any RL solution must beat to be interesting
(§1). Today `bike_rl/optim/greedy.py` is a stub — `GreedySolver.__init__` and
`solve` raise `NotImplementedError` — so the first and most important
optimiser baseline cannot run, and the §10 comparison table (the figure that
makes the RL contribution legible) cannot be produced. This is item #2 of
`OPTIMIZER_SPEC.md` §11.

Landing this plan delivers:

- A **deterministic**, theoretically-grounded greedy solver that reuses the
  shared `objective`/`objective_delta` (plan 010) and `candidate_cost`
  (plan 002) — so RL and the optimiser solve *exactly the same problem*.
- The `Solution` record (already defined as a dataclass) actually populated
  with `edges`/`objective`/`spent`/`runtime_s`/`solver`/`extra`, which the
  later `evaluate.py` harness (§11 item 5) and `local_search.py` (§11 item 3,
  which seeds from a `GreedySolver` solution) will import.
- The shared cost utility `optim/budget.py` (`cost`, `remaining_budget`),
  which `OPTIMIZER_SPEC.md` §2 places in the optim package and which greedy is
  the first consumer of. Implementing it here keeps `bike_rl/optim/*` off the
  0%-stub list that the ≥80% coverage gate (plan 008) deliberately does not
  exclude.

The implementation is **correctness-first and deterministic**: it follows the
pseudocode in `OPTIMIZER_SPEC.md` §5 verbatim, uses the existing full-recompute
`objective_delta` (a faster incremental version is explicitly deferred to a
later performance plan — see Maintenance notes), and needs no RNG/seed.

## Current state

The repo is a Python 3.10+ package `bike_rl` (see `pyproject.toml`). The RL
pipeline (plans 001–009) and the canonical objective (plan 010) are DONE. The
optimiser side is stubs except for `objective.py`.

### `bike_rl/optim/greedy.py` (stub — the main file you will implement)

Currently 33 lines. `Solution` is **already a correctly-shaped dataclass** —
keep it (you may tighten its field types, see Step 2). `GreedySolver` raises
`NotImplementedError` in both methods:

```python
"""Greedy selection baseline. See OPTIMIZER_SPEC.md §5."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bike_rl.config import Config
    from bike_rl.objective import ObjectiveWeights


@dataclass
class Solution:
    """A solver solution."""

    edges: list[object] = field(default_factory=list)
    objective: float = 0.0
    spent: float = 0.0
    runtime_s: float = 0.0
    solver: str = ""
    extra: dict[str, object] = field(default_factory=dict)


class GreedySolver:
    """Greedy selection: pick best candidate by objective delta, repeat until budget exhausted."""

    def __init__(self, cfg: Config, weights: ObjectiveWeights) -> None:
        raise NotImplementedError("Optimiser plan")

    def solve(self, graph: object, candidates: list[object], budget: float) -> Solution:
        raise NotImplementedError("Optimiser plan")
```

### `bike_rl/optim/budget.py` (stub — implement alongside greedy)

```python
"""Budget/cost utilities. See OPTIMIZER_SPEC.md."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bike_rl.config import Config


def cost(edge: object, cfg: Config) -> float:
    """Return the cost of an edge: length * edge_cost_factor."""
    raise NotImplementedError("Optimiser plan")


def remaining_budget(spent: float, budget: float) -> float:
    """Return remaining budget."""
    raise NotImplementedError("Optimiser plan")
```

### What greedy depends on (already implemented — do NOT change)

**`bike_rl/objective.py`** (plan 010, DONE) — the functions greedy calls. Key
signatures (verbatim from the file):

```python
@dataclass(frozen=True)
class ObjectiveWeights:
    connectivity: float = 0.4
    coverage: float = 0.4
    fragmentation: float = 0.2

class Edge(Protocol):  # satisfied by Candidate
    @property
    def u(self) -> int | str: ...
    @property
    def v(self) -> int | str: ...
    @property
    def length(self) -> float: ...
    @property
    def data(self) -> dict[str, Any]: ...

def objective(graph, added_edges, weights: ObjectiveWeights, cfg: Config) -> float: ...
def objective_delta(graph, added_edges, new_edge, weights: ObjectiveWeights, cfg: Config) -> float: ...
```

`objective_delta` is currently a full-recompute difference
(`objective(g, S+[e]) - objective(g, S)`) — **use it as-is**; do not try to
make greedy incremental. A `MetricsState`-backed incremental version is a
later perf plan.

**`bike_rl/candidates.py`** (plan 002, DONE) — the edge type and the cost
single source of truth. Verbatim:

```python
@dataclass(frozen=True)
class Candidate:
    u: int | str
    v: int | str
    length: float
    road_priority: int
    connects_to_bike_path: bool = field(compare=False)
    data: dict[str, Any] = field(default_factory=dict, compare=False)
    # Equality is on (u, v, length, road_priority) ONLY — so list.remove works.

def candidate_cost(c: Candidate, cfg: Config) -> float:
    """Single source of truth for cost — used by the env (§3.4) and the
    optimisers (OPTIMIZER_SPEC §4)."""
    return c.length * cfg.edge_cost_factor
```

`bike_rl/candidates.py`'s docstring explicitly states `candidate_cost` is the
single source of truth for cost. Therefore `optim/budget.py`'s `cost` **must
delegate** to `candidate_cost` (see Step 1) — do not re-derive `length *
edge_cost_factor` independently.

**`bike_rl/config.py`** (DONE) — `Config` is a frozen dataclass. Relevant
fields: `edge_cost_factor: float = 10.0`, `road_priorities: dict[str,int]`,
`min_candidate_length`, `max_candidate_length`. No new `Config` fields are
needed for this plan.

### Repo conventions to match

- **Typing**: `from __future__ import annotations` at the top; `TYPE_CHECKING`
  imports for `Config`/`ObjectiveWeights`/`Candidate`/`nx` graph types to
  avoid circular imports and runtime cost. For the `nx.MultiDiGraph` type
  alias, follow the `objective.py` pattern:
  ```python
  if TYPE_CHECKING:
      import networkx as nx
      from bike_rl.config import Config
      from bike_rl.candidates import Candidate
      from bike_rl.objective import ObjectiveWeights
      _NXGraph = nx.MultiDiGraph[Any, Any, Any]
  else:
      _NXGraph = nx.MultiDiGraph
  ```
  (Use `from typing import Any` for the `Any, Any, Any` subscript — see
  `bike_rl/objective.py` lines 12–18 for the exact pattern.)
- **Docstrings**: Google convention (ruff `D` rules enforced). Every public
  function/class/method gets a docstring with a summary line + `Args:`/`Returns:`
  where non-trivial. See `bike_rl/objective.py` and `bike_rl/candidates.py`
  for the house style.
- **Tests**: model after `tests/test_objective.py` — class-based grouping
  (`class TestGreedy...:`), a local `_edge(u, v, length, **extra)` helper that
  builds a `Candidate`, `pytest` fixtures for `Config()`/`ObjectiveWeights()`,
  and `1e-9` float tolerances. Shared graph fixtures `tiny_bike_graph` and
  `tiny_walk_graph` live in `tests/conftest.py` — reuse them; do not redefine.
- **Determinism**: greedy is deterministic and takes no seed. Tie-break by
  `(road_priority, -cost)` as the spec mandates — do not introduce `random`.

## Commands you will need

| Purpose   | Command                              | Expected on success |
|-----------|--------------------------------------|---------------------|
| Tests     | `pytest`                             | exit 0, all pass, coverage ≥80% (addopts in `pyproject.toml` already pass `--cov=bike_rl --cov-fail-under=80`) |
| One file  | `pytest tests/test_greedy.py -v`     | exit 0, all new tests pass |
| Lint      | `ruff check .`                       | exit 0, no errors |
| Format    | `ruff format --check .`              | exit 0, no diffs |
| Typecheck | `mypy --strict bike_rl`              | exit 0, no errors |

(All verified during recon — these are the exact commands the repo uses. Run
from the repo root with the project venv active: `source .venv/bin/activate`
if needed.)

## Scope

**In scope** (the only files you should modify):
- `bike_rl/optim/budget.py` — implement `cost` and `remaining_budget`.
- `bike_rl/optim/greedy.py` — implement `GreedySolver` (keep/extend the
  existing `Solution` dataclass).
- `tests/test_greedy.py` — create; greedy + budget unit tests.
- `plans/README.md` — update the status row for plan 011 (final step only).

**Out of scope** (do NOT touch, even though they look related):
- `bike_rl/optim/local_search.py` — stub; §11 item 3, a later plan. It will
  *consume* `GreedySolver` and `Solution` from this plan — keep their public
  API stable so that plan can import them unchanged.
- `bike_rl/optim/ilp.py` — stub; §11 item 4, a later plan.
- `bike_rl/optim/evaluate.py` — stub; §11 item 5, a later plan. It will
  consume `Solution` — keep the dataclass shape stable.
- `bike_rl/optim/__init__.py` — already imports `GreedySolver` and `Solution`
  correctly; do not change it.
- `bike_rl/objective.py`, `bike_rl/candidates.py`, `bike_rl/metrics.py`,
  `bike_rl/config.py` — dependencies; read-only.
- `tests/conftest.py` — reuse its fixtures; do not modify.
- The RL reward rewiring to `objective_delta` (§3.2) — a separate later
  concern; do not touch `env.py`/`reward`.
- Any CLI / `--compare` wiring — deferred to the §11 item 5/6 plans.

## Git workflow

- Branch: `advisor/011-greedy-and-tests` (matches the `advisor/NNN-<slug>`
  convention used by prior plans).
- Commit per logical unit (budget, greedy, tests). Message style: conventional
  commits, matching `git log --oneline` examples:
  `feat(optim): implement GreedySolver with budget util`,
  `test(optim): add greedy + budget test suite`.
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Implement `bike_rl/optim/budget.py`

Replace the two stub functions. `cost` delegates to the existing single source
of truth `bike_rl.candidates.candidate_cost`; `remaining_budget` is the
non-negative remaining headroom (never negative — a solver must not report it
can spend more than the budget).

Target shape:

```python
"""Budget/cost utilities shared by all optimiser solvers.

See OPTIMIZER_SPEC.md §2 and §4. ``cost`` delegates to
:func:`bike_rl.candidates.candidate_cost`, the single source of truth for
edge cost shared with the RL env, so the optimisers and the env always agree
on what an edge costs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bike_rl.candidates import candidate_cost

if TYPE_CHECKING:
    from bike_rl.candidates import Candidate
    from bike_rl.config import Config


def cost(edge: Candidate, cfg: Config) -> float:
    """Return the construction cost of ``edge``: ``length * edge_cost_factor``.

    Thin delegate to :func:`bike_rl.candidates.candidate_cost` (single source
    of truth — RECREATE_SPEC §3.4 / OPTIMIZER_SPEC §4) so the optimisers and
    the RL env never drift on cost.

    Args:
        edge: The candidate edge.
        cfg: Config providing ``edge_cost_factor``.

    Returns:
        ``edge.length * cfg.edge_cost_factor``.
    """
    return candidate_cost(edge, cfg)


def remaining_budget(spent: float, budget: float) -> float:
    """Return the non-negative remaining budget: ``max(budget - spent, 0.0)``.

    Clamped at zero so a solver that has overspent (or exactly hit the budget)
    reports no headroom rather than a negative number.

    Args:
        spent: Budget already consumed.
        budget: Total budget.

    Returns:
        ``max(budget - spent, 0.0)``.
    """
    return max(budget - spent, 0.0)
```

Note: `candidate_cost` is imported at runtime (not under `TYPE_CHECKING`)
because `cost` actually calls it. `Candidate`/`Config` stay under
`TYPE_CHECKING` for annotations only.

**Verify**:
- `ruff check bike_rl/optim/budget.py` → exit 0
- `ruff format --check bike_rl/optim/budget.py` → exit 0
- `mypy --strict bike_rl/optim/budget.py` → exit 0

### Step 2: Implement `bike_rl/optim/greedy.py`

Keep the `Solution` dataclass (tighten its `edges` field type to
`list[Candidate]` under `TYPE_CHECKING`, and `extra` stays `dict[str, object]`
— see the target below). Implement `GreedySolver` following
`OPTIMIZER_SPEC.md` §5 pseudocode verbatim:

- **Loop** while `remaining` is non-empty:
  - For each candidate `e` in `remaining`: compute `c = cost(e, cfg)`; skip if
    `spent + c > budget` (not affordable); compute
    `delta = objective_delta(graph, S, e, weights, cfg)`; skip if `delta <= 0`
    (no improvement); append `(score=delta/c, road_priority, -c, e)` to
    `scored`.
  - If `scored` is empty → break (no affordable improving edge).
  - Pick `best = max(scored)` — the tuple ordering gives the spec's tie-break:
    highest `delta/c`, then highest `road_priority`, then **lower** cost
    (because `-c` is larger for smaller `c`). Append `best` edge to `S`;
    `spent += cost(best, cfg)`; `remaining.remove(best)`.
- Return `Solution(edges=S, objective=objective(graph, S, weights, cfg),
  spent=spent, runtime_s=<elapsed>, solver="greedy", extra={"n_candidates":
  len(candidates), "n_selected": len(S)})`.

Target shape:

```python
"""Greedy selection baseline. See OPTIMIZER_SPEC.md §5."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import networkx as nx

from bike_rl.objective import Edge, ObjectiveWeights, objective, objective_delta
from bike_rl.optim.budget import cost

if TYPE_CHECKING:
    from bike_rl.candidates import Candidate
    from bike_rl.config import Config

    _NXGraph = nx.MultiDiGraph[Any, Any, Any]
else:
    _NXGraph = nx.MultiDiGraph


@dataclass
class Solution:
    """A solver solution (OPTIMIZER_SPEC §8).

    Attributes:
        edges: The selected candidate edges.
        objective: Canonical objective value of ``edges`` (shared with RL).
        spent: Total cost of ``edges``.
        runtime_s: Wall-clock solve time in seconds.
        solver: Solver name (e.g. ``"greedy"``).
        extra: Solver-specific metadata (e.g. candidate/selected counts).
    """

    edges: list[Any] = field(default_factory=list)
    objective: float = 0.0
    spent: float = 0.0
    runtime_s: float = 0.0
    solver: str = ""
    extra: dict[str, object] = field(default_factory=dict)


class GreedySolver:
    """Greedy-by-marginal-gain-per-cost selection (OPTIMIZER_SPEC §5).

    Each round, among affordable candidates with positive marginal objective
    gain, pick the one maximising ``objective_delta / cost``; tie-break by
    ``(road_priority, -cost)`` for determinism. Repeat until no affordable
    candidate improves the objective. Deterministic — no seed required.

    Under a cardinality constraint on a monotone submodular objective, greedy
    achieves ``(1 - 1/e)`` of optimal; under a budget constraint the standard
    bound is ``1/2(1 - 1/e)`` (in practice much closer). See OPTIMIZER_SPEC §5.
    """

    def __init__(self, cfg: Config, weights: ObjectiveWeights) -> None:
        self.cfg = cfg
        self.weights = weights

    def solve(
        self,
        graph: _NXGraph,
        candidates: list[Candidate],
        budget: float,
    ) -> Solution:
        """Run greedy selection and return a :class:`Solution`.

        Args:
            graph: The base bike network graph (not mutated).
            candidates: Candidate edges to consider (consumed internally via
                ``list.remove``; the input list is copied first).
            budget: Total budget; never exceeded.

        Returns:
            A ``Solution`` with the selected edges, shared objective value,
            spent cost, runtime, and counts in ``extra``.
        """
        start = time.perf_counter()
        S: list[Edge] = []
        remaining = list(candidates)
        spent = 0.0
        while remaining:
            scored: list[tuple[float, int, float, Edge]] = []
            for e in remaining:
                c = cost(e, self.cfg)
                if spent + c > budget:
                    continue
                delta = objective_delta(graph, S, e, self.weights, self.cfg)
                if delta <= 0.0:
                    continue
                scored.append((delta / c, e.road_priority, -c, e))
            if not scored:
                break
            _, _, _, best = max(scored)
            S.append(best)
            spent += cost(best, self.cfg)
            remaining.remove(best)
        runtime_s = time.perf_counter() - start
        return Solution(
            edges=list(S),
            objective=objective(graph, S, self.weights, self.cfg),
            spent=spent,
            runtime_s=runtime_s,
            solver="greedy",
            extra={"n_candidates": len(candidates), "n_selected": len(S)},
        )
```

Implementation notes:
- `remaining = list(candidates)` — copy so the caller's list is not mutated.
- `remaining.remove(best)` relies on `Candidate` equality being
  `(u, v, length, road_priority)` (plan 002) — confirmed in "Current state".
  This removes the first equal element, which is correct because duplicates are
  interchangeable.
- `e.road_priority` is used in the tie-break tuple; `Edge` (the protocol)
  does not declare `road_priority`, but `Candidate` does and the solver is
  typed `list[Candidate]`, so this is statically valid. The `Edge` protocol is
  used only for the `S` list annotation to stay consistent with `objective`'s
  `Sequence[Edge]` parameter.
- `cost(e, self.cfg)` is recomputed once per candidate per round and once more
  on selection — fine for correctness; the perf plan can cache it.

**Verify**:
- `ruff check bike_rl/optim/greedy.py` → exit 0
- `ruff format --check bike_rl/optim/greedy.py` → exit 0
- `mypy --strict bike_rl/optim/greedy.py` → exit 0
- `python -c "from bike_rl.optim import GreedySolver, Solution; print('ok')"` → prints `ok`

### Step 3: Create `tests/test_greedy.py`

Model after `tests/test_objective.py` (class-based, `_edge` helper, fixtures).
Reuse `tiny_bike_graph` and `tiny_walk_graph` from `tests/conftest.py`. The
spec's test requirements (`OPTIMIZER_SPEC.md` §9, `test_greedy.py` row):
*picks the obviously-correct edge on a hand-built tiny instance; respects
budget (never overspends); terminates when no affordable edge; deterministic
across runs.* Add budget-unit tests here too (the named test file is
`test_greedy.py`; folding the two small budget assertions in keeps one file
and covers `budget.py` for the cov gate).

Use `extract_candidates(tiny_bike_graph, tiny_walk_graph, cfg)` to build the
candidate set from the shared fixtures — this exercises the real
`Candidate`/cost path and matches how the solver is used in anger. The
`tiny_walk_graph` fixture (see `tests/conftest.py`) yields exactly three
valid candidates after filtering: `(2,5) length 200 primary`,
`(3,5) length 150 secondary`, `(4,5) length 600 tertiary` (the other walk
edges are skipped: `(1,2)` already in bike graph, `(1,5)` too short, `(2,3)`
too long, `(5,5)` self-loop, `(3,4)` already in bike graph).

Target tests (write all of these):

```python
"""Tests for the greedy solver and budget utilities (OPTIMIZER_SPEC §5, §9).

Covers: correct-edge selection on a tiny instance, budget respect, termination
on no-affordable-edge, determinism, the Solution contract, and the budget
cost/remaining helpers. See OPTIMIZER_SPEC.md §9 and plans/011-greedy-and-tests.md.
"""

from __future__ import annotations

import networkx as nx
import pytest

from bike_rl.candidates import Candidate, candidate_cost, extract_candidates
from bike_rl.config import Config
from bike_rl.objective import ObjectiveWeights, objective
from bike_rl.optim.budget import cost, remaining_budget
from bike_rl.optim.greedy import GreedySolver, Solution


# ── Helper ───────────────────────────────────────────────────────────────


def _edge(u: int | str, v: int | str, length: float, road_priority: int = 1) -> Candidate:
    """Build a Candidate satisfying the Edge protocol."""
    return Candidate(
        u=u,
        v=v,
        length=float(length),
        road_priority=road_priority,
        connects_to_bike_path=False,
        data={"length": float(length)},
    )


# ── Local fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def default_weights() -> ObjectiveWeights:
    """Default objective weights (0.4 / 0.4 / 0.2)."""
    return ObjectiveWeights()


@pytest.fixture
def default_cfg() -> Config:
    """Default config."""
    return Config()


# ── Budget utilities ────────────────────────────────────────────────────


class TestBudget:
    """Tests for optim/budget.py."""

    def test_cost_delegates_to_candidate_cost(
        self, default_cfg: Config
    ) -> None:
        """cost(edge, cfg) == candidate_cost(edge, cfg)."""
        e = _edge(1, 2, 200.0)
        assert cost(e, default_cfg) == candidate_cost(e, default_cfg)
        assert cost(e, default_cfg) == 200.0 * default_cfg.edge_cost_factor

    def test_remaining_budget_non_negative(self) -> None:
        """remaining_budget is max(budget - spent, 0.0)."""
        assert remaining_budget(0.0, 1000.0) == 1000.0
        assert remaining_budget(300.0, 1000.0) == 700.0
        assert remaining_budget(1000.0, 1000.0) == 0.0
        assert remaining_budget(1500.0, 1000.0) == 0.0  # clamped, not negative


# ── GreedySolver ────────────────────────────────────────────────────────


class TestGreedySolver:
    """Tests for GreedySolver (OPTIMIZER_SPEC §5, §9)."""

    def test_picks_obviously_correct_edge_first(
        self,
        tiny_bike_graph: nx.MultiDiGraph,
        tiny_walk_graph: nx.MultiDiGraph,
        default_cfg: Config,
        default_weights: ObjectiveWeights,
    ) -> None:
        """On a tight budget, greedy picks the best gain-per-cost edge.

        With a budget that affords only the cheapest candidate, greedy must
        select exactly that one and stop. The chosen edge must improve the
        shared objective over the empty selection (the performance floor).
        """
        candidates = extract_candidates(tiny_bike_graph, tiny_walk_graph, default_cfg)
        assert len(candidates) == 3, f"fixture should yield 3 candidates, got {len(candidates)}"

        # Budget affords only the cheapest candidate (150m * factor=10 = 1500).
        budget = candidate_cost(candidates[0], default_cfg)
        solver = GreedySolver(default_cfg, default_weights)
        sol = solver.solve(tiny_bike_graph, candidates, budget)

        assert len(sol.edges) == 1
        # The selected edge actually improves the objective vs empty selection.
        base_obj = objective(tiny_bike_graph, [], default_weights, default_cfg)
        assert sol.objective > base_obj + 1e-9

    def test_respects_budget_never_overspends(
        self,
        tiny_bike_graph: nx.MultiDiGraph,
        tiny_walk_graph: nx.MultiDiGraph,
        default_cfg: Config,
        default_weights: ObjectiveWeights,
    ) -> None:
        """spent <= budget for a range of budgets, including very tight ones."""
        candidates = extract_candidates(tiny_bike_graph, tiny_walk_graph, default_cfg)
        solver = GreedySolver(default_cfg, default_weights)
        for budget in (1.0, 1500.0, 5000.0, 1e9):
            sol = solver.solve(tiny_bike_graph, candidates, budget)
            assert sol.spent <= budget + 1e-9, (
                f"overspent budget={budget}: spent={sol.spent}"
            )
            # Every selected edge is individually within budget.
            for e in sol.edges:
                assert candidate_cost(e, default_cfg) <= budget + 1e-9

    def test_terminates_when_no_affordable_edge(
        self,
        tiny_bike_graph: nx.MultiDiGraph,
        default_cfg: Config,
        default_weights: ObjectiveWeights,
    ) -> None:
        """With budget below every candidate cost, greedy returns empty."""
        candidates = [
            _edge(2, 5, 200.0, road_priority=5),
            _edge(3, 5, 150.0, road_priority=4),
        ]
        # All costs are >= 150 * 10 = 1500; budget 1.0 affords none.
        solver = GreedySolver(default_cfg, default_weights)
        sol = solver.solve(tiny_bike_graph, candidates, 1.0)
        assert sol.edges == []
        assert sol.spent == 0.0
        assert sol.objective == pytest.approx(
            objective(tiny_bike_graph, [], default_weights, default_cfg)
        )

    def test_terminates_when_no_improving_edge(
        self,
        tiny_bike_graph: nx.MultiDiGraph,
        default_cfg: Config,
        default_weights: ObjectiveWeights,
    ) -> None:
        """A candidate whose addition does not improve the objective is skipped.

        A duplicate of an existing bike-lane edge has delta ~ 0 (plan 010's
        duplicate-edge guarantee), so greedy must not select it even with a
        huge budget.
        """
        # (1,2) is already a bike_lane edge in tiny_bike_graph.
        dup = _edge(1, 2, 120.0, road_priority=2)
        solver = GreedySolver(default_cfg, default_weights)
        sol = solver.solve(tiny_bike_graph, [dup], 1e9)
        assert sol.edges == []
        assert sol.spent == 0.0

    def test_deterministic_across_runs(
        self,
        tiny_bike_graph: nx.MultiDiGraph,
        tiny_walk_graph: nx.MultiDiGraph,
        default_cfg: Config,
        default_weights: ObjectiveWeights,
    ) -> None:
        """Two solves produce identical edge sets, objective, and spent."""
        candidates = extract_candidates(tiny_bike_graph, tiny_walk_graph, default_cfg)
        solver = GreedySolver(default_cfg, default_weights)
        sol1 = solver.solve(tiny_bike_graph, candidates, 1e9)
        sol2 = solver.solve(tiny_bike_graph, candidates, 1e9)
        assert sol1.edges == sol2.edges
        assert sol1.objective == pytest.approx(sol2.objective)
        assert sol1.spent == pytest.approx(sol2.spent)

    def test_does_not_mutate_input_candidates(
        self,
        tiny_bike_graph: nx.MultiDiGraph,
        tiny_walk_graph: nx.MultiDiGraph,
        default_cfg: Config,
        default_weights: ObjectiveWeights,
    ) -> None:
        """The candidates list passed in is unchanged after solve."""
        candidates = extract_candidates(tiny_bike_graph, tiny_walk_graph, default_cfg)
        snapshot = list(candidates)
        solver = GreedySolver(default_cfg, default_weights)
        solver.solve(tiny_bike_graph, candidates, 1e9)
        assert candidates == snapshot

    def test_solution_record_is_well_formed(
        self,
        tiny_bike_graph: nx.MultiDiGraph,
        tiny_walk_graph: nx.MultiDiGraph,
        default_cfg: Config,
        default_weights: ObjectiveWeights,
    ) -> None:
        """Solution has solver='greedy', non-negative runtime, and counts."""
        candidates = extract_candidates(tiny_bike_graph, tiny_walk_graph, default_cfg)
        sol = GreedySolver(default_cfg, default_weights).solve(
            tiny_bike_graph, candidates, 1e9
        )
        assert isinstance(sol, Solution)
        assert sol.solver == "greedy"
        assert sol.runtime_s >= 0.0
        assert sol.extra["n_candidates"] == 3
        assert sol.extra["n_selected"] == len(sol.edges)
        # objective field equals a fresh objective() recompute on the edges.
        assert sol.objective == pytest.approx(
            objective(tiny_bike_graph, sol.edges, default_weights, default_cfg)
        )

    def test_beats_random_selection_floor(
        self,
        tiny_bike_graph: nx.MultiDiGraph,
        tiny_walk_graph: nx.MultiDiGraph,
        default_cfg: Config,
        default_weights: ObjectiveWeights,
    ) -> None:
        """Greedy objective >= objective of any single-edge selection (§12 DoD).

        The greedy result must be at least as good as the best single-edge
        pick — the minimum bar. (On this fixture greedy selects >= 1 edge and
        its objective dominates every one-edge selection.)
        """
        candidates = extract_candidates(tiny_bike_graph, tiny_walk_graph, default_cfg)
        solver = GreedySolver(default_cfg, default_weights)
        sol = solver.solve(tiny_bike_graph, candidates, 1e9)
        best_single = max(
            objective(tiny_bike_graph, [e], default_weights, default_cfg)
            for e in candidates
        )
        assert sol.objective >= best_single - 1e-9
```

Notes for the executor:
- `pytest.approx` is fine for float equality of objective values recomputed
  via the same code path; use `1e-9` bare assertions only where a recompute is
  compared to the stored field (they are the same call, so exact equality is
  expected — but `pytest.approx` is safer and matches `test_objective.py`'s
  spirit). Pick one and be consistent within each test.
- Do not assert on specific `runtime_s` values beyond `>= 0.0` (timing is
  environment-dependent).
- Do not assert on the exact *number* of edges selected with a huge budget
  (the objective/fragmentation interaction makes that fragile); the
  `test_beats_random_selection_floor` and `deterministic` tests pin behaviour
  without over-constraining.

**Verify**:
- `pytest tests/test_greedy.py -v` → exit 0, all tests pass (8 tests across
  `TestBudget` + `TestGreedySolver`)

### Step 4: Run the full QA gate

Run the whole suite + lint + typecheck together, exactly as CI does.

**Verify** (all must pass):
- `pytest` → exit 0, all pass, and the coverage line reports `>=80%` (the
  `--cov-fail-under=80` addopt fails the run below 80%). Check the
  `bike_rl/optim/greedy.py` and `bike_rl/optim/budget.py` rows in the
  `--cov-report=term-missing` output: both should show high coverage (target
  ~100% for `budget.py`, ~95%+ for `greedy.py`).
- `ruff check .` → exit 0
- `ruff format --check .` → exit 0
- `mypy --strict bike_rl` → exit 0

If the coverage gate fails because of the still-stub `ilp.py` /
`local_search.py` / `evaluate.py` (0%), that is a STOP condition — report it;
do not omit those files from coverage and do not implement them here. (Recon
note: at HEAD `54dc820` overall `bike_rl` coverage is well above 80% with
those stubs present, so adding covered `greedy.py` + `budget.py` should raise
it further, not lower it.)

### Step 5: Update `plans/README.md`

Update the status table row for plan 011 to `DONE`, and update the
"Remaining optimiser plans" note to reflect that greedy is done (the next
items are `local_search.py` then `ilp.py` then `evaluate.py`). Add a row to
the status table if one does not yet exist for 011:

```
| 011  | `optim/greedy.py` (GreedySolver) + `optim/budget.py` + `test_greedy.py` (OPTIMIZER_SPEC §11.2) | P1 | M | 010 | DONE |
```

**Verify**: `grep -n "011" plans/README.md` → shows the new/updated row.

## Test plan

- **New file**: `tests/test_greedy.py` (created in Step 3).
- **Cases covered** (mapping to `OPTIMIZER_SPEC.md` §9 `test_greedy.py` row):
  - *picks the obviously-correct edge* → `test_picks_obviously_correct_edge_first`
  - *respects budget (never overspends)* → `test_respects_budget_never_overspends`
  - *terminates when no affordable edge* → `test_terminates_when_no_affordable_edge`
    + `test_terminates_when_no_improving_edge` (delta <= 0 skip)
  - *deterministic across runs* → `test_deterministic_across_runs`
  - plus Solution-contract, input-non-mutation, §12 DoD
    (beats-random-selection floor), and budget-unit tests.
- **Structural pattern**: `tests/test_objective.py` (class grouping, `_edge`
  helper, `Config()`/`ObjectiveWeights()` fixtures, `1e-9`/`pytest.approx`
  tolerances). Shared graph fixtures from `tests/conftest.py`.
- **Verification**: `pytest tests/test_greedy.py -v` → all pass; `pytest` →
  all pass with ≥80% coverage gate green.

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `pytest` exits 0; new tests in `tests/test_greedy.py` exist and pass
- [ ] `ruff check .` exits 0
- [ ] `ruff format --check .` exits 0
- [ ] `mypy --strict bike_rl` exits 0
- [ ] `bike_rl/optim/budget.py` `cost` delegates to `candidate_cost`
  (`grep -n "candidate_cost" bike_rl/optim/budget.py` shows the call) and
  `remaining_budget` clamps at zero
- [ ] `grep -n "NotImplementedError" bike_rl/optim/greedy.py
      bike_rl/optim/budget.py` returns no matches
- [ ] No files outside the in-scope list are modified (`git status` shows only
  `bike_rl/optim/budget.py`, `bike_rl/optim/greedy.py`,
  `tests/test_greedy.py`, `plans/README.md`, and the new commit on the
  `advisor/011-greedy-and-tests` branch)
- [ ] `plans/README.md` status row for 011 updated to `DONE`

## STOP conditions

Stop and report back (do not improvise) if:

- The code at the locations in "Current state" doesn't match the excerpts
  (the codebase has drifted since this plan was written) — in particular if
  `objective_delta` / `objective` signatures have changed, or `Candidate`
  equality is no longer on `(u, v, length, road_priority)`, or
  `candidate_cost` no longer exists in `bike_rl/candidates.py`.
- `extract_candidates(tiny_bike_graph, tiny_walk_graph, default_cfg)` does not
  return exactly 3 candidates — the `tiny_walk_graph` fixture has drifted; do
  not edit `conftest.py` to force a count, report it.
- The ≥80% coverage gate fails **because of** the still-stub
  `ilp.py`/`local_search.py`/`evaluate.py` (i.e. adding covered greedy/budget
  did not keep overall coverage ≥80%). Do not omit those files from coverage
  and do not implement them in this plan — report it.
- A step's verification fails twice after a reasonable fix attempt.
- The fix appears to require touching an out-of-scope file (e.g. you find
  greedy genuinely needs a new `Config` field, or a change to `objective.py`).
- You discover the assumption "greedy can be implemented purely on top of the
  existing `objective`/`objective_delta`/`candidate_cost` with no new
  dependencies" is false.

## Maintenance notes

For the human/agent who owns this code after the change lands:

- **`local_search.py` (§11 item 3) will consume this**: `LocalSearchSolver`
  seeds from `GreedySolver(cfg, weights).solve(...)` and improves over the
  `Solution`. Keep `GreedySolver.__init__(cfg, weights)` and
  `solve(graph, candidates, budget) -> Solution` stable. If you add optional
  constructor args (e.g. a pivot rule), default them so the existing call
  shape still works.
- **`evaluate.py` (§11 item 5) will consume `Solution`**: the harness scores
  every solver (and the RL policy) into a `Solution` and compares rows. Keep
  the `Solution` field set stable (`edges`, `objective`, `spent`, `runtime_s`,
  `solver`, `extra`); add fields only additively.
- **`objective_delta` is correctness-first (full recompute)**: greedy is
  `O(k · |C| · delta_eval)` where `delta_eval` is two full `objective`
  evaluations. This is fine for small/medium cities and for tests. A later
  **performance plan** should back `objective_delta` with `MetricsState`
  (union-find for fragmentation, persistent rx graph for connectivity — see
  `metrics.py` and `OPTIMIZER_SPEC.md` §3.1/§6.2/§6.3). When that lands,
  greedy needs **no change** — it already calls `objective_delta`; it will
  just get faster. Verify that plan's `objective_delta` still passes plan
  010's `test_objective_delta_matches_full_recompute`.
- **Submodularity caveat**: the `(1 - 1/e)` guarantee holds only if the
  objective is monotone submodular. `OPTIMIZER_SPEC.md` §5/§7 flag that some
  fragmentation definitions break submodularity. The research writeup must
  state this and verify submodularity empirically (plan 010's
  `TestSubmodularity` is the start). Greedy remains a valid heuristic either
  way; the guarantee is what's at stake.
- **Reviewer scrutiny**: check (a) `cost` delegates to `candidate_cost`
  rather than re-deriving `length * edge_cost_factor`; (b) the tie-break
  tuple `(delta/c, road_priority, -c, e)` with `max(...)` really does
  implement "highest ratio, then highest road_priority, then lower cost";
  (c) `spent + c > budget` uses `>` (not `>=`) so an exactly-affordable edge
  is still taken; (d) the input `candidates` list is copied, not mutated.
- **Deferred out of this plan**: incremental `objective_delta` (perf), the RL
  reward rewiring to `objective_delta` (§3.2), CLI `--compare` dispatch
  (§11.5/6), and the local-search/ILP/evaluate solvers.
