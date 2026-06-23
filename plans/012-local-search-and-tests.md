# Plan 012: Implement `optim/local_search.py` (LocalSearchSolver) + `test_local_search.py`

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**:
> `git diff --stat 9360b77..HEAD -- bike_rl/optim/local_search.py bike_rl/optim/greedy.py bike_rl/optim/budget.py bike_rl/objective.py bike_rl/candidates.py bike_rl/config.py bike_rl/metrics.py bike_rl/optim/__init__.py tests/conftest.py`
> If any in-scope or dependency file changed since this plan was written,
> compare the "Current state" excerpts against the live code before
> proceeding; on a mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: LOW
- **Depends on**: plans/011-greedy-and-tests.md (DONE — `GreedySolver` +
  `Solution` dataclass in `bike_rl/optim/greedy.py`, which LocalSearchSolver
  seeds from and reuses), plans/010-objective-and-tests.md (DONE —
  `objective`/`ObjectiveWeights` in `bike_rl/objective.py`), plans/002
  (DONE — `Candidate` + `candidate_cost`), plans/003 (DONE — `metrics.py`
  used transitively by `objective`)
- **Category**: tech-debt (implements the second direct-optimiser solver,
  the local-search polish that closes most of the gap to the ILP ceiling)
- **Planned at**: commit `9360b77`, 2026-06-24
- **Issue**: (not published)

## Why this matters

`OPTIMIZER_SPEC.md` §6 defines **greedy + local search** as the second of
three baselines: it seeds from a `GreedySolver` solution (plan 011) and
polishes it with 1-opt swaps and 2-opt exchanges until a local optimum is
reached. Today `bike_rl/optim/local_search.py` is a stub — both
`LocalSearchSolver.__init__` and `solve` raise `NotImplementedError` — so
the §10 comparison table is missing its "Greedy + LS" row, and the optimality
gap that characterises *when* RL wins cannot be measured. This is item #3 of
`OPTIMIZER_SPEC.md` §11.

Landing this plan delivers:

- A **deterministic** local-search solver that reuses the shared `objective`
  (plan 010) and `GreedySolver` seed (plan 011) — so RL, greedy, and local
  search all solve *exactly the same problem* and report the same number.
- The 2-opt neighbourhood (drop one chosen edge, add two cheaper ones that
  fit the freed budget), which greedy-by-ratio cannot reach on its own —
  this is where local search earns its gap-closing claim (verified on a
  concrete fixture below: greedy 0.302 → local search 0.655).
- A populated `Solution` record (`solver="local_search"`, `extra` carrying
  `iterations`/`seed_objective`) that the later `evaluate.py` harness
  (§11 item 5) will consume alongside greedy/ILP/RL rows.

The implementation is **correctness-first and deterministic**: it follows
the pseudocode in `OPTIMIZER_SPEC.md` §6 verbatim, uses first-improvement
pivot (the spec's default), needs no RNG/seed, and never returns a solution
worse than its greedy seed (it only accepts strict improvements).

## Current state

### `bike_rl/optim/local_search.py` (stub — the file you will implement)

```python
"""Local search baseline. See OPTIMIZER_SPEC.md §6."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bike_rl.config import Config
    from bike_rl.objective import ObjectiveWeights
    from bike_rl.optim.greedy import Solution


class LocalSearchSolver:
    """Local search over candidate swaps, initialised from a GreedySolver solution."""

    def __init__(
        self,
        cfg: Config,
        weights: ObjectiveWeights,
        max_iter: int | None = None,
        time_limit_s: float | None = None,
    ) -> None:
        raise NotImplementedError("Optimiser plan")

    def solve(self, graph: object, candidates: list[object], budget: float) -> Solution:
        raise NotImplementedError("Optimiser plan")
```

Keep the `__init__` signature **exactly** (`max_iter: int | None = None`,
`time_limit_s: float | None = None`) — `bike_rl/optim/__init__.py` already
imports `LocalSearchSolver`, and the `None` defaults mean "fall back to
`cfg.local_search_max_iter` / `cfg.local_search_time_limit_s`". Tighten the
`solve` signature's annotations to the real types (see Step 1 target).

### What local search depends on (already implemented — do NOT change)

- **`bike_rl/optim/greedy.py`** — `GreedySolver(cfg, weights).solve(graph,
  candidates, budget) -> Solution` and the `Solution` dataclass. Excerpt of
  the `Solution` record (lines 22–37) the executor must populate:
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
  `GreedySolver.solve` is deterministic, copies the candidates list
  internally (`remaining = list(candidates)`), and returns `Solution` with
  `solver="greedy"`, `extra={"n_candidates": ..., "n_selected": ...}`.
- **`bike_rl/objective.py`** — `objective(graph, added_edges, weights, cfg)
  -> float` (full recompute) and `ObjectiveWeights` dataclass
  `(connectivity=0.4, coverage=0.4, fragmentation=0.2)`. Higher is better.
- **`bike_rl/optim/budget.py`** — `cost(edge, cfg) -> float` (delegates to
  `candidate_cost` = `length * cfg.edge_cost_factor`; factor is `10.0`).
- **`bike_rl/candidates.py`** — `Candidate` frozen dataclass; equality is on
  `(u, v, length, road_priority)` only (so `list.remove` / `set` membership
  work, and two candidates with the same identity quadruple are equal). Has
  `.u`, `.v`, `.length`, `.road_priority`, `.data`.
- **`bike_rl/config.py`** — `Config` already carries the local-search knobs
  (lines ~108–110):
  ```python
  local_search_max_iter: int = 1000
  local_search_time_limit_s: float = 60.0
  ```
  Use these as the defaults when `max_iter`/`time_limit_s` are `None`.

### Repo conventions to match

- **Typing**: `from __future__ import annotations` at the top; `TYPE_CHECKING`
  imports for `Config`/`ObjectiveWeights`/`Candidate` and the `nx` graph type
  alias. Follow the `greedy.py` pattern exactly:
  ```python
  if TYPE_CHECKING:
      import networkx as nx
      from bike_rl.candidates import Candidate
      from bike_rl.config import Config
      from bike_rl.objective import ObjectiveWeights
      _NXGraph = nx.MultiDiGraph[Any, Any, Any]
  else:
      _NXGraph = nx.MultiDiGraph
  ```
  (`from typing import Any` for the `Any, Any, Any` subscript — see
  `bike_rl/optim/greedy.py` lines 6–17.)
- **Docstrings**: Google convention (ruff `D` rules enforced). Every public
  class/method gets a docstring with a summary line + `Args:`/`Returns:`
  where non-trivial. See `bike_rl/optim/greedy.py` for the house style.
- **Tests**: model after `tests/test_greedy.py` — class-based grouping
  (`class TestLocalSearchSolver:`), a local `_edge(u, v, length, rp=1)`
  helper that builds a `Candidate`, `pytest` fixtures for `Config()` /
  `ObjectiveWeights()`, and `1e-9` / `pytest.approx` float tolerances. The
  shared graph fixtures `tiny_bike_graph` and `tiny_walk_graph` live in
  `tests/conftest.py` — **reuse them; do not redefine them and do not edit
  `conftest.py`**. Fixture-specific graphs for the 2-opt test are defined
  **locally inside `tests/test_local_search.py`** (see Step 2).
- **Determinism**: local search is deterministic given its greedy seed and
  takes no seed/RNG argument. Iterate candidates in the order they are
  passed in (do not shuffle or sort). Use **first-improvement** (break on
  the first improving move in a pass) — this matches the §6 pseudocode,
  which `break`s out of the inner loops on the first improvement.
- **No input mutation**: copy the candidates list before any membership
  testing (`cand = list(candidates)`); never call `.remove` on the caller's
  list. `greedy.py` already follows this pattern (`remaining = list(...)`).

## Commands you will need

| Purpose   | Command                              | Expected on success |
|-----------|--------------------------------------|---------------------|
| Tests     | `pytest`                             | exit 0, all pass, coverage ≥80% (addopts in `pyproject.toml` already pass `--cov=bike_rl --cov-fail-under=80`) |
| One file  | `pytest tests/test_local_search.py -v` | exit 0, all new tests pass |
| Lint      | `ruff check .`                       | exit 0, no errors |
| Format    | `ruff format --check .`              | exit 0, no diffs |
| Typecheck | `mypy --strict bike_rl`              | exit 0, no errors |

(All verified during recon — these are the exact commands the repo uses. Run
from the repo root with the project venv active: `source .venv/bin/activate`
if needed.)

## Scope

**In scope** (the only files you should modify):
- `bike_rl/optim/local_search.py` (implement `LocalSearchSolver`)
- `tests/test_local_search.py` (create)

**Out of scope** (do NOT touch, even though they look related):
- `bike_rl/optim/greedy.py` — plan 011, DONE. If you find a bug in greedy
  (see "Known upstream issue" below), **do not fix it here**; stop and report.
- `bike_rl/optim/__init__.py` — already imports `LocalSearchSolver`; no
  change needed (the import works once the stub is implemented).
- `bike_rl/optim/budget.py`, `bike_rl/objective.py`, `bike_rl/candidates.py`,
  `bike_rl/config.py`, `bike_rl/metrics.py` — dependencies; read-only.
- `tests/conftest.py` — shared fixtures; reuse `tiny_bike_graph` /
  `tiny_walk_graph` but do not edit it. Define any extra graphs locally in
  `tests/test_local_search.py`.
- Any change to the public `Solution` shape or `LocalSearchSolver.__init__`
  signature — later plans (`evaluate.py`, §11 item 5) depend on both.

## Git workflow

- Branch: `advisor/012-local-search-and-tests` (matching the
  `advisor/NNN-<slug>` convention used by plans 009–011).
- Commit per logical unit (implementation, then tests); message style:
  conventional commits — `feat(optim): implement LocalSearchSolver with
  1-opt/2-opt neighbourhoods` and `test(optim): add local search test suite`.
  See `git log --oneline` for the established prefixes
  (`feat(optim):`, `test(optim):`, `docs(plans):`).
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Implement `bike_rl/optim/local_search.py`

Replace the stub with a full implementation following `OPTIMIZER_SPEC.md` §6
pseudocode verbatim. Algorithm:

1. **Seed**: run `GreedySolver(self.cfg, self.weights).solve(graph,
   candidates, budget)` to get the starting `Solution`. Copy its `edges`
   into a working `S: list[Candidate] = list(seed.edges)` and track
   `spent = seed.spent`. Copy the candidates into `cand = list(candidates)`
   (working list for membership tests; the caller's list is never mutated).
2. **Local-search loop** — first-improvement, bounded by `max_iter` and
   `time_limit_s`:
   ```
   improved = True
   iterations = 0
   while improved and iterations < self.max_iter and (now - start) < self.time_limit_s:
       improved = False
       iterations += 1
       cur = objective(graph, S, weights, cfg)
       Sset = set(S)
       non_chosen = [e for e in cand if e not in Sset]
       # 1-opt: replace one chosen edge with one non-chosen edge
       for e_out in S:
           freed = cost(e_out, cfg)
           head = budget - (spent - freed)        # budget available after dropping e_out
           for e_in in non_chosen:
               if cost(e_in, cfg) <= head + EPS:
                   new_S = [e for e in S if e is not e_out] + [e_in]
                   if objective(graph, new_S, weights, cfg) > cur + EPS:
                       S = new_S
                       spent = spent - freed + cost(e_in, cfg)
                       improved = True
                       break
           if improved:
               break
       if improved:
           continue
       # 2-opt: drop one chosen edge, add two non-chosen fitting the freed budget
       for e_out in S:
           freed = cost(e_out, cfg)
           head = budget - (spent - freed)
           for e1, e2 in itertools.combinations(non_chosen, 2):
               if cost(e1, cfg) + cost(e2, cfg) <= head + EPS:
                   new_S = [e for e in S if e is not e_out] + [e1, e2]
                   if objective(graph, new_S, weights, cfg) > cur + EPS:
                       S = new_S
                       spent = spent - freed + cost(e1, cfg) + cost(e2, cfg)
                       improved = True
                       break
           if improved:
               break
   ```
   Where `EPS = 1e-9`, `start = time.perf_counter()` (set before the greedy
   seed so `runtime_s` covers the whole solve), and `now = time.perf_counter()`.
3. **Return** a `Solution`:
   ```python
   return Solution(
       edges=list(S),
       objective=objective(graph, S, self.weights, self.cfg),
       spent=spent,
       runtime_s=time.perf_counter() - start,
       solver="local_search",
       extra={
           "n_candidates": len(candidates),
           "n_selected": len(S),
           "iterations": iterations,
           "seed_solver": "greedy",
           "seed_objective": seed.objective,
       },
   )
   ```

Key correctness details (all verified during recon):

- **`__init__` defaults**: when `max_iter is None` use
  `cfg.local_search_max_iter`; when `time_limit_s is None` use
  `cfg.local_search_time_limit_s`. Store both on `self`.
- **Identity drop** (`e is not e_out`): `S` holds the actual `Candidate`
  objects returned by `GreedySolver`, which are the original objects from the
  input `candidates` list (greedy does `remaining = list(candidates)` and
  appends from it). Identity comparison correctly removes only the dropped
  edge even if two candidates compare equal by value. Do **not** use
  `list.remove(e_out)` — that removes by value and could drop the wrong
  equal candidate.
- **`non_chosen` excludes all of `S`** (via `set(S)` value-equality), so a
  1-opt/2-opt move can never re-add an edge already chosen (no no-op swaps).
- **Budget math**: `head = budget - (spent - freed)` = remaining budget after
  dropping `e_out`. 1-opt requires `cost(e_in) <= head`; 2-opt requires
  `cost(e1)+cost(e2) <= head`. This never overspends: after the swap,
  `spent' = spent - freed + cost(incoming) <= spent - freed + head = budget`.
- **Determinism**: no RNG; candidate iteration order is the passed-in order;
  first-improvement `break`s make each pass deterministic. The greedy seed
  is itself deterministic (plan 011).
- **Never worse than the seed**: the loop only accepts moves with
  `objective > cur + EPS`, so `objective(S)` is monotonically
  non-decreasing across passes and starts at `seed.objective`.
- **Empty seed**: if greedy returns `S = []` (no affordable improving edge),
  both `for e_out in S` loops are no-ops, `improved` stays `False`, and the
  solver returns the empty seed. This is correct, not a bug.
- **mypy --strict**: annotate locals — `S: list[Candidate]`,
  `cand: list[Candidate]`, `non_chosen: list[Candidate]`, `spent: float`,
  `iterations: int`, `freed: float`, `head: float`, `cur: float`. Import
  `itertools` and `time` at the top. `itertools.combinations(non_chosen, 2)`
  yields `tuple[Candidate, Candidate]` — no extra annotation needed.

Target shape (full file):

```python
"""Local search baseline: greedy seed + 1-opt/2-opt polish. See OPTIMIZER_SPEC.md §6."""

from __future__ import annotations

import itertools
import time
from typing import TYPE_CHECKING, Any

import networkx as nx

from bike_rl.objective import ObjectiveWeights, objective
from bike_rl.optim.budget import cost
from bike_rl.optim.greedy import GreedySolver, Solution

if TYPE_CHECKING:
    from bike_rl.candidates import Candidate
    from bike_rl.config import Config

    _NXGraph = nx.MultiDiGraph[Any, Any, Any]
else:
    _NXGraph = nx.MultiDiGraph


class LocalSearchSolver:
    """Greedy seed refined by 1-opt swaps and 2-opt exchanges (OPTIMIZER_SPEC §6).

    Seeds from :class:`GreedySolver`, then repeatedly applies first-improvement
    moves until a local optimum is reached or ``max_iter``/``time_limit_s``
    bound it. Neighbourhoods: 1-opt (replace one chosen edge with one
    non-chosen edge fitting the freed budget) and 2-opt (drop one chosen edge,
    add two non-chosen edges fitting the freed budget). Deterministic given
    the greedy seed — no RNG.

    Never returns a solution worse than its greedy seed: only strict
    objective improvements (``> cur + 1e-9``) are accepted.
    """

    def __init__(
        self,
        cfg: Config,
        weights: ObjectiveWeights,
        max_iter: int | None = None,
        time_limit_s: float | None = None,
    ) -> None:
        self.cfg = cfg
        self.weights = weights
        self.max_iter = cfg.local_search_max_iter if max_iter is None else max_iter
        self.time_limit_s = (
            cfg.local_search_time_limit_s if time_limit_s is None else time_limit_s
        )

    def solve(
        self,
        graph: _NXGraph,
        candidates: list[Candidate],
        budget: float,
    ) -> Solution:
        """Run greedy + local search and return a :class:`Solution`.

        Args:
            graph: The base bike network graph (not mutated).
            candidates: Candidate edges to consider (copied internally; the
                input list is not mutated).
            budget: Total budget; never exceeded.

        Returns:
            A ``Solution`` with ``solver="local_search"`` and ``extra``
            carrying ``n_candidates``/``n_selected``/``iterations``/
            ``seed_solver``/``seed_objective``.
        """
        start = time.perf_counter()
        seed = GreedySolver(self.cfg, self.weights).solve(graph, candidates, budget)
        S: list[Candidate] = list(seed.edges)
        cand: list[Candidate] = list(candidates)
        spent = seed.spent
        eps = 1e-9
        iterations = 0
        improved = True
        while (
            improved
            and iterations < self.max_iter
            and (time.perf_counter() - start) < self.time_limit_s
        ):
            improved = False
            iterations += 1
            cur = objective(graph, S, self.weights, self.cfg)
            sset = set(S)
            non_chosen: list[Candidate] = [e for e in cand if e not in sset]
            # 1-opt: replace one chosen edge with one non-chosen edge.
            for e_out in S:
                freed = cost(e_out, self.cfg)
                head = budget - (spent - freed)
                for e_in in non_chosen:
                    if cost(e_in, self.cfg) <= head + eps:
                        new_s = [e for e in S if e is not e_out]
                        new_s.append(e_in)
                        if objective(graph, new_s, self.weights, self.cfg) > cur + eps:
                            S = new_s
                            spent = spent - freed + cost(e_in, self.cfg)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                continue
            # 2-opt: drop one chosen edge, add two non-chosen edges.
            for e_out in S:
                freed = cost(e_out, self.cfg)
                head = budget - (spent - freed)
                for e1, e2 in itertools.combinations(non_chosen, 2):
                    if cost(e1, self.cfg) + cost(e2, self.cfg) <= head + eps:
                        new_s = [e for e in S if e is not e_out]
                        new_s.append(e1)
                        new_s.append(e2)
                        if objective(graph, new_s, self.weights, self.cfg) > cur + eps:
                            S = new_s
                            spent = spent - freed + cost(e1, self.cfg) + cost(e2, self.cfg)
                            improved = True
                            break
                if improved:
                    break
        return Solution(
            edges=list(S),
            objective=objective(graph, S, self.weights, self.cfg),
            spent=spent,
            runtime_s=time.perf_counter() - start,
            solver="local_search",
            extra={
                "n_candidates": len(candidates),
                "n_selected": len(S),
                "iterations": iterations,
                "seed_solver": "greedy",
                "seed_objective": seed.objective,
            },
        )
```

**Verify**:
- `ruff check bike_rl/optim/local_search.py` → exit 0, no errors.
- `ruff format --check bike_rl/optim/local_search.py` → exit 0, no diffs
  (if it reports diffs, run `ruff format bike_rl/optim/local_search.py` and
  re-check).
- `mypy --strict bike_rl` → exit 0, no errors.

### Step 2: Create `tests/test_local_search.py`

Model the file after `tests/test_greedy.py` (same `_edge` helper, same
`default_cfg` / `default_weights` fixtures, class-based grouping). Reuse the
shared `tiny_bike_graph` / `tiny_walk_graph` fixtures from `conftest.py` for
the general-behaviour tests, and define the 2-opt-escape graph **locally** in
this file (do not edit `conftest.py`).

Tests to write (one `def test_...` per bullet, grouped under
`class TestLocalSearchSolver:` except the budget-independent ones):

1. **`test_never_worse_than_greedy_seed`** — For budgets
   `(1.0, 1500.0, 3000.0, 1e9)` on `tiny_bike_graph` + candidates from
   `extract_candidates(tiny_bike_graph, tiny_walk_graph, cfg)`, the
   `LocalSearchSolver` objective is `>=` the `GreedySolver` objective on the
   same instance (tolerance `1e-9`). Also assert `spent <= budget + 1e-9`.
   This is the core §9 invariant.
2. **`test_2opt_escape_improves_on_greedy`** — The hand-built fixture below
   where greedy is **not** 2-opt-locally-optimal. Assert
   `ls.objective > greedy.objective + 1e-9` and
   `ls.extra["iterations"] >= 1`. This is the §9 "known local-optimum fixture
   escapes under 2-opt" test.
3. **`test_respects_max_iter_zero`** — `LocalSearchSolver(cfg, weights,
   max_iter=0)` on the 2-opt fixture returns the **greedy seed unchanged**:
   `ls.edges == greedy.edges`, `ls.objective == pytest.approx(greedy.objective)`,
   `ls.extra["iterations"] == 0`. (Loop condition `iterations < 0` is false
   immediately → 0 passes.)
4. **`test_respects_max_iter_one`** — `LocalSearchSolver(cfg, weights,
   max_iter=1)` on the 2-opt fixture runs **exactly one** improving pass:
   `ls.extra["iterations"] == 1` and
   `ls.objective == pytest.approx(0.655333, rel=1e-4)` (the first 2-opt
   move's objective, verified during recon). This proves the iteration cap
   binds.
5. **`test_respects_time_limit_zero`** — `LocalSearchSolver(cfg, weights,
   time_limit_s=0.0)` on the 2-opt fixture returns the greedy seed
   (`ls.extra["iterations"] == 0`, `ls.edges == greedy.edges`). The time gate
   `(now - start) < 0.0` is false on the first pass (greedy seed takes > 0s),
   so no iteration runs.
6. **`test_deterministic_across_runs`** — Two solves on `tiny_bike_graph` /
   `tiny_walk_graph` candidates at `budget=1e9` produce identical `edges`,
   `objective`, `spent`, and `iterations`.
7. **`test_does_not_mutate_input_candidates`** — Snapshot
   `extract_candidates(...)`; after `solve`, assert the input list is
   unchanged (same pattern as `test_greedy.py::test_does_not_mutate_input_candidates`).
8. **`test_solution_record_is_well_formed`** — `solver == "local_search"`,
   `runtime_s >= 0.0`, `extra["n_candidates"] == 4` (the `tiny_walk_graph`
   fixture yields 4 candidates), `extra["n_selected"] == len(ls.edges)`,
   `extra["iterations"]` is an `int >= 0`,
   `extra["seed_solver"] == "greedy"`, and `ls.objective == pytest.approx(
   objective(tiny_bike_graph, ls.edges, weights, cfg))` (the reported
   objective is a fresh recompute on the returned edges — the §8 fairness
   contract).
9. **`test_empty_seed_when_no_affordable_edge`** — Budget `1.0` on
   `tiny_bike_graph` with two costly hand-built candidates: greedy returns
   `[]`, local search returns `[]` with `iterations == 0` and
   `objective == pytest.approx(objective(graph, [], weights, cfg))`.

**The 2-opt-escape fixture (for tests 2, 3, 4, 5)** — define it as a local
`@pytest.fixture` named `two_opt_instance` returning
`(graph, candidates, budget)`. These exact values were verified during recon
to make greedy pick a single edge that 2-opt then improves:

```python
@pytest.fixture
def two_opt_instance() -> tuple[nx.MultiDiGraph, list[Candidate], float]:
    """Fixture where greedy is suboptimal and a 2-opt move improves it.

    Greedy picks only (3,2) [cost 2500, obj 0.302] because the remaining 500
    budget cannot afford any other candidate (cheapest is 1000). 2-opt drops
    (3,2) and adds (5,1)+(6,5) [combined cost 2500 <= 3000, obj 0.655333].
    No 1-opt move improves the seed, so this is a pure 2-opt escape.
    """
    g = nx.MultiDiGraph()
    coords = {
        1: (0.0322, 0.0352), 2: (0.027, 0.0491), 3: (0.0423, 0.0491),
        4: (0.0152, 0.0303), 5: (0.0326, 0.035),  6: (0.001, 0.0487),
    }
    for n, (x, y) in coords.items():
        g.add_node(n, x=x, y=y)
    g.add_edge(4, 2, length=120.0, highway="residential", bike_lane="yes")
    g.add_edge(1, 6, length=130.0, highway="residential", bike_lane="yes")
    candidates = [
        _edge(5, 1, 150.0, rp=5),  # cost 1500
        _edge(6, 5, 100.0, rp=1),  # cost 1000
        _edge(3, 2, 250.0, rp=2),  # cost 2500  <- greedy's only pick
        _edge(5, 3, 300.0, rp=3),  # cost 3000
        _edge(4, 5, 100.0, rp=5),  # cost 1000
    ]
    return g, candidates, 3000.0
```

For tests 3–5, compute the greedy reference inline:
```python
greedy = GreedySolver(default_cfg, default_weights).solve(graph, candidates, budget)
```
then run `LocalSearchSolver(...)` with the appropriate `max_iter` /
`time_limit_s` and assert against `greedy`.

**Verify**:
- `pytest tests/test_local_search.py -v` → exit 0, all 9 tests pass.
  Expected visible: `9 passed` (plus any parametrised expansions).
- `pytest --co -q tests/test_local_search.py` → lists the 9 test node ids
  with no collection errors (sanity check before running).

### Step 3: Run the full QA gate

Run the repo's full verification suite to confirm nothing regressed and the
≥80% coverage gate still passes:

- `ruff check .` → exit 0, no errors.
- `ruff format --check .` → exit 0, no diffs.
- `mypy --strict bike_rl` → exit 0, no errors.
- `pytest` → exit 0, all tests pass, and the coverage summary reports
  `bike_rl/optim/local_search.py` at **100%** (the stub's
  `NotImplementedError` lines are gone and every branch is exercised by the
  9 tests) with the total `--cov-fail-under=80` gate green.

**Expected `bike_rl/optim/local_search.py` coverage**: 100%. If `pytest`
reports missing lines in `local_search.py`, add a test that exercises the
uncovered branch (likely the 2-opt `break` path or the empty-seed path) and
re-run.

### Step 4: Update `plans/README.md`

In the execution-order table, change the `012` row's status from `TODO` to
`DONE`. (If no `012` row exists yet — the advisor adds it as part of this
plan — see the index update below.) Add a one-line dependency note under
"Dependency notes" recording that 012 seeds from `GreedySolver` (plan 011)
and reuses `objective` (plan 010), and that `local_search.py` populates
`Solution` with `solver="local_search"` + `extra["iterations"]`, which the
later `evaluate.py` (§11 item 5) will consume.

Do **not** mark 011 DONE — it is already DONE at HEAD `9360b77`.

## Test plan

- New tests: `tests/test_local_search.py` (9 tests, listed in Step 2).
  Cases covered: never-worse-than-seed (the §9 invariant), 2-opt escape
  (§9), iteration cap (`max_iter=0`/`1`), time cap (`time_limit_s=0.0`),
  determinism, no input mutation, well-formed `Solution`, empty-seed path.
- Structural pattern to model after: `tests/test_greedy.py` — same `_edge`
  helper, `default_cfg`/`default_weights` fixtures, class-based grouping,
  `pytest.approx`/`1e-9` tolerances, reuse of `tiny_bike_graph`/
  `tiny_walk_graph` from `tests/conftest.py`.
- Verification: `pytest tests/test_local_search.py -v` → all 9 pass; full
  `pytest` → all pass, coverage ≥80% and `local_search.py` at 100%.

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `ruff check .` exits 0
- [ ] `ruff format --check .` exits 0
- [ ] `mypy --strict bike_rl` exits 0
- [ ] `pytest` exits 0; the 9 new tests in `tests/test_local_search.py`
      exist and pass
- [ ] `pytest` coverage summary shows `bike_rl/optim/local_search.py` at
      100% and the `--cov-fail-under=80` gate is green
- [ ] `grep -n "NotImplementedError" bike_rl/optim/local_search.py`
      returns no matches
- [ ] No files outside the in-scope list are modified (`git status` shows
      only `bike_rl/optim/local_search.py`, `tests/test_local_search.py`,
      and `plans/README.md`)
- [ ] `plans/README.md` `012` status row updated to `DONE`

## STOP conditions

Stop and report back (do not improvise) if:

- The code at the locations in "Current state" doesn't match the excerpts
  (the codebase has drifted since this plan was written — re-run the drift
  check at the top).
- `GreedySolver` raises `TypeError: '>' not supported between instances of
  'Candidate' and 'Candidate'` during `solve`. This is a **known upstream
  issue in `greedy.py`** (plan 011): its `max(scored)` tie-break falls
  through to comparing `Candidate` objects, which are not orderable, when
  two candidates tie on `(delta/cost, road_priority, -cost)`. It is
  **out of scope** for this plan — do not edit `greedy.py`. The fixtures in
  this plan are tie-free so they will not trigger it; if it triggers on a
  fixture you add, change the fixture rather than fixing greedy. Report it
  so a separate one-line greedy fix can be planned.
- A step's verification fails twice after a reasonable fix attempt.
- The fix appears to require touching an out-of-scope file (e.g. you think
  you need to change `objective.py` or `greedy.py`).
- The 2-opt-escape fixture does **not** reproduce the expected numbers
  (`greedy.objective ≈ 0.302`, local-search `≈ 0.655333`). This means
  `objective` / `metrics` behaviour has drifted since the plan was written;
  do not hand-tune the fixture to force it — report the actual numbers.

## Maintenance notes

For the human/agent who owns this code after the change lands:

- **Performance**: `objective` is a full-recompute (plan 010,
  correctness-first). Each local-search pass calls `objective` once per
  improving-check (`O(|S|·|C|)` 1-opt checks + `O(|S|·|C|²)` 2-opt checks,
  each with one `objective` call). This is fine for small/medium cities but
  will be slow on large instances — the planned `MetricsState`-backed
  incremental `objective_delta` (deferred in plan 010) will speed this up
  without changing the public API. When that lands, swap the
  `objective(graph, new_s, ...)` calls inside the loops for an incremental
  delta; the acceptance test (`> cur + EPS`) stays the same.
- **Pivot rule**: this implementation uses **first-improvement** (break on
  first improving move). The spec mentions an optional best-improvement
  variant (§6). If added later, gate it behind a `Config` flag
  (`local_search_pivot: str = "first"`) rather than a new constructor
  argument, to keep the `__init__` signature stable for `evaluate.py`.
- **Reviewer focus**: (1) the identity-based drop (`e is not e_out`) —
  confirm it removes only the intended edge when two candidates compare
  equal by value; (2) the budget math (`head = budget - (spent - freed)`)
  — confirm `spent` never exceeds `budget` after any accepted move; (3) the
  `iterations` semantics — it counts **passes entered**, including the final
  non-improving pass that exits the loop, so `max_iter=1` allows exactly one
  improving pass (verify via `test_respects_max_iter_one`).
- **Known upstream issue (not fixed here)**: `GreedySolver.solve` raises
  `TypeError` on exact `(delta/cost, road_priority, -cost)` ties because
  `max(scored)` falls through to comparing `Candidate` (not orderable).
  Recommend a follow-up one-line fix in `greedy.py` (e.g. append a
  deterministic final tie-break key such as `id(e)` or the candidate's
  index in the input list, or use `max(scored, key=lambda t: t[:3])`). This
  protects `LocalSearchSolver` too, since it seeds from greedy.
- **Downstream**: `evaluate.py` (§11 item 5) will call
  `LocalSearchSolver(...).solve(...)` and read `Solution.extra["iterations"]`
  and `extra["seed_objective"]` for the §10 reporting table's "Greedy + LS"
  row — keep these `extra` keys stable.
