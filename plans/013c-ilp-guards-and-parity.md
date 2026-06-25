# Plan 013c: ILP guards, parity & edge-case tests

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.

> **Drift check (run first)**: `git diff --stat bc58ad2..HEAD -- bike_rl/optim/ilp.py bike_rl/metrics.py bike_rl/objective.py bike_rl/config.py tests/test_ilp.py`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts below against the live code before proceeding; on
> a mismatch, treat it as a STOP condition.

> **Split context**: This plan is the third of three that together replace
> the former monolithic `plans/013-ilp-and-tests.md`. It assumes **013a**
> (DONE) and **013b** (DONE) have landed: `bike_rl/optim/ilp.py` contains the
> working `ILPSolver` + `_reachable_count`, and `tests/test_ilp.py` already
> defines `max_coverage_instance`, `_brute_best`, `_edge`, the three weight
> fixtures, and `class TestILPSolver` with the 013a + 013b tests. **This plan
> only appends tests to `tests/test_ilp.py`.**

## Status

- **Priority**: P2
- **Effort**: S
- **Risk**: LOW
- **Depends on**: plans/013a-ilp-solver-core.md (DONE), plans/013b-ilp-determinism-tests.md (DONE) — imports `ILPSolver`, `_reachable_count`, `max_coverage_instance`, `_brute_best`, `_edge`, `default_cfg`, `default_weights`, `cov_only_weights` from the file 013a created
- **Category**: tests (guard & edge-case hardening; no production change expected)
- **Planned at**: commit `bc58ad2`, 2026-06-25
- **Issue**: (not published)

## Why this matters

013a proved the ILP matches brute force on a happy-path instance; 013b pinned
determinism / time-limit / well-formedness. This plan pins the **contract
boundaries** that, if violated, would silently corrupt the optimiser
comparison (`OPTIMIZER_SPEC.md` §8 / §10):

- The **`coverage_mode != "radius"` guard** raises a loud
  `NotImplementedError` rather than silently approximating (the full-objective
  ILP is deferred — §7). If this guard ever stops raising, downstream code
  may believe an ILP solution is exact when it is in fact an approximation.
- The **engine guard** rejects non-`ortools` engines so no one silently falls
  back to a non-dependency.
- The **§8 comparability guarantee** — `Solution.objective == objective(graph,
  sol.edges, weights, cfg)` after the solver runs — is what makes the ILP row
  and the greedy / local-search / RL rows directly comparable in the §10
  table.
- The **metrics parity guarantee** — the ILP surrogate equals
  `metrics.coverage`'s radius-branch reachable fraction on the same graph —
  is what makes the "optimality gap" meaningful (the ILP optimises the same
  quantity `coverage()` reports).
- The **edge cases** (empty candidate list, nothing affordable) must return a
  well-formed empty `Solution` rather than crashing, so `evaluate.py` can call
  the ILP on degenerate instances without special-casing.

## Current state

### Files and their roles
- `bike_rl/optim/ilp.py` — landed by 013a (and untouched by 013b unless a regression surfaced). Contains `ILPSolver` and module-local `_reachable_count`.
- `tests/test_ilp.py` — landed by 013a + 013b. Already contains:
  - module docstring, imports: `import itertools`, `import networkx as nx`,
    `import pytest`, `from bike_rl.candidates import Candidate, candidate_cost`,
    `from bike_rl.config import Config`, `from bike_rl.metrics import _metres_between`,
    `from bike_rl.objective import ObjectiveWeights`,
    `from bike_rl.optim.greedy import Solution` (added by 013b),
    `from bike_rl.optim.ilp import ILPSolver, _reachable_count`;
  - `_edge` helper; `default_cfg`, `default_weights`, `cov_only_weights` fixtures;
  - `max_coverage_instance` fixture; `_brute_best` helper;
  - `class TestILPSolver` with 013a's 3 tests + 013b's 5 tests.
- `bike_rl/objective.py` — `objective(graph, added_edges, weights, cfg)`; reuse (the §8 comparability test recomputes it). Do NOT modify.
- `bike_rl/metrics.py` — `_metres_between` and `coverage`; the parity test imports `_metres_between` (already imported). Do NOT modify.
- `bike_rl/config.py` — has `coverage_mode: str = "radius"` among the fields 013a already uses. Do NOT modify.

### Excerpt — ILPSolver guards (the behaviour this plan pins)
```python
# __init__:
engine = solver if solver is not None else cfg.ilp_default_solver
if engine != "ortools":
    raise ValueError(f"ILPSolver: unsupported engine {engine!r}; ...")

# solve:
if self.cfg.coverage_mode != "radius":
    raise NotImplementedError("...coverage_mode='...' not supported; full-objective ILP deferred...")
```

### Excerpt — `_reachable_count` (the shared helper, `bike_rl/optim/ilp.py`)
```python
def _reachable_count(graph, chosen, cfg) -> int:
    """Number of graph nodes within cfg.coverage_radius_m of any bike-lane endpoint."""
```
The descendant parity test will use `_metres_between` directly (already imported) to reproduce the reachable fraction independently.

### Conventions to match
- Mirror the existing `TestILPSolver` style in `tests/test_ilp.py` (pytest
  style, `pytest.approx` for floats, `pytest.raises` for the guard tests).
  Append new methods to the SAME `class TestILPSolver` — do NOT create a new
  class.
- ruff (`E,F,I,UP,B,SIM,D`; line-length 100) and `mypy --strict` are enforced.
  Add `dataclasses` (or `import dataclasses`) and `from bike_rl.objective import objective`
  to the import block if not already present — see Step 1 and Step 2.

## Commands you will need

Run all commands from the repo root with the venv active.

| Purpose | Command | Expected on success |
|---|---|---|
| Activate venv | `source .venv/bin/activate` | shell prompt changes |
| Run only the ILP tests | `pytest -q tests/test_ilp.py` | all pass (013a + 013b + 013c) |
| Full test suite | `pytest -q` | all pass, coverage ≥80% |
| Lint | `ruff check .` | exit 0 |
| Format check | `ruff format --check .` | exit 0 (run `ruff format .` if it reports diffs) |
| Type check | `mypy --strict bike_rl` | exit 0 |

## Scope

**In scope** (the only file you should modify):
- `tests/test_ilp.py` — append the tests listed below to `class TestILPSolver`, plus the minimal import additions (Step 1).

**Out of scope** (do NOT touch):
- `bike_rl/optim/ilp.py` and any production file — unless a test below proves a real regression in a guard/contract (per the STOP conditions). Guard tests failing means a 013a guard regressed; report, do NOT weaken the assertion.
- The `max_coverage_instance` fixture, `_brute_best`, the weight fixtures — owned by 013a.
- The full-objective ILP / `coverage_mode == "component"` support — explicitly deferred (one of these tests confirms the guard fires).

## Git workflow

- Branch: `advisor/013c-ilp-guards-and-parity`.
- Single commit is fine. Message: `test(optim): pin ILP engine/mode guards, objective & metrics parity, edge cases`.
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Add the minimal import additions the tests need

Confirm the current import block in `tests/test_ilp.py` (after 013a+013b). This
plan's tests use `dataclasses.replace`, `from bike_rl.objective import objective`,
and `from bike_rl.metrics import coverage` — the latter two are NOT in 013a's
import list (013a imported only `_metres_between` from `bike_rl.metrics`, and
`ObjectiveWeights` from `bike_rl.objective`). Add (do not duplicate):

```python
import dataclasses

from bike_rl.metrics import _metres_between, coverage
from bike_rl.objective import ObjectiveWeights, objective
```

`_metres_between` is already imported by 013a — keep it as part of the same
import line, do not add a second `from bike_rl.metrics import ...` line. If
`objective` and `ObjectiveWeights` are currently on the same line already
(013a imported only `ObjectiveWeights`), merge them:

```python
from bike_rl.objective import ObjectiveWeights, objective
```

`import dataclasses` goes with the stdlib imports at the top (`import
itertools` is already there). Run `ruff check --fix .` / `ruff format .` to
keep `I` (isort) ordering; verify the diff by eye before committing.

### Step 2: Append the 013c test methods to `class TestILPSolver`

Append these methods to the existing `class TestILPSolver`. Each uses
`max_coverage_instance`, `default_cfg`, `default_weights`, or `cov_only_weights`
fixtures from 013a.

1. `test_unsupported_engine_raises` — `ILPSolver(default_cfg, default_weights,
   solver="gurobi")` raises `ValueError`. Also assert that `solver="ortools"`
   and `solver=None` do NOT raise (construct only — solving is exercised by
   every other test). Pattern:
   ```python
   def test_unsupported_engine_raises(self, default_cfg, default_weights):
       with pytest.raises(ValueError):
           ILPSolver(default_cfg, default_weights, solver="gurobi")
       # these must NOT raise:
       ILPSolver(default_cfg, default_weights, solver="ortools")
       ILPSolver(default_cfg, default_weights, solver=None)
   ```
2. `test_non_radius_mode_raises` — build a `Config` with
   `coverage_mode="component"` via `dataclasses.replace(default_cfg,
   coverage_mode="component")`; `ILPSolver(cfg, default_weights).solve(graph,
   candidates, 1800.0)` raises `NotImplementedError`. Confirms the
   deferred-mode guard fires loudly, not silently. (Do NOT assert the message
   text verbatim — just that it is `NotImplementedError`.)
   ```python
   def test_non_radius_mode_raises(self, max_coverage_instance, default_cfg, default_weights):
       graph, candidates = max_coverage_instance
       cfg = dataclasses.replace(default_cfg, coverage_mode="component")
       with pytest.raises(NotImplementedError):
           ILPSolver(cfg, default_weights).solve(graph, candidates, 1800.0)
   ```
3. `test_empty_candidates_returns_empty` — `ILPSolver(default_cfg,
   default_weights).solve(graph, [], 1e9)`; assert `sol.edges == []`,
   `sol.extra["n_candidates"] == 0`, `sol.extra["status"] == "OPTIMAL"`,
   `sol.objective == pytest.approx(objective(graph, [], default_weights, default_cfg))`.
   Uses the `max_coverage_instance` graph (so the base coverage is non-trivial).
4. `test_empty_when_no_affordable_edge` — with budget `0.0` (nothing
   affordable since every candidate costs ≥ 600): assert `sol.edges == []`,
   `sol.spent == 0.0`, `sol.extra["surrogate_count"] == 3` (only the base
   {1,2,3} reachable), `sol.extra["status"] == "OPTIMAL"`. The empty selection
   is trivially optimal under zero budget.
   ```python
   def test_empty_when_no_affordable_edge(self, max_coverage_instance, default_cfg, default_weights):
       graph, candidates = max_coverage_instance
       sol = ILPSolver(default_cfg, default_weights).solve(graph, candidates, 0.0)
       assert sol.edges == []
       assert sol.spent == 0.0
       assert sol.extra["surrogate_count"] == 3
       assert sol.extra["status"] == "OPTIMAL"
   ```
5. `test_objective_consistency_with_shared_scorer` — the §8 comparability
   guarantee. Solve with `cov_only_weights`; assert
   `sol.objective == pytest.approx(objective(graph, sol.edges, cov_only_weights,
   default_cfg))`. The ILP's reported `Solution.objective` must equal a fresh
   `objective()` recompute on its edges. (Coverage-only weights make the
   comparison crisp: no connectivity/fragmentation terms to mask drift.)
   ```python
   def test_objective_consistency_with_shared_scorer(
       self, max_coverage_instance, default_cfg, cov_only_weights
   ):
       graph, candidates = max_coverage_instance
       sol = ILPSolver(default_cfg, cov_only_weights).solve(graph, candidates, 1800.0)
       assert sol.objective == pytest.approx(
           objective(graph, sol.edges, cov_only_weights, default_cfg)
       )
   ```
6. `test_surrogate_matches_metrics_coverage_ratio` — the metrics-parity pin.
   For the ILP's chosen `S` at budget 1800, independently recompute the
   reachable fraction with `_metres_between` directly in the test and assert
   it equals `sol.extra["surrogate"]` within `1e-9`; ALSO assert the ILP's
   reachable fraction ≥ the base `coverage_ratio` of `metrics.coverage`
   (sanity that adding edges never decreases reachable population). This
   pins the ILP surrogate to `metrics.coverage`'s radius definition.
   ```python
   def test_surrogate_matches_metrics_coverage_ratio(
       self, max_coverage_instance, default_cfg, default_weights
   ):
       graph, candidates = max_coverage_instance
       sol = ILPSolver(default_cfg, default_weights).solve(graph, candidates, 1800.0)
       # Independent recompute of the reachable fraction (radius branch of coverage()).
       chosen = sol.edges
       bike = {n for u, v, d in graph.edges(data=True)
               if d.get("bike_lane") == "yes" for n in (u, v)}
       for e in chosen:
           bike.add(e.u); bike.add(e.v)
       covered = 0
       for n, ndata in graph.nodes(data=True):
           for bn in bike:
               if _metres_between(ndata, graph.nodes[bn]) <= default_cfg.coverage_radius_m:
                   covered += 1
                   break
       total = graph.number_of_nodes()
       assert sol.extra["surrogate"] == pytest.approx(covered / total, abs=1e-9)
       # Adding edges never decreases reachable population.
       base_fraction = coverage(graph, default_cfg)
       assert sol.extra["surrogate"] >= base_fraction - 1e-9
   ```

**Verify**:
- `ruff check .` → exit 0
- `ruff format --check .` → exit 0
- `mypy --strict bike_rl` → exit 0 (test additions are not under `bike_rl/`).
- `pytest -q tests/test_ilp.py` → all pass (013a's 3 + 013b's 5 + this plan's 6 = 14 tests; with 013a's parametrisation, ~18 cases — the original monolithic 013 target count).
- `pytest -q` → full suite green, coverage still ≥80%.

## Test plan

- Append 6 methods to `class TestILPSolver` in `tests/test_ilp.py`.
- Minimal import additions (Step 1): `dataclasses`, `coverage`, `objective`.
- Zero production change expected.

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `pytest -q tests/test_ilp.py` exits 0; the 6 new tests pass alongside 013a+013b's tests (total 14 test methods).
- [ ] `pytest -q` exits 0; whole-package coverage ≥80% (unchanged — `ilp.py` not modified).
- [ ] `ruff check .` exits 0.
- [ ] `ruff format --check .` exits 0.
- [ ] `mypy --strict bike_rl` exits 0.
- [ ] No file outside `tests/test_ilp.py` is modified (`git status --short`).
- [ ] `plans/README.md` status row for plan 013c updated (TODO → DONE).

## STOP conditions

Stop and report back (do not improvise) if:

- `tests/test_ilp.py`, `bike_rl/optim/ilp.py`, `bike_rl/metrics.py`, `bike_rl/objective.py`, or `bike_rl/config.py` no longer matches "Current state" (drift since `bc58ad2` / 013a-013b landing).
- `test_unsupported_engine_raises` finds the engine guard no longer raises `ValueError` for `"gurobi"` — the 013a guard regressed; report. Do NOT re-add the guard in the test.
- `test_non_radius_mode_raises` finds `ILPSolver.solve` no longer raises `NotImplementedError` for `coverage_mode="component"` (i.e. it silently solves or approximates) — the 013a deferred-mode guard regressed; report. Do NOT weaken the assertion.
- `test_objective_consistency_with_shared_scorer` finds `sol.objective !=
  objective(graph, sol.edges, weights, cfg)` — the §8 comparability contract
  broke (013a's `solve` no longer recomputes `objective` on its `chosen`);
  report. Do NOT "fix" by overwriting `sol.objective` in the test.
- `test_surrogate_matches_metrics_coverage_ratio` finds the ILP surrogate
  diverges from the independent `_metres_between` recompute — this means
  `_reachable_count` in `ilp.py` and `metrics.coverage`'s radius branch have
  drifted apart (the single source of truth broke). Report; do NOT paper over
  by recomputing the surrogate differently in the test.
- `test_empty_when_no_affordable_edge` finds `surrogate_count != 3` (the base
  reachable set changed shape) — the fixture or `_reachable_count` drifted;
  report rather than re-tuning the magic number.
- You find that any test requires editing a production file to pass — STOP and report which one and why; 013c is meant to be test-only.

## Maintenance notes

- **Guards**: `test_unsupported_engine_raises` and
  `test_non_radius_mode_raises` exist so the deferred PuLP/full-objective ILP
  follow-up ships by *removing* a guard, and that removal is visible in the
  test suite (these tests will then need updating as part of the follow-up
  plan — that is the intended signal, not a leak).
- **§8 comparability** (`test_objective_consistency_with_shared_scorer`) is
  the contract `evaluate.py` (§11 item 5) relies on to put the ILP row in the
  §10 table next to greedy / local-search / RL. If it regresses, the table is
  comparing non-comparable numbers.
- **Metrics parity** (`test_surrogate_matches_metrics_coverage_ratio`) ties
  the ILP's optimisation target to the metric the RL reward and `evaluate.py`
  both read; drift here invalidates any "optimality gap" reported against the
  ILP.
- **Reviewer focus**: (a) the guard tests use bare `pytest.raises(ValueError)` /
  `pytest.raises(NotImplementedError)` — do NOT assert message text (the
  message is documentation, not contract); (b) the empty-input tests assert the
  SOLVER returns well-formed empty output, leaving `evaluate.py` free to
  special-case degenerate instances at the harness layer, not the solver.
- After 013c, all 14 tests the original monolithic 013 planned are landed; the
  ILP family (013a/b/c) is complete and the next optimiser plan is
  `evaluate.py` (§11 item 5), which depends on 013a/b/c all being DONE.