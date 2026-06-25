# Plan 013b: ILP determinism, time-limit & well-formedness tests

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.

> **Drift check (run first)**: `git diff --stat bc58ad2..HEAD -- bike_rl/optim/ilp.py tests/test_ilp.py`
> If either file changed since this plan was written, compare the "Current
> state" excerpts below against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

> **Split context**: This plan is the second of three that together replace
> the former monolithic `plans/013-ilp-and-tests.md`. It assumes plan **013a**
> has landed (DONE): `bike_rl/optim/ilp.py` contains the working `ILPSolver`
> and `tests/test_ilp.py` already defines the `max_coverage_instance` fixture,
> `_brute_best`, `_edge`, `default_cfg`/`default_weights`/`cov_only_weights`,
> and imports `_reachable_count` from `bike_rl.optim.ilp`. **This plan only
> appends tests to `tests/test_ilp.py`.** The third plan, 013c, adds the
> guard/parity/edge-case tests.

## Status

- **Priority**: P2
- **Effort**: S
- **Risk**: LOW
- **Depends on**: plans/013a-ilp-solver-core.md (DONE) — imports `ILPSolver`, `_reachable_count`, `max_coverage_instance`, `_brute_best`, `_edge`, `default_cfg`, `default_weights` from the file 013a created
- **Category**: tests (test hardening; no production change expected)
- **Planned at**: commit `bc58ad2`, 2026-06-25
- **Issue**: (not published)

## Why this matters

The brute-force parity tests in 013a prove the ILP *matches* the optimum on a
small instance; they do NOT prove the solver is **deterministic across runs**,
**honours its time limit**, or **produces a well-formed `Solution` record** for
the future `evaluate.py` harness (OPTIMIZER_SPEC §11 item 5 / §10 table) to
consume. Those properties are the difference between an oracle you can trust
as a ceiling and one that only "happens to match" once. This plan pins the
non-parity properties a follow-on `evaluate.py` row will silently depend on.

If any of these tests fail, that is a **real regression in the solver**, not a
test bug — report it (do NOT weaken the assertions or make the ILP
nondeterministic to "fix" it).

## Current state

### Files and their roles
- `bike_rl/optim/ilp.py` — landed by 013a. Contains `ILPSolver` and the
  module-local `_reachable_count(graph, chosen, cfg)`. Do NOT modify unless a
  test below surfaces a real regression in the solver (see "Scope").
- `tests/test_ilp.py` — landed by 013a. Already contains:
  - module docstring, imports (`import itertools`, `networkx as nx`,
    `pytest`, `Candidate`/`candidate_cost`, `Config`, `_metres_between`,
    `ObjectiveWeights`, `ILPSolver`, `_reachable_count`);
  - `_edge(u, v, length, road_priority=1) -> Candidate`;
  - `default_cfg`, `default_weights`, `cov_only_weights` fixtures;
  - `max_coverage_instance` fixture (8 candidates, 9 nodes, base coverage 3/9;
    brute optima: budget 600→5, 1000→5, 1200→7, 1800→9);
  - `_brute_best(graph, candidates, cfg, budget)`;
  - `class TestILPSolver` with the three 013a tests (`test_matches_brute_force_on_tiny_instance`, `test_respects_budget_never_overspends`, `test_optimal_status_and_zero_gap_on_small_instance`).

### Excerpt — `ILPSolver.__init__` / `solve` signature (reuse unchanged)
```python
def __init__(self, cfg, weights, solver: str | None = None, time_limit_s: float | None = None) -> None: ...
def solve(self, graph, candidates: list[Candidate], budget: float) -> Solution: ...
```
`Solution.extra` keys (set by 013a): `status`, `objective_kind`,
`surrogate`, `surrogate_count`, `n_candidates`, `n_selected`, `solver_engine`,
`coverage_mode`, `opt_gap`.

### Excerpt — `_reachable_count` (the shared helper, `bike_rl/optim/ilp.py`)
```python
def _reachable_count(graph, chosen, cfg) -> int:
    """Number of graph nodes within cfg.coverage_radius_m of any bike-lane endpoint."""
```

### Conventions to match
- Mirror the existing `TestILPSolver` style in `tests/test_ilp.py` (pytest
  style, `pytest.approx` for floats, `assert ... + 1e-9` tolerances). Append
  new test methods to the SAME `TestILPSolver` class — do NOT create a second
  class.
- ruff (`E,F,I,UP,B,SIM,D`; line-length 100) and `mypy --strict` are enforced
  (plan 008). Append-only to the test file should not import anything new
  beyond what 013a already imported.

## Commands you will need

Run all commands from the repo root with the venv active.

| Purpose | Command | Expected on success |
|---|---|---|
| Activate venv | `source .venv/bin/activate` | shell prompt changes |
| Run only the ILP tests | `pytest -q tests/test_ilp.py` | all pass (013a + 013b tests) |
| Full test suite | `pytest -q` | all pass, coverage ≥80% |
| Lint | `ruff check .` | exit 0 |
| Format check | `ruff format --check .` | exit 0 (run `ruff format .` if it reports diffs) |
| Type check | `mypy --strict bike_rl` | exit 0 |

## Scope

**In scope** (the only file you should modify):
- `tests/test_ilp.py` — append the tests listed in Step 1 to `class TestILPSolver`.

**Out of scope** (do NOT touch):
- `bike_rl/optim/ilp.py` — unless a test below fails in a way that proves a
  real solver regression (per the STOP conditions). A failed determinism
  assertion most likely means the 013a `num_search_workers = 1` setting or the
  weighted lex tie-break is insufficient; fix in `ilp.py` ONLY by raising the
  lex-objective primary multiplier (`* 100_000`) or confirming
  `num_search_workers = 1` is set — do NOT relax determinism.
- Any other production file, the `max_coverage_instance` fixture, or any
  helper. These are owned by 013a.
- The guard/parity/edge-case tests (plan 013c).

## Git workflow

- Branch: `advisor/013b-ilp-determinism-tests` (matches the `advisor/NNN-<slug>` convention).
- Single commit is fine. Message: `test(optim): pin ILP determinism, time-limit, well-formedness`. See `git log --oneline -10` for the repo's existing style.
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Append the 013b test methods to `class TestILPSolver`

Append these test methods to the existing `class TestILPSolver` in
`tests/test_ilp.py`. Do not add new imports (everything needed is already
imported by 013a). Each test uses the `max_coverage_instance`, `default_cfg`,
`default_weights` fixtures from 013a.

1. `test_deterministic_across_runs` — solve twice at budget 1200; assert
   identical `edges`, `spent`, `extra["surrogate_count"]`, `extra["status"]`.
   Determinism comes from `num_search_workers = 1` + the weighted lex tie-break
   (013a Step 2.5). Use `sol1.edges == sol2.edges` (equality on `Candidate` is
   `(u, v, length, road_priority)` per the dataclass identity) and
   `pytest.approx` on `spent`.
   ```python
   def test_deterministic_across_runs(self, max_coverage_instance, default_cfg, default_weights):
       graph, candidates = max_coverage_instance
       sol1 = ILPSolver(default_cfg, default_weights).solve(graph, candidates, 1200.0)
       sol2 = ILPSolver(default_cfg, default_weights).solve(graph, candidates, 1200.0)
       assert sol1.edges == sol2.edges
       assert sol1.spent == pytest.approx(sol2.spent)
       assert sol1.extra["surrogate_count"] == sol2.extra["surrogate_count"]
       assert sol1.extra["status"] == sol2.extra["status"]
   ```
2. `test_respects_time_limit_zero` — `ILPSolver(default_cfg, default_weights,
   time_limit_s=0.0).solve(graph, candidates, 1800.0)`: assert it returns a
   `Solution` (no crash) and `runtime_s >= 0.0`. Do NOT assert optimality at
   time limit 0 — assert only `sol.solver == "ilp"` and that `sol.edges` is a
   valid list (possibly empty; CP-SAT may return nothing in zero time). The
   point is the solver must not hang or raise when given no time.
3. `test_solution_record_is_well_formed` — solve at budget 1800; assert
   `isinstance(sol, Solution)`, `sol.solver == "ilp"`, `sol.runtime_s >= 0.0`,
   and that `extra` has all documented keys with correct types/values:
   - `status` is a `str` and ∈ {`"OPTIMAL"`, `"FEASIBLE"`, `"INFEASIBLE"`,
     `"UNKNOWN"`} (expect `"OPTIMAL"` here but assert membership for
     robustness);
   - `objective_kind == "coverage_only"`;
   - `surrogate` is a `float` and `0.0 <= surrogate <= 1.0`;
   - `surrogate_count` is an `int` and `== 9` on this fixture at budget 1800;
   - `n_candidates == 8` (len(`max_coverage_instance` candidates));
   - `n_selected == len(sol.edges)`;
   - `solver_engine == "ortools"`;
   - `coverage_mode == "radius"`;
   - `opt_gap` is a `float` (and `== pytest.approx(0.0)` here since status is
     OPTIMAL).
   Import `Solution` at the top of the test module ONLY if not already
   imported by 013a — check first; 013a's import block does not currently
   import `Solution`, so add `from bike_rl.optim.greedy import Solution` to
   this plan's import additions (see Step 2).
4. `test_does_not_mutate_input_candidates` — snapshot the candidate list
   (copy the list and freeze each `Candidate`'s identity tuple) before solve;
   after solve assert the list is unchanged (same length, same order, same
   identities). `Candidate` is frozen, but assert the *list* identity/order
   too. Pattern:
   ```python
   def test_does_not_mutate_input_candidates(self, max_coverage_instance, default_cfg, default_weights):
       graph, candidates = max_coverage_instance
       snap = [(c.u, c.v, c.length, c.road_priority) for c in candidates]
       ILPSolver(default_cfg, default_weights).solve(graph, candidates, 1800.0)
       assert [(c.u, c.v, c.length, c.road_priority) for c in candidates] == snap
       assert len(candidates) == len(snap)
   ```
5. `test_disjoint_optimum_at_budget_1200` — at budget 1200 assert that the
   ILP's chosen set is a max-coverage-with-disjoint-endpoints global solution,
   not a greedy-style chain: `sol.extra["surrogate_count"] == 7` AND the chosen
   edges are pairwise disjoint (no shared endpoint). Reason: the optimal
   coverage-7 solutions at budget 1200 are exactly two disjoint 600-cost
   edges each covering 2 new far nodes (e.g. `{(10,11),(12,13)}`). Assert:
   ```python
   def test_disjoint_optimum_at_budget_1200(self, max_coverage_instance, default_cfg, default_weights):
       graph, candidates = max_coverage_instance
       sol = ILPSolver(default_cfg, default_weights).solve(graph, candidates, 1200.0)
       assert sol.extra["surrogate_count"] == 7
       endpoints = []
       for e in sol.edges:
           endpoints += [e.u, e.v]
       assert len(endpoints) == len(set(endpoints)), "chosen edges must be pairwise endpoint-disjoint"
   ```

### Step 2: Ensure `Solution` is importable for `test_solution_record_is_well_formed`

If `tests/test_ilp.py` does not yet import `Solution` (013a's import block
did not), add it. **Add** to the existing import block (do not duplicate):
```python
from bike_rl.optim.greedy import Solution
```
Place it alongside the other `bike_rl.optim...` imports 013a already made,
keeping ruff `I` (isort) ordering. Run `ruff check .` and `ruff format .` to
confirm ordering; `ruff check --fix .` will sort if needed.

**Verify**:
- `ruff check .` → exit 0
- `ruff format --check .` → exit 0
- `mypy --strict bike_rl` → exit 0 (test files are not under `bike_rl/`, so
  mypy won't type-check the test additions — but `bike_rl` must still be clean.)
- `pytest -q tests/test_ilp.py` → all pass (013a's 3 + this plan's 5 = 8 tests; with 013a's parametrisation, ~11 cases).
- `pytest -q` → full suite green, coverage still ≥80% (no coverage drop — `ilp.py` unchanged).

## Test plan

- Append 5 methods to `class TestILPSolver` in `tests/test_ilp.py` (no new file, no new class).
- Optionally add the single `Solution` import (Step 2). Adds zero production change.

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `pytest -q tests/test_ilp.py` exits 0; the 5 new tests pass alongside 013a's tests.
- [ ] `pytest -q` exits 0; whole-package coverage ≥80% (unchanged from 013a — `ilp.py` not modified).
- [ ] `ruff check .` exits 0.
- [ ] `ruff format --check .` exits 0.
- [ ] `mypy --strict bike_rl` exits 0.
- [ ] No file outside `tests/test_ilp.py` is modified (`git status --short` — unless a 013a solver regression forced a documented `ilp.py` fix under the STOP conditions, in which case report it explicitly).
- [ ] `plans/README.md` status row for plan 013b updated (TODO → DONE).

## STOP conditions

Stop and report back (do not improvise) if:

- `tests/test_ilp.py` or `bike_rl/optim/ilp.py` no longer matches "Current state" (drift since `bc58ad2`/013a landing).
- `test_deterministic_across_runs` fails — this is a REAL solver regression, not a test bug. Report. Fixing it by relaxing the assertion or removing `num_search_workers = 1` is FORBIDDEN. The only sanctioned fix in `ilp.py` is raising the lex-objective primary multiplier (`* 100_000`) — if that does not help, STOP.
- `test_respects_time_limit_zero` raises (CP-SAT cannot accept `max_time_in_seconds = 0.0`) — report; do NOT silently clamp the time limit. If OR-Tools rejects exactly-0, the 013a solver should already defend against it; if it does not, that is a 013a bug to file, not a 013b workaround.
- `test_solution_record_is_well_formed` finds a missing/wrong-typed `extra` key — that means 013a's `Solution` construction drifted; report (do not paper over with type coercions in the test).
- `test_disjoint_optimum_at_budget_1200` finds the ILP picking a contiguous chain (shared endpoints) AND `surrogate_count == 7` — meaning the lex tie-break is selecting a different but equally-optimal set. That is NOT a regression (multiple disjoint optima exist), but the disjoint-endpoints assertion may be too strict; report so the assertion can be relaxed to "count == 7" only. Do NOT weaken on your own.
- You find that any of these tests requires editing `ilp.py` to pass — STOP and report which one and why; 013b is meant to be test-only.

## Maintenance notes

- These tests pin the non-parity contract `evaluate.py` (§11 item 5) will
  silently rely on: a reproducible, time-limit-safe, well-formed `Solution`.
  If a future change makes the ILP nondeterministic or drops an `extra` key,
  these fail loudly.
- **Reviewer focus**: (a) `test_deterministic_across_runs` is the load-bearing
  test — if it is ever marked `xfail` or skipped, that is a determinism bug to
  fix, not a flaky test to suppress; (b) `test_respects_time_limit_zero` must
  NOT assert optimality (zero time is too little to guarantee OPTIMAL); (c)
  the `extra` keys asserted by `test_solution_record_is_well_formed` are the
  contract for `evaluate.py`'s optimality-gap column.
- After 013b, plan 013c adds the guard/parity/edge-case tests to this same
  file. The `cov_only_weights` fixture 013a defined is used by 013c, not here.