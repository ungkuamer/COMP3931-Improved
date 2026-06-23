# Plan 010: Implement the canonical objective (`objective.py`) + tests

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**:
> `git diff --stat c648b2e..HEAD -- bike_rl/objective.py tests/test_objective.py tests/conftest.py bike_rl/metrics.py bike_rl/candidates.py bike_rl/config.py`
> If any in-scope or dependency file changed since this plan was written,
> compare the "Current state" excerpts against the live code before
> proceeding; on a mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: LOW
- **Depends on**: plans/003-metrics-and-tests.md (DONE — `metrics.py` is the
  source of `connectivity`/`coverage`/`fragmentation`), plans/002-graph-utils-and-candidates.md
  (DONE — `Candidate` dataclass used as the edge type)
- **Category**: tech-debt (implements a stubbed module that the whole
  optimiser baseline and the fair RL-vs-optimiser comparison depend on)
- **Planned at**: commit `c648b2e`, 2026-06-23
- **Issue**: (not published)

## Why this matters

`OPTIMIZER_SPEC.md` §3 makes `objective.py` the **single source of truth** for
"what we are optimising". Today `bike_rl/objective.py` is a stub — every
function raises `NotImplementedError` — so none of the direct optimisers
(`optim/greedy.py`, `local_search.py`, `ilp.py`, `evaluate.py`, all stubs) can
be implemented, and there is no shared scalar that lets RL and the optimisers
be compared apples-to-apples (§3.2). Landing this module unblocks the entire
optimiser baseline (OPTIMIZER_SPEC §11 items 2–5) and establishes the
contract `objective`/`objective_delta`/`apply_added_edges` that every later
optimiser plan imports. It is item #1 of OPTIMIZER_SPEC §11.

The implementation is deliberately **correctness-first**: `objective_delta`
is defined as a difference of two full `objective` evaluations, which the
spec's own test table (`OPTIMIZER_SPEC.md` §9) requires to "match a full
recompute". A faster `MetricsState`-backed incremental `objective_delta` is
explicitly deferred to a later performance plan (see Maintenance notes) — it
is not needed for correctness and would risk bugs in the baseline.

## Current state

The repo is a Python 3.10+ package `bike_rl` (see `pyproject.toml`). The RL
pipeline (plans 001–009) is DONE; the optimiser side is entirely stubs.

### `bike_rl/objective.py` (stub — the file you will implement)

Currently 47 lines, all three public functions raise `NotImplementedError`.
`ObjectiveWeights` is already implemented and **pinned by
`tests/test_smoke.py:16`** — do NOT change its defaults:

```python
@dataclass(frozen=True)
class ObjectiveWeights:
    connectivity: float = 0.4
    coverage: float = 0.4
    fragmentation: float = 0.2

def objective(graph, added_edges, weights, cfg) -> float:
    raise NotImplementedError

def objective_delta(graph, added_edges, new_edge, weights, cfg) -> float:
    raise NotImplementedError

def apply_added_edges(graph, added_edges) -> object:
    raise NotImplementedError
```

`tests/test_smoke.py:16` asserts the three defaults are exactly `0.4/0.4/0.2`.
Keep `ObjectiveWeights` byte-for-byte as-is.

### The metric functions this module calls (`bike_rl/metrics.py`)

These are DONE and fully tested. Their exact signatures (the contract you
must call against):

```python
def connectivity(graph: nx.MultiDiGraph, cfg: Config) -> float   # [0,1], transitivity of largest CC
def coverage(graph: nx.MultiDiGraph, cfg: Config) -> float       # [0,1], §5.7 population-served proxy
def fragmentation(graph: nx.MultiDiGraph, cfg: Config) -> float  # [0,1], lower is better
```

All three are **pure and deterministic** functions of `(graph, cfg)`. They
read `bike_lane='yes'` edge tags and node `x`/`y` attributes. The objective
composes them per `OPTIMIZER_SPEC.md` §3.1:

```
f(graph, added_edges) = w_connectivity * connectivity + w_coverage * coverage
                        - w_fragmentation * fragmentation
```

(fragmentation is **subtracted** because lower is better — confirmed by the
`ObjectiveWeights` docstring "fragmentation: ... subtracted" and §3.1.)

### The edge type: `bike_rl/candidates.py` `Candidate`

The optimisers select `Candidate` objects (DONE, tested). Its identity is
`(u, v, length, road_priority)`; the `data` attribute is a copy of the walk
edge's attributes (includes `length`). The env applies a candidate exactly
this way — `bike_rl/env.py:271` `_apply_edge`:

```python
u, v = candidate.u, candidate.v
data = dict(candidate.data)
data["bike_lane"] = "yes"
data["length"] = candidate.length
self._graph.add_edge(u, v, **data)
```

`objective.py` must do the **same** in `apply_added_edges`, but on a **copy**
of the graph (the objective is pure — it must not mutate the input graph,
which the greedy solver reuses across iterations).

### Config (`bike_rl/config.py`)

`Config` is a frozen dataclass; the metrics read `coverage_mode`,
`coverage_radius_m`, `seed`, `sampling_thresholds`, `path_sample_formula`.
No new `Config` fields are needed for this plan.

### Repo conventions to match

- **Typing**: `from __future__ import annotations`; use the
  `TYPE_CHECKING` pattern for the `Config` import and the
  `_NXGraph = nx.MultiDiGraph` alias exactly as `bike_rl/metrics.py:18-23`
  does it. `mypy --strict` is enforced (see Commands).
- **Docstrings**: Google convention (`.ruff` selects `D` with
  `convention = "google"`). Every public function gets an Args/Returns
  docstring. See `bike_rl/metrics.py` `connectivity`/`coverage` as exemplars.
- **Tests**: one class per concern, descriptive method names, one-line
  docstring per test stating the invariant, local fixtures in the test file.
  Model after `tests/test_metrics.py` (e.g. `class TestConnectivity`,
  `class TestCoverage`). Shared fixtures `tiny_bike_graph`,
  `two_component_bike_graph`, etc. live in `tests/conftest.py` and are
  available without import.
- **No new dependencies**. `objective.py` only needs `networkx` (already a
  dep) and `bike_rl.metrics` / `bike_rl.config` / `bike_rl.candidates`.
- **Coverage gate**: `pyproject.toml` sets
  `--cov=bike_rl --cov-fail-under=80`. The stub functions currently
  contribute 0 covered lines; after this plan `objective.py` must be near
  100% covered so the gate stays green (whole-repo coverage was ~89% at
  `c648b2e`).

## Commands you will need

| Purpose    | Command                                              | Expected on success |
|------------|------------------------------------------------------|---------------------|
| Lint       | `ruff check .`                                       | "All checks passed!" |
| Format     | `ruff format --check .`                              | "N files already formatted" |
| Typecheck  | `mypy --strict bike_rl`                              | "Success: no issues found in 18 source files" (count may rise by 0 — `objective.py` is already counted) |
| Tests      | `python -m pytest tests/test_objective.py -q --no-cov` | all pass |
| Full gate  | `python -m pytest -q`                                | all pass, coverage ≥80% |

All commands run from the repo root `/home/ungku/programming/COMP3931-Improved`.
`ruff`, `mypy`, `pytest` are installed via the `dev` extra
(`pip install -e ".[dev]"` — do **not** run installs yourself; the
environment is already set up).

## Scope

**In scope** (the only files you should modify):
- `bike_rl/objective.py` — replace the three `NotImplementedError` bodies
  with real implementations; keep `ObjectiveWeights` unchanged; you may add
  a `Protocol`/type alias and imports as needed.
- `tests/test_objective.py` — create (new file).

**Out of scope** (do NOT touch, even though they look related):
- `bike_rl/metrics.py` — DONE; call its public functions only, never edit.
- `bike_rl/candidates.py` — DONE; import `Candidate` only if useful for the
  `Edge` protocol; do not edit it.
- `bike_rl/config.py` — no new fields. Do not edit.
- `bike_rl/optim/*` — all stubs; a later plan implements them. Do not edit.
- `bike_rl/env.py` / `bike_rl/training.py` / reward derivation —
  `OPTIMIZER_SPEC.md` §3.2 says the RL reward should *derive* from
  `objective_delta`, but rewiring the env reward is a **separate** concern
  and is NOT part of §11 item 1. Do not touch the env.
- `tests/test_smoke.py` — its `ObjectiveWeights` assertion must keep passing;
  do not edit it.

## Git workflow

- Branch: `advisor/010-objective-and-tests`
- Commit per logical unit (implementation, then tests), conventional-commit
  style matching `git log` (e.g. `feat(objective): implement canonical
  objective + apply_added_edges`, `test(objective): add objective test suite`).
- Do NOT push or open a PR.

## Steps

### Step 1: Replace `bike_rl/objective.py` with a real implementation

Rewrite `bike_rl/objective.py` so that:

1. **Imports**: add `from typing import TYPE_CHECKING, Any, Protocol, Sequence`
   and `import networkx as nx`; keep the `Config` import under `TYPE_CHECKING`
   and define the `_NXGraph` alias exactly like `bike_rl/metrics.py:18-23`:
   ```python
   if TYPE_CHECKING:
       from bike_rl.config import Config
       _NXGraph = nx.MultiDiGraph[Any, Any, Any]
   else:
       _NXGraph = nx.MultiDiGraph
   ```
2. **`ObjectiveWeights`**: keep the existing dataclass **unchanged** (its
   defaults are pinned by `tests/test_smoke.py`).
3. **`Edge` protocol** (new): a structural protocol so `objective` accepts
   `Candidate` objects without importing `candidates.py` (avoids coupling and
   keeps `objective` decoupled from the candidate extractor):
   ```python
   class Edge(Protocol):
       """Structural type for an addable bike-lane edge (satisfied by Candidate)."""
       u: int | str
       v: int | str
       length: float
       data: dict[str, Any]
   ```
   Do **not** make it `@runtime_checkable` (not needed; mypy strict is
   happier without it). `bike_rl.candidates.Candidate` has exactly these
   attributes, so it satisfies `Edge` structurally.
4. **`apply_added_edges(graph, added_edges) -> _NXGraph`**: return a **copy**
   of `graph` with each edge added as a bike-lane edge. Mirror
   `bike_rl/env.py:280-284` exactly:
   ```python
   g = graph.copy()
   for e in added_edges:
       if e.u == e.v:
           continue  # match _bike_lane_subgraph / MetricsState self-loop skip
       data = dict(e.data)
       data["bike_lane"] = "yes"
       data["length"] = e.length
       g.add_edge(e.u, e.v, **data)
   return g
   ```
   `nx.MultiDiGraph.add_edge` auto-adds missing endpoints as nodes (without
   `x`/`y`); that is acceptable and matches env behaviour. The input
   `graph` must not be mutated — work on the copy only.
5. **`objective(graph, added_edges, weights, cfg) -> float`**:
   ```python
   g = apply_added_edges(graph, added_edges)
   conn = connectivity(g, cfg)
   cov = coverage(g, cfg)
   frag = fragmentation(g, cfg)
   return (
       weights.connectivity * conn
       + weights.coverage * cov
       - weights.fragmentation * frag
   )
   ```
   Import `connectivity`, `coverage`, `fragmentation` from `bike_rl.metrics`
   at module top (not under `TYPE_CHECKING` — they are used at runtime).
6. **`objective_delta(graph, added_edges, new_edge, weights, cfg) -> float`**:
   the marginal change from adding `new_edge`, defined as a difference of two
   full evaluations (correctness-first, per the §9 test requirement
   "`objective_delta` matches a full recompute"):
   ```python
   before = objective(graph, added_edges, weights, cfg)
   after = objective(graph, [*added_edges, new_edge], weights, cfg)
   return after - before
   ```
7. Update each function's docstring (Google style) to describe Args/Returns
   and note that `objective` is pure/deterministic and `objective_delta` is
   currently a full-recompute difference (perf-deferred).

Change the public signatures' `added_edges` type from
`list[tuple[int | str, int | str]]` to `Sequence[Edge]`, and `new_edge` from
a tuple to `Edge`. (The stub signatures were placeholders; nothing imports
these signatures today — `tests/test_smoke.py` only touches
`ObjectiveWeights`.)

**Verify**:
- `ruff check bike_rl/objective.py` → "All checks passed!"
- `ruff format --check bike_rl/objective.py` → already formatted (if not, run
  `ruff format bike_rl/objective.py` and re-check).
- `mypy --strict bike_rl/objective.py` → "Success: no issues found in 1
  source file".

### Step 2: Create `tests/test_objective.py`

Create the test file modelled on `tests/test_metrics.py` (classes, docstring
per test, local fixtures, known-answer asserts). Use the shared
`two_component_bike_graph` and `tiny_bike_graph` fixtures from
`tests/conftest.py` plus local fixtures as needed. Build added edges with
`bike_rl.candidates.Candidate` (so the `Edge` protocol is exercised against
the real type) — e.g. a helper:

```python
def _edge(u, v, length, **extra) -> Candidate:
    return Candidate(u=u, v=v, length=float(length),
                     road_priority=1, connects_to_bike_path=False,
                     data={"length": float(length), **extra})
```

Required test cases (each as its own method, with a docstring stating the
invariant) — these directly satisfy `OPTIMIZER_SPEC.md` §9 `test_objective.py`
row:

1. **`test_objective_is_pure_and_deterministic`** — calling `objective`
   twice on the same `(graph, added_edges, weights, cfg)` returns identical
   results; and the input graph is unchanged afterwards
   (`assert not any(d.get("bike_lane")=="yes") for ...` on the original, or
   compare `graph.copy()` snapshot before/after).
2. **`test_objective_delta_matches_full_recompute`** — for a fixture graph
   and an edge `e`, `objective_delta(g, S, e, w, cfg) ==
   objective(g, S+[e], w, cfg) - objective(g, S, w, cfg)` within `1e-9`.
3. **`test_objective_monotone_for_useful_edge`** — on
   `two_component_bike_graph`, adding a bike-lane edge `(2,5)` (length 200)
   increases `objective` (delta > 0). This is the optimiser's core premise.
4. **`test_objective_submodular_decreasing_returns`** — adding a given edge
   to an empty selection gives a delta **≥** the delta from adding the same
   edge to an already-richer selection (add a couple of other bike-lane
   edges first). Assert `delta_sparse >= delta_rich - 1e-9`. This is the
   §9 "submodularity check (adding an edge helps less on a richer graph)".
5. **`test_objective_decomposes_into_weighted_metrics`** — with weights
   `{connectivity:1, coverage:0, fragmentation:0}`, `objective(g,S,w,cfg)
   == connectivity(apply_added_edges(g,S), cfg)` (within `1e-9`); and
   similarly for a coverage-only weight and a fragmentation-only weight
   (fragmentation-only: `objective == -fragmentation(...)`). Three asserts,
   one test method.
6. **`test_objective_empty_added_edges_equals_base`** —
   `objective(g, [], weights, cfg) == w.c*connectivity(g) + w.cov*coverage(g)
   - w.frag*fragmentation(g)` within `1e-9` (no edges added → metrics on the
   original graph).
7. **`test_apply_added_edges_does_not_mutate_input`** — snapshot
   `list(two_component_bike_graph.edges(data=True))` before, call
   `apply_added_edges`, assert the snapshot equals the post-call edges of
   the original graph.
8. **`test_apply_added_edges_sets_bike_lane_tag`** — after
   `apply_added_edges(g, [_edge(2,5,200)])`, the returned graph has an edge
   `(2,5)` with `bike_lane == "yes"` and `length == 200.0`.
9. **`test_apply_added_edges_skips_self_loop`** — adding `_edge(5,5,10)`
   produces no new edge in the returned graph (self-loops are skipped, matching
   `_bike_lane_subgraph` and `MetricsState.add_edge`).
10. **`test_objective_delta_zero_for_duplicate_edge`** — adding an edge that
    is already in `added_edges` yields `abs(delta) < 1e-9` (the second add
    changes no bike-lane structure — same `(u,v)` already tagged).

Use `Config()` defaults (radius mode, 300m). Where a known numeric answer is
asserted, compute the three metrics on the fixture and combine by hand in the
test (don't hardcode magic numbers without derivation — see how
`tests/test_metrics.py` `test_fragmentation_two_components` shows its work).

**Verify**:
- `ruff check tests/test_objective.py` → "All checks passed!"
- `ruff format --check tests/test_objective.py` → already formatted.
- `mypy --strict tests/test_objective.py` → "Success: no issues found in 1
  source file" (mypy is configured for `bike_rl`; running on the test file
  directly should still pass because `tests/` uses the same strict settings —
  if mypy reports config scope issues, fall back to `mypy --strict bike_rl
  tests/test_objective.py` and confirm no errors in the test file).
- `python -m pytest tests/test_objective.py -q --no-cov` → all 10 tests pass.

### Step 3: Run the full QA gate

Run the whole suite to confirm nothing regressed and the coverage gate holds
(this is the gate CI enforces via `pyproject.toml` `addopts`):

**Verify**:
- `ruff check .` → "All checks passed!"
- `ruff format --check .` → "N files already formatted"
- `mypy --strict bike_rl` → "Success: no issues found in 18 source files"
- `python -m pytest -q` → all pass, coverage ≥80% (the summary prints
  `bike_rl/objective.py` near 100%; total stays ≥80%).

If the coverage gate fails, the cause is **under-covered code in
`objective.py`** — add a test hitting the uncovered branch (most likely the
self-loop `continue` in `apply_added_edges`, already covered by test 9) and
re-run. Do NOT lower the gate or add `pragma: no cover`.

### Step 4: Update `plans/README.md`

Add a row to the execution-order table and a dependency note:

- Row: `| 010 | Implement canonical objective (objective.py) + test_objective.py | P1 | M | 003 | DONE |`
- In "Dependency notes", append: `010 (objective.py) is the first optimiser
  plan (OPTIMIZER_SPEC §11.1). It depends on 003 (metrics) and 002
  (Candidate). All later optimiser plans (greedy/local_search/ilp/evaluate)
  import objective/objective_delta/apply_added_edges from it. The RL reward
  is NOT rewired to objective_delta here — that is a separate, later concern
  (§3.2).`
- Mark the row status `DONE` (you ran the gate green) — or `BLOCKED` with a
  one-line reason if you stopped on a STOP condition.

## Test plan

- New file `tests/test_objective.py` with the 10 cases listed in Step 2,
  covering: purity/determinism, delta-matches-recompute, monotonicity,
  submodularity (decreasing returns), weighted decomposition, empty-selection
  base case, input non-mutation, bike-lane tagging, self-loop skip,
  duplicate-edge zero delta. Together these satisfy every bullet in
  `OPTIMIZER_SPEC.md` §9's `test_objective.py` row.
- Structural pattern: `tests/test_metrics.py` (class-per-concern, docstring
  per test, local fixtures, known-answer asserts that show their work).
- Verification: `python -m pytest tests/test_objective.py -q --no-cov` →
  all pass; then `python -m pytest -q` → all pass with coverage ≥80%.

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `ruff check .` exits 0 ("All checks passed!")
- [ ] `ruff format --check .` exits 0
- [ ] `mypy --strict bike_rl` exits 0 ("Success: no issues found …")
- [ ] `python -m pytest -q` exits 0; `tests/test_objective.py` exists with
      ≥10 tests, all passing
- [ ] `bike_rl/objective.py` has no remaining `raise NotImplementedError`
      (`grep -n "NotImplementedError" bike_rl/objective.py` → no matches)
- [ ] `ObjectiveWeights` defaults unchanged
      (`python -c "from bike_rl.objective import ObjectiveWeights as w; \
       assert (w().connectivity,w().coverage,w().fragmentation)==(0.4,0.4,0.2)"`)
- [ ] No files outside the in-scope list are modified
      (`git status --short` shows only `bike_rl/objective.py`,
      `tests/test_objective.py`, and `plans/README.md`)
- [ ] `plans/README.md` has a `010` row marked `DONE` (or `BLOCKED` + reason)

## STOP conditions

Stop and report back (do not improvise) if:

- `bike_rl/objective.py`'s current contents do not match the stub excerpt in
  "Current state" (e.g. the functions are no longer `NotImplementedError`, or
  `ObjectiveWeights` defaults differ) — the codebase has drifted; re-plan.
- `bike_rl/metrics.py`'s public signatures
  (`connectivity`/`coverage`/`fragmentation(graph, cfg) -> float`) differ
  from those quoted above — the metrics contract has changed; do not guess
  how to call them.
- `tests/test_smoke.py:16` (`test_objective_weights_defaults`) fails after
  your changes — you accidentally altered `ObjectiveWeights`; revert that
  dataclass and re-run.
- Implementing `objective` correctly appears to require editing
  `bike_rl/metrics.py` or `bike_rl/candidates.py` — it must not; report the
  mismatch instead.
- The coverage gate (`--cov-fail-under=80`) fails for a reason other than
  under-covered `objective.py` lines — report rather than editing
  `pyproject.toml` or adding `pragma: no cover`.

## Maintenance notes

For whoever owns this after it lands:

- **Future optimiser plans** (`optim/greedy.py` etc., OPTIMIZER_SPEC §11
  items 2–5) will import `objective`, `objective_delta`, `apply_added_edges`,
  `ObjectiveWeights`, and the `Edge` protocol from this module. The
  `objective_delta` offered here is a **full-recompute difference** — correct
  but O(2 · metric_eval) per call. Greedy calls it `O(k·|C|)` times, so on a
  real city this will be the optimiser's bottleneck. A later **performance**
  plan should swap `objective_delta`'s body for a `bike_rl.metrics.MetricsState`-backed
  incremental computation (build a `MetricsState` from `graph + added_edges`,
  read the four scalars, `add_edge(new_edge)`, read again, return the
  weighted difference) without changing its signature or tests. The
  `test_objective_delta_matches_full_recompute` test is the regression guard
  for that swap.
- **RL reward derivation (§3.2)** is intentionally NOT done here. When a
  later plan rewires the env reward to `objective_delta * scale + shaped`,
  the reported metric must remain the canonical `objective` (not the reward)
  — `evaluate_rl_policy` (§8, future plan) is what scores a policy with
  `objective` for the fair comparison.
- **Reviewer focus**: confirm `apply_added_edges` works on a copy and never
  mutates the input (test 7), that fragmentation is subtracted not added
  (test 5), and that `objective_delta` is exactly `after - before` (test 2).
- **Submodularity caveat** (OPTIMIZER_SPEC §5): the greedy `(1−1/e)`-style
  guarantee only holds if `f` is monotone submodular. Test 4 checks
  decreasing returns empirically on a tiny fixture; it is **not** a proof.
  The research writeup must state this and verify submodularity empirically
  on real instances (§7). Do not claim the guarantee holds without that.
