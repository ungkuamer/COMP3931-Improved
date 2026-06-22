# Direct Optimiser Baseline — Specification

This document specifies the **non-RL baseline solvers** for the bike-path network expansion problem. It is designed to live **alongside** `RECREATE_SPEC.md` (the RL spec) in the same improved repo, sharing the objective, candidate, and metrics modules so that **RL and direct optimisation solve exactly the same problem and are directly comparable**.

The baseline has three solvers of increasing strength:
1. **Greedy by marginal gain per cost** — fast, deterministic, theoretically grounded (submodularity).
2. **Greedy + local search** — swaps/2-opt polish; closes most of the gap to optimal.
3. **ILP (integer linear programming)** — exact oracle for small cities; the ground-truth ceiling.

---

## 1. Why this baseline exists

A reviewer's first question for any RL solution to a deterministic selection problem is: *"Does it beat a simple optimiser?"* Without an answer, RL results are uninterpretable. The direct optimiser provides:

- **A performance floor**: greedy is the minimum bar RL must clear to be interesting.
- **A performance ceiling**: ILP gives the true optimum on small instances, so you can report *optimality gap*.
- **A runtime/quality trade-off curve**: greedy (ms) vs local-search (s) vs ILP (min–∞) vs PPO (min–h).
- **A fairness guarantee**: both solvers consume the same `candidates.py` + `metrics.py` + `objective.py`, so any difference in outcome is due to the *search strategy*, not the problem formulation.

This is what turns "I applied RL" into "I evaluated RL against the alternatives and characterised when it wins."

---

## 2. Shared modules (imported from `bike_rl/`)

The optimiser reuses the RL package's modules so the two approaches never drift apart. New/extended modules:

```
bike_rl/
├── objective.py        # NEW: the canonical scalar objective (single source of truth)
├── candidates.py       # shared with RL env (per §5.8 of RECREATE_SPEC)
├── metrics.py          # shared with RL env
├── graph_utils.py      # shared
├── optim/
│   ├── __init__.py
│   ├── greedy.py       # GreedySolver
│   ├── local_search.py # LocalSearchSolver (wraps greedy + swaps/2-opt)
│   ├── ilp.py          # ILPSolver (OR-Tools/PuLP)
│   ├── budget.py       # budget-tracking utilities shared by all solvers
│   └── evaluate.py     # run a solver on an instance, collect metrics
└── ...                 # env.py, training.py, etc. from RL spec
tests/
├── test_objective.py
├── test_greedy.py
├── test_local_search.py
├── test_ilp.py
└── test_optim_evaluate.py
```

`objective.py` is the **single source of truth** for "what we're optimising." The RL reward should be *derived* from it (see §3), and the optimisers call it directly.

---

## 3. The objective (`objective.py`)

The original project conflated the *objective* with the *reward* and hand-tuned a linear soup of weights. For research-grade work, separate them: define the objective as one (or a small family of) well-defined scalar function(s), and let both the optimiser and the RL reward derive from it.

### 3.1 Canonical objective

```python
@dataclass(frozen=True)
class ObjectiveWeights:
    connectivity: float = 0.4
    coverage: float = 0.4      # bike-lane-reachable population
    fragmentation: float = 0.2 # penalty (so we subtract)

def objective(graph, added_edges, weights: ObjectiveWeights, cfg: Config) -> float:
    """
    Higher is better. Pure function of (graph, chosen edge set).
    Components come from metrics.py (shared with the RL env):
      - connectivity(graph)      in [0,1]
      - coverage(graph)          in [0,1]  (the FIXED population-served, §5.7 of RL spec)
      - fragmentation(graph)     in [0,1]  (lower is better)
    """
    g = apply_added_edges(graph, added_edges)
    conn = connectivity(g, cfg)
    cov  = coverage(g, cfg)
    frag = fragmentation(g, cfg)
    return weights.connectivity * conn + weights.coverage * cov - weights.fragmentation * frag
```

Requirements:
- **Pure & deterministic**: same inputs → same output, no RNG, no global state.
- **Cheap to evaluate incrementally**: provide an `objective_delta(graph, added_edges, new_edge)` helper returning the marginal change from adding one edge, so greedy doesn't recompute from scratch. Backed by the incremental data structures in `metrics.py` (union-find for fragmentation, persistent rx graph for connectivity — see §6.2/6.3 of the RL spec).
- **Versioned**: if you define multiple objectives (e.g. coverage-only vs combined), tag them by name so results are reproducible.

### 3.2 Relation to the RL reward
The RL reward for adding edge `e` should be `objective_delta(...) * scale + shaped_terms`. The shaped bonuses (continuity, road priority, isolation penalty) stay in the *reward* as learning aids, but the **reported metric** in evaluation is always the canonical `objective`, never the reward. This is what makes the comparison fair: both approaches report the same number.

---

## 4. Problem formulation (shared)

```
Given:
  - base bike graph G_bike
  - candidate edges C = {e_1, ..., e_n} from the walk graph (candidates.py)
  - cost(e) = length(e) * edge_cost_factor
  - budget B
  - objective f(G_bike ∪ S) for S ⊆ C

Find: S* = argmax_{S ⊆ C} f(G_bike ∪ S)  subject to  Σ_{e∈S} cost(e) ≤ B
```

This is a **budgeted maximum-coverage / knapsack-constrained network-design** problem. Coverage-style objectives are typically **submodular** (adding an edge helps less as the network grows), which is why greedy has a guarantee.

---

## 5. Solver 1 — Greedy by marginal gain per cost (`greedy.py`)

```python
class GreedySolver:
    def __init__(self, cfg: Config, weights: ObjectiveWeights): ...

    def solve(self, graph, candidates, budget) -> Solution:
        S, remaining, spent = [], list(candidates), 0.0
        while remaining:
            # marginal gain per unit cost for every affordable candidate
            scored = []
            for e in remaining:
                c = cost(e)
                if spent + c > budget:
                    continue
                delta = objective_delta(graph, S, e)   # incremental, §3.1
                if delta <= 0:
                    continue
                scored.append((delta / c, e))
            if not scored:
                break
            _, best = max(scored, key=lambda t: t[0])
            S.append(best); spent += cost(best)
            remaining.remove(best)
        return Solution(edges=S, objective=objective(graph, S, ...), spent=spent)
```

Properties:
- **Deterministic** (no seed needed); tie-break by `(road_priority, -cost)` for stability.
- **Time**: `O(k · |C| · delta_eval)` where `k` = number of edges added. With incremental `objective_delta` this is fast (seconds for a city).
- **Guarantee**: under a cardinality constraint and a monotone submodular `f`, greedy achieves `(1 − 1/e)` of optimal. Under a budget constraint the standard bound is `1/2(1 − 1/e)`; in practice greedy-by-ratio is much closer to optimal.
- **Caveat to document**: if `f` is *not* submodular (e.g. some fragmentation definitions), the guarantee doesn't hold — state this explicitly in the research writeup and verify submodularity empirically (§7).

---

## 6. Solver 2 — Greedy + local search (`local_search.py`)

```python
class LocalSearchSolver:
    def __init__(self, cfg, weights, max_iter=1000, time_limit_s=60): ...

    def solve(self, graph, candidates, budget) -> Solution:
        sol = GreedySolver(cfg, weights).solve(graph, candidates, budget)
        improved = True
        while improved and iters < max_iter and time < time_limit_s:
            improved = False
            # 1-opt swap: replace a chosen edge with a non-chosen one
            for e_out in list(sol.edges):
                for e_in in affordable_non_chosen(...):
                    if objective(new_set) > objective(sol) + epsilon:
                        apply swap; improved = True; break
            # 2-opt: drop one edge, add two cheaper ones fitting freed budget
            for e_out in sol.edges:
                for (e1, e2) in affordable_pairs_with_freed_budget(...):
                    if objective(...) > objective(sol) + epsilon:
                        apply; improved = True; break
        return sol
```

Neighbourhoods:
- **1-opt swap**: replace one chosen edge with one non-chosen edge of ≤ freed cost.
- **2-opt exchange**: drop one chosen edge, add two non-chosen ones whose combined cost fits the freed budget.
- Optional **first-improvement** vs **best-improvement** pivot rule (config flag).

Properties:
- **Deterministic** given the greedy seed; converges to a local optimum.
- **Time**: bounded by `max_iter` and `time_limit_s`; each iteration is `O(|S| · |C|)` swaps with incremental objective.
- **Result**: empirically closes most of the gap to the ILP optimum — report the gap.

---

## 7. Solver 3 — ILP oracle (`ilp.py`)

```python
class ILPSolver:
    def __init__(self, cfg, weights, solver="ortools", time_limit_s=300): ...

    def solve(self, graph, candidates, budget) -> Solution:
        # Variables: x_e ∈ {0,1} per candidate edge
        # Constraint: Σ cost_e * x_e ≤ budget
        # Objective: linearised f(G_bike ∪ {e : x_e = 1})
        # ...
```

Implementation notes:
- Use **OR-Tools** (`CP-SAT` or `GLOP`) as the default free solver; `PuLP` with CBC as a fallback; support `gurobipy` if available (optional, licensed).
- **Linearising the objective**: connectivity and fragmentation involve non-linear graph computations (connected components, clustering). Two strategies:
  - **Coverage-only ILP** (exact, scalable): if the objective reduces to "maximise bike-lane-reachable population under budget," this is a classic maximum-coverage ILP that scales well. Use this as the primary oracle.
  - **Full-objective ILP** (exact, small only): encode connectivity via flow/Steiner-tree-style auxiliary variables; only tractable for graphs up to a few hundred candidate edges. Use this on small synthetic instances to validate the heuristics.
- **Time limit**: always set one; report status (`OPTIMAL`, `FEASIBLE`, `INFEASIBLE`, `TIMEOUT`). A timeout is itself a result ("ILP doesn't scale past N candidates; greedy/RL do").

Properties:
- Returns the **true optimum** when it solves to optimality → ground-truth ceiling.
- Use only on **small/medium instances** (parameterise a `max_candidates_for_ilp` threshold in `Config`).

---

## 8. Shared evaluation harness (`optim/evaluate.py`)

One runner for all solvers + the RL policy, so the comparison is apples-to-apples:

```python
@dataclass
class Solution:
    edges: list
    objective: float
    spent: float
    runtime_s: float
    solver: str
    extra: dict             # solver-specific (e.g. ILP status, opt gap)

def evaluate_solver(solver, instance) -> Solution: ...
def evaluate_rl_policy(policy, instance, budget) -> Solution: ...  # deterministic rollout
```

`evaluate_rl_policy` runs the trained PPO policy deterministically on the instance, collects the chosen edges, and scores them with the **same** `objective` — so the RL number and the optimiser number are directly comparable (§3.2).

---

## 9. Tests

| File | Covers |
|---|---|
| `test_objective.py` | `objective` is pure/deterministic; `objective_delta` matches a full recompute; monotonicity on small fixtures; submodularity check (adding an edge helps less on a richer graph) |
| `test_greedy.py` | picks the obviously-correct edge on a hand-built tiny instance; respects budget (never overspends); terminates when no affordable edge; deterministic across runs |
| `test_local_search.py` | never returns a solution worse than its greedy seed; a known local-optimum fixture escapes under 2-opt; respects time/iter limits |
| `test_ilp.py` | matches brute force on tiny instances (≤ 12 candidates); reports optimality status correctly; respects time limit |
| `test_optim_evaluate.py` | `evaluate_solver`/`evaluate_rl_policy` produce comparable `Solution` records; same objective computed for identical edge sets regardless of solver |

Coverage target ≥80% on `bike_rl/optim/` and `bike_rl/objective.py`.

---

## 10. Reporting (feeds the research writeup)

For each city × budget × objective, produce a table:

| Solver | Objective | Budget used | Runtime | Optimality gap (vs ILP) | # edges |
|---|---|---|---|---|---|
| Greedy | … | … | 0.3s | 4.1% | 37 |
| Greedy + LS | … | … | 2.1s | 1.2% | 39 |
| ILP (oracle) | … | … | 48s | 0% | 41 |
| PPO (RL) | … | … | 3m (train) | … | 38 |

Also plot **objective vs budget** curves per solver, and **runtime vs instance size**. These are the figures that make the RL contribution (or lack thereof) legible.

---

## 11. Implementation order (after the shared modules exist)

1. `objective.py` + `test_objective.py` (depends on `metrics.py` from RL spec).
2. `optim/greedy.py` + `test_greedy.py`.
3. `optim/local_search.py` + `test_local_search.py`.
4. `optim/ilp.py` (start with coverage-only ILP) + `test_ilp.py`.
5. `optim/evaluate.py` + `evaluate_rl_policy` integration + `test_optim_evaluate.py`.
6. Run the full comparison on one small city → produce the §10 table → sanity-check that greedy ≥ RL or diagnose why not.
7. Scale up to the real cities; record optimality gaps and runtimes.

---

## 12. Definition of Done (optimiser baseline)

- All three solvers run on the same instance and return a `Solution` scored by the shared `objective`.
- Greedy is deterministic and beats brute-force random selection by a large margin on tiny fixtures.
- ILP matches brute force on ≤12-candidate instances.
- `evaluate_rl_policy` scores a trained policy with the same objective, producing a directly comparable row.
- The §10 reporting table can be generated by a single CLI command (`python -m bike_rl.cli --compare ...` or a `scripts/run_comparison.py`).
- Tests green, ≥80% coverage on `bike_rl/optim/` + `objective.py`.
