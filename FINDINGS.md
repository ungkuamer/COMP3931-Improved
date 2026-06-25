# Findings: Direct Optimiser vs RL Baseline Comparison

This document records the headline experimental result of the direct-optimiser
baseline work specified in [`OPTIMIZER_SPEC.md`](OPTIMIZER_SPEC.md) §11 item 6:
**run the full optimiser/RL comparison on one small city, produce the §10
reporting table, and sanity-check that greedy ≥ RL — or diagnose why not.**

The headline answer: **on the default weighted objective, RL beats greedy.**
That is *not* a bug in greedy; it is a property of the objective function we
chose to report. The pages below explain, in order, what we measured, how each
solver works, what the numbers say, and why greedy stalls while RL and the ILP
do not. The short version is in the [TL;DR](#tldr); the full reasoning is in
[Diagnosis](#diagnosis-why-greedy-stalls-and-rl-wins).

---

## TL;DR

- **Instance:** Otley, UK (OSMnx). 132 candidate edges, budget 200,000,
  shared by every solver and the RL env (§8 fairness).
- **Four solvers compared:** greedy, greedy + local search, ILP (OR-Tools
  CP-SAT, coverage-only oracle), MaskablePPO (4096 timesteps, 2 envs).
- **On the default weighted objective** (`0.4·connectivity + 0.4·coverage −
  0.2·fragmentation`), **RL (0.1004) beats greedy (0.0584)**. Greedy stalls at
  a single edge; RL builds 112 edges; the ILP builds 35 edges and scores
  highest (0.1175).
- **The stall is the objective's fault, not greedy's.** On the
  **coverage-only** objective (monotone submodular), greedy picks 38 edges and
  reaches **99.72% of the ILP optimum (0.28% gap)** — the textbook greedy
  behaviour the spec promises. So the greedy/local-search machinery is
  correct; the weighted objective's fragmentation penalty breaks
  monotonicity, which is the documented §5 caveat.
- **Fix direction:** keep fragmentation as an RL *reward-shaping* term only
  (per §3.2's separation of objective and reward), and report a monotone
  objective. That should restore greedy ≥ RL on the reported metric.

---

## Background: what is being compared and why

`OPTIMIZER_SPEC.md` exists because a reviewer's first question for any RL
solution to a deterministic selection problem is *"does it beat a simple
optimiser?"* The spec therefore defines three direct solvers of increasing
strength and requires that **all solvers — including the RL policy — are scored
by the same canonical `objective`** (§3.2/§8), so the comparison is
apples-to-apples. The RL *reward* used during training is allowed to contain
extra shaping terms (continuity bonuses, isolation penalties, etc.), but the
**reported number** is always the canonical objective. That is the fairness
guarantee: a difference in the reported numbers is a difference in *search
strategy*, not in problem formulation.

### The canonical objective

From [`bike_rl/objective.py`](bike_rl/objective.py):

```
f(graph, added_edges) = 0.4·connectivity + 0.4·coverage − 0.2·fragmentation
```

Higher is better. The three components (computed in
[`bike_rl/metrics.py`](bike_rl/metrics.py)) are each in `[0, 1]`:

- **connectivity** — transitivity of the largest connected component of the
  bike-lane subgraph. Rises as the bike-lane network becomes a denser mesh.
- **coverage** — `0.7·coverage_ratio + 0.3·largest_component_ratio`, where
  `coverage_ratio` is the fraction of all graph nodes within
  `coverage_radius_m` (300 m, Euclidean) of any bike-lane edge endpoint, and
  `largest_component_ratio` is the size of the largest bike-lane component
  divided by the total graph node count. This is the population-served proxy
  (RECREATE_SPEC §5.7).
- **fragmentation** — `0.3·min(1,(n_comp−1)/10) + 0.4·(1 − largest/total) +
  0.3·isolated/total`. **Lower is better**, hence it is *subtracted* in the
  objective. It penalises having many small components and isolated nodes.

The critical fact for everything below: **because fragmentation is
subtracted, the weighted objective is not monotone.** Adding a brand-new
isolated bike-lane edge can *decrease* the objective even though it increases
coverage, because the fragmentation penalty (`0.3·isolated/total` and the
`(1 − largest/total)` term) can outweigh the coverage gain.

### The solvers

1. **Greedy** (`bike_rl/optim/greedy.py`) — each round, among affordable
   candidates with positive marginal objective gain, pick the one maximising
   `objective_delta / cost`; tie-break by `(road_priority, −cost)` then by
   earliest input index. Repeat until no affordable candidate improves the
   objective. Deterministic. On a *monotone submodular* objective under a
   budget constraint, greedy-by-ratio achieves `½(1−1/e)` of optimal
   theoretically and is much closer in practice.
2. **Greedy + local search** (`bike_rl/optim/local_search.py`) — seeds from
   greedy, then applies first-improvement 1-opt (replace one chosen edge with
   one non-chosen edge fitting the freed budget) and 2-opt (drop one chosen
   edge, add two non-chosen edges fitting the freed budget) until a local
   optimum or a time/iteration limit. Never returns a solution worse than its
   greedy seed.
3. **ILP** (`bike_rl/optim/ilp.py`) — OR-Tools CP-SAT **coverage-only oracle**:
   budgeted maximum-coverage that maximises the reachable-node count (the
   radius branch of `metrics.coverage`) subject to `Σ cost ≤ budget`. Exact
   and fast. The optimality guarantee is over the **coverage surrogate**, not
   the full weighted objective (the full-objective ILP with connectivity via
   flow/Steiner auxiliaries is deferred — §7). So the ILP is a *true optimality
   ceiling only when weights are coverage-only*; with default weights the ILP
   row is still the highest-scoring solution but its `OPTIMAL` status is over
   the surrogate.
4. **RL** — MaskablePPO from `sb3-contrib`, trained on `BikePathEnv` with the
   shaped reward (state gain + road priority + continuity bonus + budget
   efficiency + isolation penalty; §3.5/§5.5). Scored here by a deterministic
   rollout collected through `evaluate_rl_policy`, which scores the chosen
   edges with the **shared canonical objective** — never the reward.

---

## Experimental setup

- **City:** Otley, UK (`--city "Otley, UK"`, OSMnx, cached).
  - Bike graph: |V|=1422, |E|=3212.
  - Walk graph: |V|=2314, |E|=6254.
  - **132 candidates** extracted once via `extract_candidates(bike, walk, cfg)`
    and shared with every solver *and* the RL env (the env re-extracts the same
    way — §8 fairness).
- **Budget:** 200,000 (cost = `length · edge_cost_factor`, factor = 10).
- **RL training:** 4096 timesteps, 2 parallel envs, seed 0. This is a *very
  light* training run by RL standards — enough to learn a sensible policy, not
  enough to converge.
- **Local search:** 60 s time limit.
- **ILP:** 60 s time limit (solved in 0.32 s; OPTIMAL).
- **Two objective settings:**
  - **Run A — weighted:** default `ObjectiveWeights(0.4, 0.4, 0.2)`.
  - **Run B — coverage-only:** `ObjectiveWeights(0.0, 1.0, 0.0)`, so the ILP is
    a true optimality ceiling.
- **Reproduce:**
  ```
  python scripts/run_comparison.py --city "Otley, UK" --budget 200000 \
      --with-rl --timesteps 4096 --n-envs 2 --ls-time-limit 60
  python scripts/run_comparison.py --city "Otley, UK" --budget 200000 \
      --coverage-only --ls-time-limit 60
  ```
  (The script writes `comparison_table.md` into a timestamped
  `bike_path_figures/compare_*` directory; `FINDINGS.md` is the human-written
  analysis of those runs.)

---

## Results

### Run A — weighted objective (`0.4 / 0.4 / 0.2`)

| Solver | Objective | Budget used | Runtime | # edges | Optimality gap (vs ILP) | ILP status |
|---|---|---|---|---|---|---|
| greedy | 0.0584 | 1138.7 | 26.98s | 1 | 50.28% | — |
| local_search | 0.0584 | 1138.7 | 1m00s | 1 | 50.28% | — |
| ilp | 0.1175 | 70314.0 | 0.32s | 35 | 0.00% | OPTIMAL |
| rl | 0.1004 | 199050.5 | 9.01s | 112 | 14.52% | — |

> Note: with default (non-coverage-only) weights, the ILP's `OPTIMAL` status
> is over its **coverage surrogate**, not the full weighted objective. The
> "gap vs ILP" column is therefore indicative in this run, not a true
> optimality gap. Run B is the run with a valid ceiling.

**Headline:** RL (0.1004) > greedy (0.0584). Greedy and local search both stop
at a single edge and spend only 1138.7 of the 200,000 budget. The ILP spends
70,314 on 35 edges and scores highest. RL spends nearly the whole budget on
112 edges and scores between greedy and the ILP.

### Run B — coverage-only objective (`0.0 / 1.0 / 0.0`)

| Solver | Objective | Budget used | Runtime | # edges | Optimality gap (vs ILP) | ILP status |
|---|---|---|---|---|---|---|
| greedy | 0.5756 | 72423.4 | 9m41s | 38 | 0.28% | — |
| local_search | 0.5756 | 72423.4 | 8m57s | 38 | 0.28% | — |
| ilp | 0.5772 | 70314.0 | 0.32s | 35 | 0.00% | OPTIMAL |

Here the ILP **is** a true optimality ceiling (coverage-only weights match the
ILP's surrogate), so the gaps are real optimality gaps.

**Headline:** greedy reaches **99.72% of the ILP optimum** (0.28% gap) on the
monotone submodular coverage objective — exactly the `(1−1/e)`-beating
behaviour OPTIMIZER_SPEC §5 promises. No RL row in this run (`--with-rl` not
set); it is here to validate the greedy/LS machinery against a known ceiling.

---

## Diagnosis: why greedy stalls and RL wins

### 1. Greedy stalls at one edge on the weighted objective

Greedy-by-marginal-gain-per-cost only adds an edge while
`objective_delta > 0`. On the weighted objective, after the first edge is
added, **every affordable second edge decreases the objective**. Why: the
first edge connects to the existing bike network (small fragmentation hit,
positive coverage gain). A second edge that does *not* touch the growing
component creates a new isolated component, which pushes up the
`isolated/total` and `(1 − largest/total)` fragmentation terms. Because
fragmentation is subtracted with weight 0.2, that penalty outweighs the
coverage/connectivity gain of the second edge, so `objective_delta < 0` and
greedy stops.

In other words, **the weighted objective is non-monotone** along the greedy
construction path on this instance: it goes up at step 1, then down at step 2,
even though it is much higher again at step 35 (the ILP's solution) and step
112 (RL's solution). Greedy is a hill-climber; it cannot descend in order to
later climb a taller hill. It is trapped in a local optimum of size 1.

This is precisely the caveat the spec calls out (§5): *"if `f` is not
submodular (e.g. some fragmentation definitions), the [greedy] guarantee
doesn't hold — state this explicitly in the research writeup and verify
submodularity empirically (§7)."* This run is that empirical verification, and
the weighted objective **fails** it.

### 2. Local search cannot escape either

`LocalSearchSolver` seeds from greedy and tries 1-opt (swap one chosen edge
for one non-chosen edge) and 2-opt (drop one edge, add two). With a single
chosen edge, 1-opt can only replace that one edge with another single edge,
and 2-opt can only replace it with a pair — all of which still produce a
small/isolated network that scores worse than the 1-edge seed. The 2-opt move
that would *add* an edge without dropping one (growing from 1 to 2 edges) is
not in the neighbourhood, and even if it were, it would be a downhill move
that first-improvement local search rejects. So local search also stops at one
edge. This confirms the stall is structural in the objective landscape, not a
fluke of the greedy path.

### 3. The greedy machinery is correct — Run B proves it

Run B removes the fragmentation penalty (coverage-only weights). Coverage is
monotone and submodular (covering more nodes can only help, and helps less as
the covered set grows). On that objective greedy picks 38 edges and reaches
**99.72% of the ILP optimum**. This is the textbook result: greedy-by-ratio on
a monotone submodular function under a budget constraint gets within
`(1−1/e)` of optimal theoretically and much closer in practice. So the
greedy/local-search code is not broken; the **weighted objective** is what
breaks the greedy guarantee.

### 4. Why RL wins the weighted run

RL is *not* optimising the canonical objective directly. During training it
optimises a **shaped reward** that includes a continuity bonus (for extending
an existing bike path), a road-priority bonus, a budget-efficiency term, and
an **isolation penalty** (for adding an edge that touches no existing bike
node). Crucially, that isolation penalty is *mild* relative to greedy's hard
`objective_delta > 0` cutoff: RL is encouraged to keep building a connected
network even when the immediate objective gain is negative, because the
continuity bonus and the long-horizon return push it past the dip.

The result is a 112-edge network. When that network is scored by the **shared
canonical objective** (§3.2/§8 — the whole point of the harness), it lands at
0.1004: above the 1-edge greedy local optimum (0.0584) and below the ILP's
35-edge solution (0.1175). This is exactly the comparison the spec is designed
to make legible: RL's number and the optimisers' numbers are directly
comparable because they are all scored by the same `objective`, and the
difference is due to *search strategy* (RL's shaped-reward hill-climbing
escapes a local optimum that greedy's strict-gain hill-climbing cannot).

### 5. Why the ILP scores highest despite optimising a surrogate

The ILP maximises the coverage surrogate (reachable-node count), not the full
weighted objective. But a solution that covers many nodes also tends to have
high coverage and (if the covered edges connect) high connectivity, so when
re-scored by the full weighted objective it still scores well — here, the best
of the four. The caveat is that its `OPTIMAL` status is *not* a proof that no
higher weighted-objective solution exists; it is a proof that no
higher-coverage solution exists under the budget. For a true weighted ceiling,
the full-objective ILP (§7) is needed.

---

## What this means for the research writeup

1. **Report both objectives.** The weighted objective alone would make greedy
   look broken and RL look like a clear winner — a misleading story. The
   coverage-only run shows greedy is within 0.28% of optimal on a well-behaved
   objective. Both rows belong in the §10 table.
2. **State the submodularity caveat explicitly** (§5). The weighted objective
   is non-monotone due to the fragmentation penalty; the greedy guarantee does
   not apply; this run verifies that empirically.
3. **The RL contribution is "escapes a local optimum greedy cannot", not
   "beats the optimum".** RL is still 14.5% below the ILP on the weighted
   objective. The interesting result is *why* RL beats greedy here (shaped
   reward vs strict-gain hill-climbing on a non-monotone landscape), not that
   it does.

---

## Recommended next steps (out of scope for item 6)

- **Make the reported objective monotone / submodular.** The cleanest fix:
  drop the fragmentation penalty from the *objective* and keep it only as an
  RL reward-shaping term (exactly the §3.2 separation: objective ≠ reward).
  Then greedy regains its guarantee on the reported metric and the comparison
  reverts to "greedy ≥ RL" as the spec expects. Fragmentation can still guide
  RL training without corrupting the optimiser ceiling.
- **Incremental `objective_delta` via `MetricsState`.** `objective.py`'s
  `objective_delta` is currently a full-recompute difference (plan 010
  deferred the incremental version for correctness-first). The coverage-only
  greedy run took **9m41s** purely from re-`objective`-ing the whole graph per
  candidate per round. An incremental delta backed by the union-find +
  persistent rx graph in `metrics.py` (§6.2/§6.3) would cut this to seconds,
  let local search run many more iterations, and make larger cities tractable.
- **Full-objective ILP** (§7) with connectivity encoded via flow/Steiner-style
  auxiliary variables, so the ILP is a true ceiling for the weighted objective
  too (currently the ceiling is only valid for coverage-only weights).
- **Tune RL.** 4096 timesteps across 2 envs is a smoke-test-scale run. The
  14.5% gap to the ILP may close with more timesteps, tuned PPO
  hyperparameters, and/or better reward shaping — a natural follow-up
  experiment once the objective is monotone.

---

## Artefact locations

- **This file:** `FINDINGS.md` (top level).
- **Raw §10 tables (per run, written by the script):**
  `bike_path_figures/compare_otley_uk_<timestamp>/comparison_table.md`
  (gitignored runtime output; the tables above are transcribed from the
  2026-06-25 runs).
- **The runner:** [`scripts/run_comparison.py`](scripts/run_comparison.py)
  (single-command §10-table generator, OPTIMIZER_SPEC §12 DoD).
- **The harness:** [`bike_rl/optim/evaluate.py`](bike_rl/optim/evaluate.py)
  (`evaluate_solver`, `evaluate_rl_policy`, `run_comparison`,
  `format_comparison_table`).
