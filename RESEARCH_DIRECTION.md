# Research Direction — Bike-Path Network Expansion Optimisation

This document frames the **research goals, questions, methodology, and intended contributions** of revisiting the undergraduate RL bike-path project with stronger post-grad knowledge. It is the "why" that governs the two engineering specs (`RECREATE_SPEC.md` for RL, `OPTIMIZER_SPEC.md` for the direct-optimiser baseline). Read this first.

---

## 1. Background & motivation

Urban cycling networks are typically built incrementally over years under fixed budgets. The *network expansion* problem — *which road segments to upgrade next, given a budget, to maximise the quality of the resulting bike network* — is a combinatorial optimisation problem on real-world street graphs. It matters because:

- Connectivity and coverage of a bike network strongly influence cycling uptake; fragmented networks underperform.
- Municipal budgets are tight, so the *ordering* and *selection* of segments has real-world value.
- The problem combines graph structure (connectivity, fragmentation), spatial structure (coverage of population/destinations), and a knapsack constraint (budget) — making it rich enough to study but concrete enough to evaluate.

The original undergraduate project (`COMP3931/nx-rx-simple-ur.py`) framed this as a **single-agent reinforcement learning** problem: a PPO agent sequentially adds edges under a budget, rewarded by a hand-tuned combination of connectivity, path-efficiency, population-served, and fragmentation metrics. It worked in the sense that it produced plausible maps, but it had fundamental weaknesses as *research*: no baselines, an unvalidated objective, a near-dead population metric, a degenerate action space, and no analysis of *whether RL is the right tool*.

## 2. The core research question

> **For budgeted bike-network expansion on real city graphs, does a learned RL policy offer measurable advantages over direct combinatorial optimisation (greedy / local search / ILP), and under what conditions?**

This decomposes into sub-questions:

- **RQ1 (performance):** How does PPO compare to greedy, greedy + local search, and ILP on solution quality (the shared objective) at matched budgets?
- **RQ2 (scalability):** How do the approaches scale with graph size and candidate count, in runtime and solution quality? Where does ILP stop being tractable, and do the heuristics/RL still scale?
- **RQ3 (generalisation):** Can a single RL policy trained across many cities generalise to an unseen city *without retraining*, and does it remain competitive with per-instance greedy on that unseen city? (This is the only setting where RL's added complexity is clearly justified.)
- **RQ4 (solution diversity/structure):** Do the different solvers produce *structurally different* networks (e.g. RL favouring connected corridors vs greedy favouring scattered high-coverage edges)? Is there a qualitative difference a planner would care about?
- **RQ5 (objective sensitivity):** How sensitive are the rankings to the choice of objective (coverage-only vs connectivity vs combined)? Are the conclusions robust?

Answering these — even partially — is a genuine contribution; the original project couldn't ask them because it had no baselines and no validated objective.

## 3. Hypotheses

- **H1:** On a single fixed city, greedy-by-marginal-gain/cost will match or beat an undertrained PPO, because the problem is deterministic and the objective is (approximately) submodular — greedy has a theoretical guarantee and the RL signal is weak.
- **H2:** Greedy + local search will close most of the gap to the ILP optimum on small/medium instances, establishing a strong practical baseline.
- **H3:** ILP will be optimal but won't scale past a few hundred candidate edges / a few thousand nodes, defining the regime where heuristics are necessary.
- **H4:** A *generalising* RL policy (GNN observation + edge-level policy, trained on many cities) will be competitive with per-instance greedy on unseen cities while avoiding per-instance retraining — the defensible RL win, if any.
- **H5:** RL's value (if any) lies in RQ4 — producing structurally different, more "robust" or diverse networks — rather than in raw objective value.

H1–H3 are near-certain and establish the baseline story. H4–H5 are the research bets.

## 4. Approach

### 4.1 Two solvers, one problem

Build **both** the RL pipeline (per `RECREATE_SPEC.md`) and the direct optimisers (per `OPTIMIZER_SPEC.md`) in a single shared codebase, using a **common objective module** (`objective.py`), **common candidate extraction** (`candidates.py`), and **common metrics** (`metrics.py`). The RL reward is *derived* from the canonical objective; the optimisers call it directly; both report the same number. This guarantees the comparison is fair — differences are due to search strategy, not problem formulation.

### 4.2 Formalise the objective

Separate *what we're optimising* from *how we learn to optimise it*. Define one or a small family of scalar objectives (e.g. coverage-only, connectivity-weighted, fragmentation-penalised) with explicit weights documented in `Config`. Verify the objective is **monotone** and test for **submodularity** empirically (this determines whether greedy's guarantee applies and is itself a methodological note). Make the objective a pure, deterministic, incrementally-evaluable function.

### 4.3 Fix the measurement

The original "population served" metric barely moved because it was defined over the whole graph. Redefine coverage as *bike-lane-reachable population* (nodes within `X` metres of a bike lane, or in the same connected component as one) so the metric is **sensitive** enough to distinguish policies. Validate sensitivity: on a tiny fixture, adding an edge must produce a measurable objective change.

### 4.4 Evaluation methodology

- **Multiple seeds** (≥5) for the stochastic RL runs; report mean ± std and confidence intervals.
- **Deterministic** solvers (greedy, local search, ILP) run once per instance.
- **Cities as the unit of evaluation.** Curate a small benchmark of cities spanning sizes (e.g. Otley, Leeds, Bristol, a Berlin district) so scalability (RQ2) and generalisation (RQ3) can be studied.
- **Train/test split for RQ3:** train the generalising policy on a set of cities, evaluate on held-out cities, compare to per-instance greedy run on each held-out city.
- **Report optimality gap** against ILP on the small instances where ILP solves to optimality.
- **Runtime budgets** reported alongside solution quality — RL training cost amortised over the number of deployment cities.

### 4.5 The generalising-policy extension (RQ3/RQ4, the research bet)

The single-city RL setup is the weakest case for RL (it's just an expensive optimiser). The defensible angle is a **policy that generalises across cities**:

- **Observation:** a graph representation (GNN or hand-crafted node/edge features: degree, road type, distance to existing bike network, population density proxy) so the policy isn't tied to one graph's node IDs.
- **Action:** an edge-level policy over the candidate set (score each candidate, sample/argmax within budget) rather than a `Discrete(fixed_index)` tied to one city's candidate count — this is the bug that broke the original action space.
- **Training distribution:** many cities of varying sizes; data augmentation via random subgraphs.
- **Evaluation:** held-out cities, no retraining, compared to per-instance greedy.

This is substantially more ambitious than the original and may or may not pay off — which is exactly why H4/H5 are research questions and not assumptions. If it doesn't pay off, that's a finding: "single-city RL is not competitive with greedy; a generalising GNN policy closes but doesn't exceed the gap while adding training cost." Either outcome is publishable.

## 5. Intended contributions

1. **A fair, shared-objective comparison** of RL vs direct combinatorial optimisation (greedy / local search / ILP oracle) for budgeted bike-network expansion — something the original lacked and the literature often lacks.
2. **A formalised, validated objective** with documented (non-)submodularity, separating the optimisation target from the learning signal.
3. **A reproducible, modular benchmark** (shared codebase, pinned deps, cached OSM, multiple cities, multiple seeds, CI) that future work can extend.
4. **A characterisation of regimes**: where ILP is tractable, where greedy suffices, and whether/where a generalising RL policy earns its complexity.
5. **Qualitative analysis of solution structure** (RQ4): do learned policies produce meaningfully different networks, and is that difference planner-relevant?

## 6. Scope & non-goals

- **Not** modelling dynamic demand or temporal network evolution (that's a separate, online-RL setting — noted as future work in §7).
- **Not** predicting cycling uptake from network quality (would require behavioural data; out of scope).
- **Not** multi-objective Pareto optimisation in the first pass — start with a scalarised objective; Pareto fronts are a natural extension once the scalar comparison is solid.
- **Not** replacing the planner: the output is a *decision-support* ranking of candidate segments, not a construction plan.

## 7. Risks & mitigations

| Risk                                                                       | Mitigation                                                                                                                                    |
| -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| RL never beats greedy on a single city (H1 holds strongly)                 | That's an expected and reportable result; pivot the contribution to the generalising-policy (RQ3) and the qualitative-structure (RQ4) angles. |
| Generalising policy doesn't transfer across cities of very different sizes | Normalise features by graph size; train on size-binned groups; report per-size-bin generalisation.                                            |
| Objective turns out non-submodular, so greedy's guarantee doesn't apply    | Document it; report greedy's empirical gap to ILP anyway; submodularity is sufficient but not necessary for good greedy performance.          |
| ILP never solves on any realistic city                                     | Use small synthetic / subgraph instances as the oracle regime; report the scalability frontier as a result.                                   |
| OSM data drift between runs                                                | Pin OSMnx version + cache downloaded graphs to disk; record the download date per city in the run summary.                                    |
| Compute budget for the generalising policy is large                        | Start with the single-city RL + greedy/ILP comparison (cheap, publishable on its own); treat the generalising policy as a stretch goal.       |

## 8. Success criteria

- **Minimum viable contribution (always achievable):** a clean, reproducible, shared-objective comparison showing where greedy / local search / ILP / single-city PPO sit on solution quality, runtime, and scalability across a small city benchmark. This alone is a substantial improvement over the original.
- **Stretch contribution (the bet):** a generalising GNN-based edge policy that is competitive with per-instance greedy on held-out cities without retraining, plus a qualitative analysis of the structural differences in the networks each method produces.
- **Either way:** a methodology section that justifies *why* RL is or isn't appropriate, grounded in measured results rather than assumption.

## 9. Relationship to the engineering specs

- `RECREATE_SPEC.md` — implements the RL side (cleaned, fixed, modular) and the shared `candidates.py` / `metrics.py` / `objective.py` modules.
- `OPTIMIZER_SPEC.md` — implements the direct-optimiser baselines reusing those shared modules.
- This document — decides *what questions to ask of the resulting system* and *what counts as a contribution*.

Build order: shared modules + objective → optimisers (fast, gives the baseline numbers immediately) → fixed single-city RL → comparison → (stretch) generalising policy. Getting the optimiser baseline working first is deliberate: it gives you the performance floor/ceiling *before* investing in RL, so you always know whether the RL work is paying off.
