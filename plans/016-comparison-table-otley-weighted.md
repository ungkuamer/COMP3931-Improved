| Solver | Objective | Budget used | Runtime | # edges | Optimality gap (vs ILP) | ILP status |
|---|---|---|---|---|---|---|
| greedy | 0.0584 | 1138.7 | 26.98s | 1 | 50.28% | — |
| local_search | 0.0584 | 1138.7 | 1m00s | 1 | 50.28% | — |
| ilp | 0.1175 | 70314.0 | 0.32s | 35 | 0.00% | OPTIMAL |
| rl | 0.1004 | 199050.5 | 9.01s | 112 | 14.52% | — |

**Instance:** otley_uk

> Note: ILP status is OPTIMAL over its **coverage surrogate**, not the full weighted objective (weights are not coverage-only). For a true optimality ceiling, re-run with `ObjectiveWeights(connectivity=0.0, coverage=1.0, fragmentation=0.0)`.

**Sanity check:** RL (0.1004) > greedy (0.0584) — RL clears the greedy bar. ✅

*Total wall time: 298.2s.*
