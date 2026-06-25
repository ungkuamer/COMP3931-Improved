# Step 1: Add the deterministic final tie-break key in `GreedySolver.solve`

File: `bike_rl/optim/greedy.py`. Inside `GreedySolver.solve`, in the
`while remaining:` loop, the score tuple is a 4-tuple ending with the
`Candidate`. `max(scored)` compares tuples element-wise; when two candidates
share the same first three numeric keys, Python compares the 4th element — a
`Candidate` (frozen dataclass, no `order=True`) → `TypeError`.

Fix: append a final, always-comparable key `-index` (negated input index) so
the earliest candidate in the current `remaining` order wins a full tie, and
`Candidate` is never compared.

### Replace this exact block

```python
        while remaining:
            scored: list[tuple[float, int, float, Candidate]] = []
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
```

### With this exact block

```python
        while remaining:
            # Score = (gain-per-cost, road_priority, -cost, -index, edge).
            # The first three keys are the documented OPTIMIZER_SPEC §5
            # tie-break. The final ``-index`` key is a stable, always-comparable
            # tie-breaker: ``Candidate`` is a frozen dataclass without
            # ``order=True`` (identity = (u, v, length, road_priority), plan
            # 002), so without it ``max`` would compare ``Candidate`` objects
            # and raise ``TypeError`` on exact score ties. ``-index`` makes
            # the earliest candidate in the (current) remaining order win a
            # full tie; it never reorders non-tied candidates.
            scored: list[tuple[float, int, float, int, Candidate]] = []
            for i, e in enumerate(remaining):
                c = cost(e, self.cfg)
                if spent + c > budget:
                    continue
                delta = objective_delta(graph, S, e, self.weights, self.cfg)
                if delta <= 0.0:
                    continue
                scored.append((delta / c, e.road_priority, -c, -i, e))
            if not scored:
                break
            _, _, _, _, best = max(scored)
            S.append(best)
            spent += cost(best, self.cfg)
            remaining.remove(best)
```

### Rules

- The tuple annotation grows from a 4-tuple to `tuple[float, int, float, int, Candidate]` — keep it exact (ruff/mypy strict check it).
- The unpack grows from `_, _, _, best` to `_, _, _, _, best`.
- `enumerate(remaining)` is over the **current** `remaining` list each round (it shrinks via `remove` between rounds). `list.remove` preserves order, so "earliest in remaining" is deterministic and reproducible.
- Change **nothing else** in the file: imports, docstring, `Solution`, the `return Solution(...)` block stay untouched.

### Verify

- `ruff check bike_rl/optim/greedy.py` → exit 0
- `ruff format --check bike_rl/optim/greedy.py` → exit 0 (run `ruff format bike_rl/optim/greedy.py` if it reports a diff)
- `mypy --strict bike_rl` → exit 0
- `pytest -q tests/test_greedy.py` → all existing tests still pass (additive; existing selection order on non-tied instances is unchanged; `test_deterministic_across_runs` still green)

Proceed to `step2-tests.md` only after all four pass.