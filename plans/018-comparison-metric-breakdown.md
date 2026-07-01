# Plan 018: Show component metrics in comparison tables

> **Executor instructions**: Execute only this slice. Run each verification
> command before moving on. If a STOP condition occurs, stop and report — do
> not broaden scope or pull work from neighboring plans.
>
> **Drift check**: `git diff --stat 7570929..HEAD -- bike_rl/optim/evaluate.py tests/test_optim_evaluate.py FINDINGS.md`
> If any in-scope file changed, compare the current-state evidence below with
> live code before editing. Treat mismatches as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: S
- **Risk**: LOW
- **Depends on**: `plans/017-monotone-comparison-objective.md`
- **Category**: direction | docs
- **Planned at**: commit `7570929`, 2026-07-01

## Slice outcome

The optimiser/RL comparison table still ranks by the reported objective, but also shows the underlying component metrics: coverage, connectivity, fragmentation, budget used, runtime, and edge count. Fragmentation is visible to readers without being subtracted from the default headline score.

## Why this matters

Making the headline objective coverage-only fixes the greedy/ILP comparison story, but it can hide whether RL produces a more connected or less fragmented network. The research direction explicitly asks whether solvers create structurally different networks. This slice keeps the headline score simple while showing the values needed to discuss trade-offs.

## Current-state evidence

- `RESEARCH_DIRECTION.md:26` — RQ4 asks whether solvers produce structurally different networks.
- `FINDINGS.md:269` — current writeup says both objective settings should be reported.
- `FINDINGS.md:286` — fragmentation should move out of the reported objective, not disappear.
- `bike_rl/optim/evaluate.py:293` — `format_comparison_table` currently renders the §10 Markdown table.
- `bike_rl/optim/evaluate.py:315` — current rows include only solver, objective, budget, runtime, edge count, gap, and ILP status.
- `bike_rl/objective.py:115` — objective already computes connectivity, coverage, and fragmentation indirectly via metric functions.

## Commands needed

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Focused tests | `pytest tests/test_optim_evaluate.py -q --no-cov` | exit 0; table tests pass |
| Typecheck | `mypy --strict bike_rl` | exit 0 |
| Lint | `ruff check bike_rl/optim/evaluate.py tests/test_optim_evaluate.py` | exit 0 |

## Scope

**In scope**:
- `bike_rl/optim/evaluate.py`
- `tests/test_optim_evaluate.py`
- `FINDINGS.md` only if a short note is needed to explain the new columns

**Out of scope**:
- Changing objective formulas or defaults; that is `plans/017-monotone-comparison-objective.md`.
- Changing solver selection behavior.
- Adding city-level benchmark aggregation or plots.
- Rewriting `Solution` unless a minimal `extra`-based approach cannot work.

## Steps

### Step 1: Add metric breakdown computation to the formatter path

In `bike_rl/optim/evaluate.py`, add a small helper such as `_component_metrics(instance, edges)` that applies the chosen edges to the base graph and returns coverage, connectivity, and fragmentation. Use existing metric functions from `bike_rl.metrics` and `apply_added_edges` from `bike_rl.objective` to avoid duplicate formulas.

**Verify**: `mypy --strict bike_rl` → no new type errors.

### Step 2: Expand `format_comparison_table` columns

When `instance` is provided, include columns for `Coverage`, `Connectivity`, and `Fragmentation ↓` between `Objective` and `Budget used`. Keep the old compact output when `instance is None` if that avoids breaking callers without graph context.

**Verify**: `pytest tests/test_optim_evaluate.py -q --no-cov` → existing table tests pass after expected assertion updates.

### Step 3: Update table tests for component metrics

In `tests/test_optim_evaluate.py`, update `test_format_comparison_table_has_one_row_per_solver` to assert the new headers appear when an instance is supplied. Add one assertion that a known solution row includes numeric component values formatted to four decimals.

**Verify**: `pytest tests/test_optim_evaluate.py -q --no-cov` → all tests pass.

### Step 4: Preserve the ILP surrogate footnote

Keep the existing footnote behavior for non-coverage-only weights. The table should make both things clear: component metrics are shown, and an `OPTIMAL` ILP status is only a true ceiling for coverage-only weights.

**Verify**: `pytest tests/test_optim_evaluate.py::TestRunComparison::test_format_comparison_table_coverage_only_no_footnote -q --no-cov` → passes.

## Test plan

- Update `tests/test_optim_evaluate.py` table assertions for the new headers and row shape.
- Add or update tests only in the existing `TestRunComparison` class.
- Verification: focused test command plus `mypy --strict bike_rl` and targeted `ruff check`.

## Done criteria

All must hold:

- [ ] Comparison tables with an `Instance` show coverage, connectivity, and fragmentation values.
- [ ] Fragmentation column is labelled as lower-is-better, e.g. `Fragmentation ↓`.
- [ ] Objective/gap columns still behave as before.
- [ ] Coverage-only tables still omit the surrogate footnote.
- [ ] Non-coverage-only tables still include the surrogate footnote.
- [ ] Focused tests, typecheck, and lint pass.
- [ ] No files outside the in-scope list are modified, except `plans/README.md` status if instructed.
- [ ] No secret values are added to code, tests, logs, or plan files.

## STOP conditions

Stop and report if:

- Current-state evidence does not match live code after drift check.
- Metric computation requires a broad `Solution` dataclass redesign.
- Table generation becomes slow enough to noticeably affect tests.
- The slice grows beyond 5 implementation steps.
- Verification fails twice after reasonable fixes.

## Maintenance notes

- This plan intentionally keeps metrics in the table rather than mixing them back into a fragile weighted score.
- A future benchmark-matrix plan can aggregate these same columns across cities, budgets, and seeds.
- If `plans/017` is not implemented first, this table still works, but it will display the current weighted objective as the headline score.
