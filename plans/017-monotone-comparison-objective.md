# Plan 017: Make comparisons use a monotone reported objective

> **Executor instructions**: Execute only this slice. Run each verification
> command before moving on. If a STOP condition occurs, stop and report — do
> not broaden scope or pull work from neighboring plans.
>
> **Drift check**: `git diff --stat 7570929..HEAD -- scripts/run_comparison.py tests/test_run_comparison_script.py FINDINGS.md`
> If any in-scope file changed, compare the current-state evidence below with
> live code before editing. Treat mismatches as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: S
- **Risk**: MED
- **Depends on**: `plans/016-*` complete
- **Category**: correctness | direction
- **Planned at**: commit `7570929`, 2026-07-01

## Slice outcome

`python scripts/run_comparison.py ...` ranks solvers by a monotone coverage-only reported objective by default. The old weighted objective remains available through an explicit CLI flag so prior findings can still be reproduced.

## Why this matters

The current default weighted objective subtracts fragmentation, so adding a useful edge can temporarily reduce the score. That made greedy stop after one edge in the Otley comparison, while coverage-only greedy reached 99.72% of the ILP optimum. This slice makes the default comparison score match the valid ILP ceiling and keeps fragmentation available for reward shaping / diagnostics rather than headline ranking.

## Current-state evidence

- `FINDINGS.md:24` — the weighted objective made RL beat greedy because greedy stalled.
- `FINDINGS.md:29` — coverage-only greedy reached 99.72% of the ILP optimum.
- `FINDINGS.md:286` — recommended fix is to drop fragmentation from the reported objective and keep it as RL reward shaping.
- `scripts/run_comparison.py:78` — there is a `--coverage-only` flag today, meaning the non-monotone weighted objective is still the default.
- `scripts/run_comparison.py:164` — `Config(...)` is built locally in the script, so changing comparison defaults can be isolated here.
- `scripts/run_comparison.py:206` — the script already routes all solver rows through the shared comparison formatter.

## Commands needed

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Focused tests | `pytest tests/test_run_comparison_script.py -q --no-cov` | exit 0; new script tests pass |
| Existing comparison tests | `pytest tests/test_optim_evaluate.py -q --no-cov` | exit 0; harness behavior still passes |
| Typecheck | `mypy --strict bike_rl` | exit 0 |
| Lint | `ruff check scripts/run_comparison.py tests/test_run_comparison_script.py` | exit 0 |

## Scope

**In scope**:
- `scripts/run_comparison.py`
- `tests/test_run_comparison_script.py`
- `FINDINGS.md` only if a short note is needed to document the new default

**Out of scope**:
- Changing `bike_rl.objective.ObjectiveWeights` defaults globally.
- Changing RL reward calculation in `bike_rl/env.py`.
- Adding metric breakdown columns; that is `plans/018-comparison-metric-breakdown.md`.
- Running live OSM comparisons as part of unit verification.

## Steps

### Step 1: Change the comparison CLI objective flags

In `scripts/run_comparison.py`, make coverage-only weights the default:
`ObjectiveWeights(connectivity=0.0, coverage=1.0, fragmentation=0.0)`.
Keep `--coverage-only` accepted as a backward-compatible no-op, and add an explicit flag such as `--weighted-objective` to use the previous `ObjectiveWeights()` behavior.

**Verify**: `python scripts/run_comparison.py --help | grep -E "coverage-only|weighted-objective"` → both flags are visible.

### Step 2: Update the sanity-check messaging

Adjust the top docstring and printed diagnosis so it no longer says greedy ≥ RL is expected for every default run. The default run should describe coverage-only objective as the headline comparison; the weighted run should be labelled as an objective-sensitivity / legacy diagnostic.

**Verify**: `rg -n "greedy ≥ RL|weighted-objective|coverage-only" scripts/run_comparison.py` → wording distinguishes default coverage-only from weighted diagnostic.

### Step 3: Add script-level tests without network access

Create `tests/test_run_comparison_script.py`. Import `scripts.run_comparison`, exercise `_build_parser()`, and assert:
- default parsed args produce coverage-only mode unless `--weighted-objective` is set;
- `--coverage-only` remains accepted;
- the helper logic that chooses weights returns coverage-only by default and weighted weights only for the explicit legacy flag.

If weight selection is currently inline inside `main`, extract a small private helper like `_objective_weights(args)` in `scripts/run_comparison.py` to keep this test network-free.

**Verify**: `pytest tests/test_run_comparison_script.py -q --no-cov` → all new tests pass.

### Step 4: Preserve existing harness behavior

Run the existing optimiser-evaluation tests to confirm the script change did not alter the shared evaluator API or table formatter behavior.

**Verify**: `pytest tests/test_optim_evaluate.py -q --no-cov` → all tests pass.

## Test plan

- Add `tests/test_run_comparison_script.py` for parser and objective-weight selection.
- Prefer testing private helpers over invoking `main()`, because `main()` loads OSM graphs.
- Verification: focused tests above, plus `mypy --strict bike_rl` and targeted `ruff check`.

## Done criteria

All must hold:

- [ ] Default comparison runs use coverage-only objective weights.
- [ ] Legacy weighted objective remains available via an explicit flag.
- [ ] `--coverage-only` remains accepted for backward compatibility.
- [ ] New tests verify default and legacy weight selection without network access.
- [ ] Existing optimiser-evaluation tests still pass.
- [ ] No files outside the in-scope list are modified, except `plans/README.md` status if instructed.
- [ ] No secret values are added to code, tests, logs, or plan files.

## STOP conditions

Stop and report if:

- Current-state evidence does not match live code after drift check.
- The change requires editing `bike_rl/env.py` or changing RL reward semantics.
- The script cannot be tested without OSM/network access.
- The slice grows beyond 5 implementation steps.
- Verification fails twice after reasonable fixes.

## Maintenance notes

- This is a comparison/reporting default change, not a claim that fragmentation is unimportant.
- `plans/018-comparison-metric-breakdown.md` should follow so fragmentation/connectivity remain visible in the table.
- Future benchmark runners should inherit the coverage-only default unless explicitly running objective-sensitivity experiments.
