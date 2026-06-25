# Plan 014: Fix `GreedySolver` `TypeError` on exact scoring ties

> **Executor instructions**: Read **this file first**. Then execute
> `step1-fix.md`, `step2-tests.md`, `step3-verify.md` **in order**, running
> each verification command and confirming the expected result before moving
> to the next file. If any "STOP conditions" at the bottom of **this** file
> occurs, stop and report — do not improvise. When done, update the status
> row for this plan in `plans/README.md`.
>
> **Drift check (run first)**: `git diff --stat bc58ad2..HEAD -- bike_rl/optim/greedy.py tests/test_greedy.py`
> If either in-scope file changed since this plan was written, compare the
> excerpts in the step files against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: S
- **Risk**: LOW
- **Depends on**: plans/011-greedy-and-tests.md (DONE) — adds a test to existing `tests/test_greedy.py`; touches only `GreedySolver.solve`
- **Category**: bug
- **Planned at**: commit `bc58ad2`, 2026-06-25

## Why this matters

`GreedySolver.solve` raises `TypeError: '>' not supported between instances of
'Candidate' and 'Candidate'` whenever two affordable candidates produce an
exactly equal `(delta/cost, road_priority, -cost)` score tuple. On real
OSM-derived instances parallel streets of the same class and similar length
routinely tie, so greedy (the performance floor RL must beat, OPTIMIZER_SPEC
§5) and the `LocalSearchSolver` that seeds from it can crash mid-run with no
recovery. This is the "known upstream caveat" plans 012/013 deferred. The fix
is a one-line additive tie-break making greedy fully deterministic without
changing the documented priority or the `Candidate` identity contract.

## File index of this plan

| File | When to read | What it does |
|------|--------------|--------------|
| `step1-fix.md` | before editing | Edits `bike_rl/optim/greedy.py` (the `max(scored)` loop) |
| `step2-tests.md` | after step 1 green | Adds `TestGreedyTieBreak` class to `tests/test_greedy.py` |
| `step3-verify.md` | after step 2 green | Runs the full QA gate + the reproduction script |

## Scope

**In scope** (the only files you should modify):
- `bike_rl/optim/greedy.py` — the `while remaining:` scoring loop inside
  `GreedySolver.solve` only.
- `tests/test_greedy.py` — append one new test class `TestGreedyTieBreak`.

**Out of scope** (do NOT touch, even though they look related):
- `bike_rl/candidates.py` — do NOT add `order=True` to `Candidate`. It is a
  frozen dataclass with identity `(u, v, length, road_priority)`
  (`connects_to_bike_path`/`data` are `compare=False`), used by the env,
  `extract_candidates`, set membership, and every optimiser. Adding ordering
  broadens its surface, and `u`/`v` are `int | str` so a generated `__lt__`
  would itself raise `TypeError` on mixed node ids — it wouldn't even fix the
  bug universally. The fix belongs in `greedy.py`.
- `bike_rl/optim/local_search.py` — seeds from greedy and inherits the fix;
  no change needed.
- `bike_rl/optim/ilp.py`, `evaluate.py` — stubs; leave alone.
- The `GreedySolver` docstring, `Solution`, `objective`, `objective_delta`,
  `budget.cost` — reuse unchanged.
- The greedy priority *order* (`delta/cost`, `road_priority`, `-cost`). Only
  a **final** stable key is appended; do not reorder or drop existing keys.

## Commands you will need

Run all commands from the repo root with the venv active (`source .venv/bin/activate`).

| Purpose | Command | Expected on success |
|---|---|---|
| Run greedy tests only | `pytest -q tests/test_greedy.py` | all pass |
| Full suite | `pytest -q` | all pass, coverage ≥80% |
| Lint | `ruff check .` | exit 0 |
| Format check | `ruff format --check .` | exit 0 (run `ruff format .` if it reports diffs) |
| Type check | `mypy --strict bike_rl` | exit 0 |

## Git workflow

- Branch: `advisor/014-greedy-tiebreak-fix` (matches the `advisor/NNN-<slug>` convention).
- Two commits, conventional-commit style (see `git log --oneline -10`):
  1. `fix(optim): make GreedySolver tie-break deterministic (OPTIMIZER_SPEC §5)`
  2. `test(optim): add greedy exact-tie regression test`
- Do NOT push or open a PR unless the operator instructed it.

## Done criteria (machine-checkable, ALL must hold)

- [ ] `pytest -q tests/test_greedy.py` exits 0; the 4 new `TestGreedyTieBreak`
      tests pass.
- [ ] `pytest -q` exits 0; whole-package coverage ≥80%.
- [ ] `ruff check .` exits 0.
- [ ] `ruff format --check .` exits 0.
- [ ] `mypy --strict bike_rl` exits 0.
- [ ] `grep -n "_, _, _, best = max(scored)" bike_rl/optim/greedy.py` returns
      no matches; `grep -n "_, _, _, _, best" bike_rl/optim/greedy.py` returns
      the new unpack.
- [ ] No files outside `bike_rl/optim/greedy.py`, `tests/test_greedy.py`, and
      `plans/README.md` are modified (`git status --short`).
- [ ] The reproduction script in `step3-verify.md` prints `OK 1 1000.0` with
      no traceback.
- [ ] `plans/README.md` status row for plan 014 updated (TODO → DONE).

## STOP conditions

Stop and report back (do not improvise) if:

- The code at the locations in the step files doesn't match the excerpts (the
  codebase has drifted since `bc58ad2`) — in particular if `GreedySolver.solve`
  no longer uses `max(scored)` with a 4-tuple, or if `Candidate` has gained
  `order=True` (the bug may already be fixed differently; re-evaluate rather
  than layering a second tie-break).
- `Candidate` is no longer a frozen dataclass with the documented
  `(u, v, length, road_priority)` identity (plan 002 contract changed).
- The two candidates in `_tie_candidates` do **not** produce exactly equal
  `objective_delta` on your environment (the regression relies on an exact
  tie). Report so the fixture can be hardened rather than silently shipping a
  non-triggering test.
- `step2-tests.md`'s `test_tie_break_picks_earliest_candidate` fails because
  greedy picks the *latest* candidate on a full tie — do NOT weaken the test;
  fix step 1 to use `-i` (earliest wins, the deliberate direction).
- The fix appears to require touching `candidates.py` or `local_search.py` —
  it must not; report instead.