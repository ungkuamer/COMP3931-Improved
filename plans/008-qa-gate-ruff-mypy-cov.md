# Plan 008: Enforce the ≥80% coverage gate in CI (`pytest --cov`)

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**: `git diff --stat 8a04ebf..HEAD -- pyproject.toml .github/workflows/ci.yml`
> If either file changed since this plan was written, compare the "Current
> state" excerpts against the live code before proceeding; on a mismatch,
> treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: S
- **Risk**: LOW
- **Depends on**: plans/007-cli-and-tests.md (all `bike_rl/` modules + tests must exist)
- **Category**: dx | tech-debt
- **Planned at**: commit `8a04ebf`, 2026-06-23
- **Issue**: (omitted — not published via `--issues`)

## Why this matters

`RECREATE_SPEC.md` §11 item 8 and §12 (Definition of Done) require that
`ruff check`, `mypy --strict bike_rl`, and `pytest --cov` are all green with
≥80% coverage on `bike_rl/`, and that CI enforces this. As of SHA `8a04ebf`
(after the vendored caveman skill scripts were removed from `.agents/`), the
three quality commands are already green when run by hand:

- `ruff check .` → `All checks passed!`
- `ruff format --check .` → `30 files already formatted`
- `mypy --strict bike_rl` → `Success: no issues found in 18 source files`
- `pytest -q --cov=bike_rl --cov-report=term-missing` → `100 passed`,
  `TOTAL 943 101 89%`.

**The remaining gap is enforcement.** `.github/workflows/ci.yml` runs
`pytest -q` with no `--cov` and no `--cov-fail-under`. Coverage can silently
regress below the 80% DoD threshold and CI still passes. The Definition of
Done is unenforced.

This plan adds the coverage gate to `pytest`'s `addopts` in `pyproject.toml`
— a single config edit, no source changes — so both local `pytest` and CI
`pytest -q` fail non-zero if `bike_rl/` coverage drops below 80%. CI is
green by construction (advisor measured 89.29% with margin), and the gate
becomes self-policing. This is the prerequisite for plan 009's end-to-end
smoke test and for the downstream optimiser plans.

## Current state

**`.github/workflows/ci.yml`** — the `lint-type-test` job. Relevant steps
(verified at SHA `8a04ebf`):

```yaml
      - name: Ruff lint
        run: ruff check .
      - name: Ruff format check
        run: ruff format --check .
      - name: Mypy
        run: mypy --strict bike_rl
      - name: Pytest
        run: pytest -q
```

All four steps already use the right commands. The `Pytest` step has no
coverage enforcement — it relies entirely on `addopts` for flags, and
`addopts` currently has none.

**`pyproject.toml`** — the pytest section (verified at SHA `8a04ebf`):

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"
```

There is **no** `--cov` flag and **no** `[tool.coverage.*]` section.

**Verified recon facts** (run by the advisor at SHA `8a04ebf`, Python 3.12,
ruff 0.15, mypy 2.1, pytest 9.1, pytest-cov 7.1, after the caveman skill
scripts were removed from `.agents/`):

- `ruff check .` → `All checks passed!` (no exclude needed —
  `.agents/skills/improve` contains only `.md` files, which ruff ignores).
- `ruff format --check .` → `30 files already formatted`.
- `mypy --strict bike_rl` → `Success: no issues found in 18 source files`.
- `pytest -q --cov=bike_rl --cov-report=term-missing` → `100 passed`,
  `TOTAL 943 101 89%`. Lowest-covered real module is `bike_rl/objective.py`
  at 79%; `bike_rl/optim/*` stubs are 0% (unimplemented, per
  `OPTIMIZER_SPEC.md` §11 — future optimiser plans). Total 89% > 80%.
- Adding `--cov=bike_rl --cov-report=term-missing --cov-fail-under=80` to
  `addopts` makes `pytest -q` print
  `Required test coverage of 80% reached. Total coverage: 89.29%` and
  exit 0.

**Repo conventions to honor**:

- Config lives in `pyproject.toml` (single source of truth — no `setup.cfg`,
  no `tox.ini`, no `.coveragerc`). Add the new flags there; do not create
  sidecar config files and do not add a `[tool.coverage.*]` section — the
  `addopts` flags are self-contained and match the `pytest --cov` command in
  `RECREATE_SPEC.md` §11 item 8 literally.
- The `addopts` value is a single double-quoted string. Keep it that way;
  append the new flags inside the same string, space-separated.
- CI uses plain `pytest -q` and relies on `addopts` for flags — keep that
  pattern; do not duplicate `--cov` flags into the CI yaml. (Rationale:
  local `pytest` and CI `pytest` then enforce the same gate from one place.)
- Commit messages follow Conventional Commits — e.g.
  `ci: enforce ≥80% coverage gate on bike_rl` (see `git log --oneline` for
  the `feat(...)`/`fix(...)`/`test(...)`/`refactor(...)` style used in this
  repo).

## Commands you will need

| Purpose              | Command                                                      | Expected on success |
|----------------------|--------------------------------------------------------------|---------------------|
| Ruff lint (repo-wide)| `ruff check .`                                               | `All checks passed!`, exit 0 |
| Ruff format check    | `ruff format --check .`                                      | `N files already formatted`, exit 0 |
| Mypy                 | `mypy --strict bike_rl`                                      | `Success: no issues found in 18 source files`, exit 0 |
| Tests + coverage gate| `pytest -q`                                                  | `100 passed` (or more), `Required test coverage of 80% reached. Total coverage: NN.NN%`, exit 0 |
| Confirm no source touched | `git status --short bike_rl tests`                    | empty output |

All four quality commands must be run from the repo root
(`/home/ungku/programming/COMP3931-Improved`).

## Scope

**In scope** (the only files you should modify):
- `pyproject.toml` — extend the `[tool.pytest.ini_options] addopts` string
  with the coverage flags.

**Out of scope** (do NOT touch, even though they look related):
- `.github/workflows/ci.yml` — no change required. The `pytest -q` step
  picks up the new `addopts` automatically, and the ruff/mypy steps already
  pass. Only touch this file if a step's command has drifted from what's
  quoted in "Current state" (see Step 2's STOP condition).
- `bike_rl/optim/*` — unimplemented stubs (per `OPTIMIZER_SPEC.md` §11).
  Their 0% coverage is expected. **Do not** add `# pragma: no cover` to
  them and **do not** add an `[tool.coverage.run] omit` for them. They pull
  the total down to 89%, which still clears 80%, and leaving them in the
  measurement means the gate will correctly force the future optimiser
  plans to add tests. (See "Maintenance notes".)
- `bike_rl/objective.py` (79% covered) — do not add tests here to bump the
  number; 89% total already passes. Test improvements belong in the
  optimiser plans that will actually use `objective.py`.
- `.agents/` — vendored agent skill tooling. It is **not** excluded from
  ruff in this plan because it currently contains no `.py` files (only
  `.md`), so `ruff check .` is already green. Do not add an
  `extend-exclude` preemptively — see "Maintenance notes" for the policy
  if a tracked `.py` is ever added under `.agents/`.
- `requirements*.txt`, `pyproject.toml` dependencies, CI Python matrix,
  pip caching — not part of this gate.

## Git workflow

- Branch: `advisor/008-qa-gate` (matches the `advisor/NNN-<slug>` convention;
  no existing branch-naming convention is evident in `git log`).
- Single commit is appropriate (one logical change). Suggested message:
  `ci: enforce ≥80% coverage gate on bike_rl`
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Add the coverage gate to `pytest` via `addopts`

In `pyproject.toml`, in the `[tool.pytest.ini_options]` table, replace the
`addopts` line with:

```toml
addopts = "-ra --strict-markers --cov=bike_rl --cov-report=term-missing --cov-fail-under=80"
```

Keep it as a single double-quoted string. Do not change `testpaths` and do
not add a separate `[tool.coverage.*]` section — the `addopts` flags are
self-contained. Do not modify any other section of `pyproject.toml`.

**Verify**:

```
mypy --strict bike_rl
```
→ `Success: no issues found in 18 source files`, exit 0. (Unchanged by this
edit, but re-run to confirm no accidental edit to the mypy block.)

```
ruff check .
```
→ `All checks passed!`, exit 0. (Unchanged, but re-run to confirm no
accidental edit to the ruff block.)

```
pytest -q
```
→ all tests pass (`N passed`), and the coverage footer prints
`Required test coverage of 80% reached. Total coverage: <NN.NN>%` with
`<NN.NN>` ≥ 80, exit 0.

If you see `Required test coverage of 80% not reached` (non-zero exit),
**STOP** — do not lower the threshold and do not add `# pragma: no cover`.
Report the actual percentage and the missing-line report. Something
regressed since SHA `8a04ebf` (advisor measured 89.29%); that must be
diagnosed, not papered over.

### Step 2: Confirm CI yaml needs no change

Open `.github/workflows/ci.yml`. Confirm the four step commands are still
exactly:

```yaml
        run: ruff check .
...
        run: ruff format --check .
...
        run: mypy --strict bike_rl
...
        run: pytest -q
```

If they match, **make no edit** — `pytest -q` now enforces coverage via
`addopts`, and `ruff check .` / `ruff format --check .` / `mypy` already
pass. CI is green by construction.

If any of those four `run:` lines have drifted (different command, extra
flags, different paths), **STOP** and report the drift — do not rewrite the
CI yaml to match the plan's quoted version, because the drift means someone
changed CI intent and that needs human review, not an automated revert.

**Verify** (read-only — do not run CI locally):

```
grep -nE 'run: (ruff check \.|ruff format --check \.|mypy --strict bike_rl|pytest -q)' .github/workflows/ci.yml
```
→ exactly four matching lines, one per command above.

## Test plan

No new tests are written in this plan — it is a config-only gate. The
"tests" are the four quality commands above, which must all exit 0.

However, confirm the gate actually *bites* (so a future regression is
caught): the Step 1 verification output must contain the line
`Required test coverage of 80% reached. Total coverage: <NN>%`. That line
proves `--cov-fail-under` is active — without it, `pytest` would exit 0
even at 0% coverage.

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `ruff check .` exits 0 with `All checks passed!`
- [ ] `ruff format --check .` exits 0 with no `Would reformat` lines
- [ ] `mypy --strict bike_rl` exits 0, `Success: no issues found`
- [ ] `pytest -q` exits 0, prints
      `Required test coverage of 80% reached. Total coverage: <NN>%` with
      `<NN>` ≥ 80
- [ ] `git status --short bike_rl tests` is empty (no source/test files
      modified — this is a config-only plan)
- [ ] `grep -n 'cov-fail-under' pyproject.toml` returns exactly one line
      (inside the `addopts` string)
- [ ] `grep -n 'cov=bike_rl' pyproject.toml` returns exactly one line
      (inside the `addopts` string)
- [ ] `git diff --stat` shows only `pyproject.toml` changed (CI yaml
      unchanged unless Step 2 legitimately found drift, which is a STOP)
- [ ] `plans/README.md` status row for 008 updated to DONE

## STOP conditions

Stop and report back (do not improvise) if:

- `pyproject.toml` or `.github/workflows/ci.yml` at the locations in
  "Current state" doesn't match the excerpts (the codebase has drifted
  since SHA `8a04ebf`).
- `ruff check .` fails — the advisor verified it green at SHA `8a04ebf`
  after the caveman skill removal, so a failure means either the `.agents/`
  tree changed (a tracked `.py` was added under it — see "Maintenance
  notes") or project code has a lint error. Neither is something to paper
  over with an exclude; report the error.
- `mypy --strict bike_rl` fails — a real type error in project code, not
  in scope. Report it.
- `pytest -q` fails with `Required test coverage of 80% not reached`.
  Something regressed coverage below 80% since SHA `8a04ebf` (advisor
  measured 89.29%). Do not lower the threshold or add pragmas. Report the
  actual number and the missing-line report.
- The CI yaml's four `run:` lines do not match the quotes in Step 2.
- Any verification command fails twice after a reasonable fix attempt.

## Maintenance notes

For the human/agent who owns this code after the change lands:

- **`addopts` is the single source of truth for the coverage gate.** Both
  local `pytest` and CI `pytest -q` read it. If you need a one-off local
  run without coverage, use `pytest -o addopts="" -q` rather than editing
  `pyproject.toml`.
- **Coverage gate is 80% on all of `bike_rl/`, including the `optim/`
  stubs.** The `bike_rl/optim/*` modules are currently 0%-covered stubs
  (per `OPTIMIZER_SPEC.md` §11). They are deliberately **not** omitted from
  coverage: total is 89% with them included, and leaving them in means the
  future optimiser plans (greedy/local_search/ilp/evaluate) *must* add
  tests or CI goes red. When those plans land, do **not** add
  `[tool.coverage.run] omit = ["bike_rl/optim/*"]` as a shortcut — write
  the tests. If the optimiser work is abandoned and the stubs are deleted,
  coverage rises; no action needed here.
- **`.agents/` ruff policy.** `.agents/` holds vendored pi-agent skill
  tooling (currently only the `improve` skill, which is `.md`-only). It is
  not excluded from ruff because ruff ignores non-`.py` files anyway and
  `ruff check .` is green. **If a tracked `.py` is ever added under
  `.agents/`** (e.g. a skill with helper scripts), decide deliberately:
  either bring it under the project's ruff rules (fix any lint errors) or
  add `extend-exclude = [".agents"]` under `[tool.ruff]` in
  `pyproject.toml`. Do not let `ruff check .` silently start failing or
  silently exclude new project code. The same applies to any new tracked
  Python directory added at the repo root.
- **What a reviewer should scrutinize in the PR:** (1) the `addopts`
  string contains `--cov=bike_rl` (not `--cov=bike_rl.optim` or a narrower
  source that would game the percentage); (2) `--cov-fail-under=80` is
  present; (3) no `bike_rl/`, `tests/`, or `.github/` files are in the
  diff; (4) no `[tool.coverage.run] omit` was added to skip the optim
  stubs.
- **Follow-up explicitly deferred:** `RECREATE_SPEC.md` §11 item 9 — the
  end-to-end smoke test on a tiny bbox — is plan 009, not this plan. This
  plan only makes the static gates green and self-enforcing; it does not
  exercise the live OSM/training/eval pipeline.
