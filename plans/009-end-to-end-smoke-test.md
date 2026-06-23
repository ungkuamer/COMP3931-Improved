# Plan 009: End-to-end headless smoke test (small city + tiny bbox)

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**:
> `git diff --stat 393fe2e..HEAD -- bike_rl/graph_utils.py tests/conftest.py tests/test_graph_utils.py pyproject.toml requirements.txt plans/README.md`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: S
- **Risk**: LOW
- **Depends on**: plans/008-qa-gate-ruff-mypy-cov.md (DONE; gates are green)
- **Category**: tests | bug
- **Planned at**: commit `393fe2e`, 2026-06-23
- **Issue**: _(none — not published via `--issues`)_

## Why this matters

Plans 001–008 are marked DONE, every unit test mocks the OSM / PPO layers, and
CI enforces ≥80% coverage — so the suite is green **without ever running the
real pipeline end-to-end**. RECREATE_SPEC §11 item 9 and the §12 Definition of
Done both require an actual headless run that downloads a real (tiny) network,
trains, evaluates, and writes figure artefacts. Recon for this plan found two
things: the `--city` path already works (Otley cached → ~3m49s, all 6
artefacts, exit 0), but the `--bbox` path is **silently broken** under the
installed `osmnx 2.1.0` because `load_bbox_graph` passes coordinates in the
pre-1.x `(N,S,E,W)` order while `osmnx 2.x` expects `(left, bottom, right, top)`
= `(W,S,E,N)`. The mismatch inflates the Overpass query area ~13,000× and the
run stalls until timeout. Landing this plan gives a repeatable smoke harness
that catches both regressions and fixes the bbox loader so §9 CLI's `--bbox`
branch actually works.

## Current state

Recon ground truth (all verified live at commit `393fe2e` with the project
`.venv` on `osmnx 2.1.0`):

- `bike_rl/graph_utils.py` — graph conversion / OSM loading / caching.
  `load_bbox_graph` (lines ~112–138) currently does, at its return:
  ```python
  def load_bbox_graph(north, south, east, west, cfg, network_type="bike"):
      import osmnx as ox
      configure_osm_cache(cfg)
      return ox.graph_from_bbox((north, south, east, west), network_type=network_type)
  ```
  `osmnx 2.x` signature (printed from the installed package):
  ```
  graph_from_bbox(bbox: tuple[float,float,float,float], *, network_type="all", ...)
      bbox: Bounding box as `(left, bottom, right, top)`. (W, S, E, N)
  ```
  So `(north, south, east, west)` is fed where `(left, bottom, right, top)` is
  expected → N lands in `left` (longitude slot of ~53.9°), widening the query
  to span the whole globe. **This is the bug.**

- `tests/conftest.py` — `mock_osm` fixture's `fake_bbox` records the osmnx
  tuple **as if** it were `(N,S,E,W)`:
  ```python
  def fake_bbox(bbox, network_type="bike", **kw):
      calls["bbox"].append({
          "N": bbox[0], "S": bbox[1], "E": bbox[2], "W": bbox[3],
          "network_type": network_type, **kw,
      })
      ...
  ```
  This encoding *bakes in* the bug, so it must change with the fix.

- `tests/test_graph_utils.py:148–159` — `test_load_bbox_graph_uses_nsew_order`
  asserts the buggy order:
  ```python
  load_bbox_graph(53.9, 53.8, -1.6, -1.7, Config())
  call = mock_osm["bbox"][0]
  assert call["N"] == 53.9   # actually bbox[0]="left"=W after the fix
  assert call["S"] == 53.8
  assert call["E"] == -1.6
  assert call["W"] == -1.7
  ```
  These four assertions become wrong once `load_bbox_graph` passes
  `(W,S,E,N)` to osmnx. The only consumer of `mock_osm["bbox"][*]` keys in the
  whole `tests/` tree is this one test (verified by grep).

- Dependency pin — `requirements.txt:1` and `pyproject.toml:14` both say
  `osmnx>=1.9`, but the codebase already uses the **osmnx 2.x** API
  (`graph_from_bbox((tuple), ...)` and `ox.settings.cache_folder`). 1.x used
  four positional floats. So the realistic lower bound is `osmnx>=2.0`. Bump
  it to match reality.

- The working smoke command (recon ran this; completed in 3m49s, exit 0, no
  network re-download because Otley is already in `osm_cache/`):
  ```
  python -m bike_rl.cli --city "Otley, UK" \
      --budget 200000 --timesteps 2048 --eval-episodes 2 --n-envs 2 --seed 0 \
      --out-dir <OUT>
  ```
  Produced artefacts in the run dir: `final_model.zip`, `run_summary.txt`,
  `training_rewards.png`, `evaluation_metrics.png`, `evaluation_rewards.png`,
  `best_solution_map.png`. (`suggested_bike_paths.geojson` only appears with
  `--export-geojson`.) Note: `--timesteps 2048` with `ppo_n_steps=2048` and
  `n_envs=2` is rounded up by `train_model` to **4096** = exactly one PPO
  update, which is the cheapest real training pass.

- Pipeline config knob `Config.ppo_n_steps = 2048` (`bike_rl/config.py`);

  `train_model` rounds `total_timesteps` up to `ppo_n_steps * n_envs`. Keep
  `--timesteps 2048 --n-envs 2` (→ 4096) for the quick smoke.

**Repo conventions to match** (read these files before editing):
- Error handling / logging: use stdlib `logging.getLogger(__name__)`, never
  `print` outside `cli.py`, never bare `except`. Exemplar: `bike_rl/graph_utils.py`.
- Tests use pytest `monkeypatch`, the `mock_osm` fixture in `tests/conftest.py`,
  and synthetic `tiny_bike_graph` / `tiny_walk_graph` fixtures. Model new bbox
  test on the existing `test_load_bbox_graph_uses_nsew_order`.
- Shell scripts (when added) go under `scripts/` (RECREATE_SPEC §4 layout).
  Be `set -euo pipefail`, `MPLBACKEND=Agg` explicit, and POSIX-ish bash.

## Commands you will need

| Purpose                | Command                                             | Expected on success |
|------------------------|-----------------------------------------------------|---------------------|
| Unit tests (fast, mocked, no network) | `python -m pytest -q tests/test_graph_utils.py` | all pass |
| Full unit suite + gates | `python -m pytest -q`                              | all pass, coverage ≥80% |
| Ruff lint              | `ruff check .`                                       | exit 0 |
| Ruff format check      | `ruff format --check .`                             | exit 0 |
| Mypy strict           | `mypy --strict bike_rl`                             | exit 0 |
| Env check (osmnx 2.x)  | `python -c "import osmnx; print(osmnx.__version__)"` | prints `2.x` |
| City smoke (headless)  | `bash scripts/smoke_e2e.sh --city` (or run the CLI line in Step 4) | exit 0, ≥6 artefacts |
| Bbox smoke (headless)  | `bash scripts/smoke_e2e.sh --bbox`                  | exit 0, ≥6 artefacts |

Use the project venv for everything: `source .venv/bin/activate` (it already
has `osmnx 2.1.0`, `sb3-contrib`, `rustworkx`).

## Scope

**In scope** (the only files you should modify or create):
- `bike_rl/graph_utils.py` — fix `load_bbox_graph` osmnx-2.x bbox order.
- `tests/conftest.py` — update `mock_osm.fake_bbox` to record the real
  `(left, bottom, right, top)` osmnx tuple.
- `tests/test_graph_utils.py` — update `test_load_bbox_graph_uses_nsew_order`
  to assert the `(W,S,E,N)` order, and keep the test docstring truthful.
- `requirements.txt` and `pyproject.toml` — bump `osmnx>=1.9` → `osmnx>=2.0`.
- `scripts/smoke_e2e.sh` — **create** the headless end-to-end smoke harness.
- `plans/README.md` — flip row 009 to DONE.

**Out of scope** (do NOT touch, even though they look related):
- `bike_rl/cli.py`, `bike_rl/training.py`, `bike_rl/evaluation.py`,
  `bike_rl/plotting.py`, `bike_rl/env.py` — these already work end-to-end
  (proven by recon's city run). Do not "tidy" them while in here.
- The `--city` branch of `load_city_graph` — it is correct, leave it.
- `.agents/skills/...` — pre-existing dirty deletions (`cavecrew`,
  `caveman-*`) are unrelated to this plan. Do not stage, restore, or touch them.
- CI workflow `.github/workflows/` — the smoke script is intentionally **off**
  the fast CI path (it spins subprocesses and may need OSM). Do not wire it
  into the gating `pytest` run.
- Cache contents in `osm_cache/` and any output dirs (`bike_path_figures/`,
  `/tmp/bsp_*`, `my_results/`) — these are gitignored; never commit them.

## Git workflow

- Branch: `advisor/009-e2e-smoke-test` (follows the repo's
  `advisor/NNN-<slug>` convention seen in `git log`).
- Commit per logical unit; **conventional commits** (match recent history,
  e.g. `fix(graph_utils): pass (W,S,E,N) to osmnx 2.x graph_from_bbox`,
  `test(graph_utils): assert osmnx-2.x bbox axis order`,
  `feat(scripts): add headless end-to-end smoke harness`,
  `deps: bump osmnx lower bound to 2.0`).
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Verify the environment and the bug

Confirm the installed osmnx is 2.x and reproduce the bbox stall so the fix is
grounded.

**Verify**:
- `source .venv/bin/activate && python -c "import osmnx;print(osmnx.__version__)"` → prints `2.x` (≥2.0). If it prints `<2.0`, treat as a **STOP condition** (the codebase's `graph_from_bbox((tuple), ...)` call already assumes 2.x; report instead of adapting to 1.x).
- (Optional, time-budget ~280 s) Confirm the stall: from the repo root run
  `timeout 60 python -m bike_rl.cli --bbox 53.9065 53.9045 -1.6929 -1.6949 --budget 40000 --timesteps 2048 --eval-episodes 2 --n-envs 2 --out-dir /tmp/bsp_bbox_before`
  → expect a `UserWarning: This area is 13,169 times your configured Overpass max query area size` on stderr and **timeout** (exit 124) / no artefacts. This is the live bug; you do not need it to finish.

### Step 2: Fix the osmnx-2.x bbox axis order in `load_bbox_graph`

In `bike_rl/graph_utils.py`, change the single return line of
`load_bbox_graph` from:
```python
    return ox.graph_from_bbox((north, south, east, west), network_type=network_type)
```
to:
```python
    return ox.graph_from_bbox((west, south, east, north), network_type=network_type)
```
Keep the public signature `(north, south, east, west, cfg, network_type="bike")`
**unchanged** — RECREATE_SPEC §3.1 and §9 CLI expose the `(N,S,E,W)` order to
users; the translation to osmnx's `(left=W, bottom=S, east, top=N)` is an
internal detail. Update the docstring's "Args" to note that the internal call
passes `(W,S,E,N)` to osmnx ≥2.0.

Also bump the dependency lower bound in both `requirements.txt` and
`pyproject.toml` from `osmnx>=1.9` to `osmnx>=2.0` (the code already uses the
2.x API; this just documents reality).

**Verify**:
- `mypy --strict bike_rl` → exit 0 (no new errors; signature unchanged).
- `ruff check . && ruff format --check .` → exit 0.

### Step 3: Update the bbox mock + unit test for the correct order

1. In `tests/conftest.py`, change `fake_bbox` to record the osmnx tuple it was
   actually called with, using osmnx-2.x semantics. Replace the `calls["bbox"].append({...})`
   body's four `N/S/E/W` keys with positional labels for the osmnx order:
   ```python
       def fake_bbox(bbox, network_type="bike", **kw):
           calls["bbox"].append(
               {
                   "osmnx_bbox": tuple(bbox),
                   "network_type": network_type,
                   **kw,
               }
           )
           g = nx.MultiDiGraph()
           g.add_node(1, x=0.0, y=0.0)
           g.add_node(2, x=0.001, y=0.0)
           g.add_edge(1, 2, length=120.0, highway="residential")
           return g
   ```
2. In `tests/test_graph_utils.py`, rewrite
   `test_load_bbox_graph_uses_nsew_order` to assert the osmnx `(W,S,E,N)` order.
   For the call `load_bbox_graph(53.9, 53.8, -1.6, -1.7, Config())`
   (i.e. `N=53.9, S=53.8, E=-1.6, W=-1.7`), osmnx must receive
   `(left=-1.7, bottom=53.8, right=-1.6, top=53.9)`:
   ```python
   def test_load_bbox_graph_uses_nsew_order(mock_osm: _MockCalls) -> None:
       """load_bbox_graph translates (N,S,E,W) → osmnx-2.x (W,S,E,N).

       RECREATE_SPEC §3.1 exposes (N,S,E,W) to users; osmnx ≥2.0 expects
       `(left, bottom, right, top)` = `(W, S, E, N)`.
       """
       load_bbox_graph(53.9, 53.8, -1.6, -1.7, Config())
       assert len(mock_osm["bbox"]) == 1
       call = mock_osm["bbox"][0]
       assert call["osmnx_bbox"] == (-1.7, 53.8, -1.6, 53.9)
       assert call["network_type"] == "bike"
   ```
   Update the test's one-line module docstring note (line 4) only if it still
   reads like the order is `(N,S,E,W)` end-to-end — keep it accurate.

**Verify**:
- `python -m pytest -q tests/test_graph_utils.py` → all pass, **including**
  the rewritten `test_load_bbox_graph_uses_nsew_order`.
- `python -m pytest -q` → all pass, coverage ≥80% (unchanged; `bike_rl/optim/*`
  0% stubs are still counted, don't try to cover them here).
- It is fine if the line numbers of later tests in `tests/test_graph_utils.py`
  shift; the plan does not reference them.

### Step 4: Add the headless end-to-end smoke harness `scripts/smoke_e2e.sh`

Create `scripts/` (it does not exist yet: `mkdir -p scripts`). Write
`scripts/smoke_e2e.sh`:

```bash
#!/usr/bin/env bash
# Headless end-to-end smoke test (RECREATE_SPEC §11 item 9 / §12 DoD).
# Runs the real CLI on a small area, asserts exit 0 and the 6 core artefacts.
set -euo pipefail
export MPLBACKEND=Agg

usage() { echo "usage: $0 [--city|--bbox] [extra cli args...]" >&2; exit 2; }
MODE="${1:-}"
[[ $# -gt 0 ]] && shift
case "$MODE" in
  --city) GEO=(--city "Otley, UK"); LABEL="otley"; BUDGET=200000 ;;
  --bbox) GEO=(--bbox 53.9065 53.9045 -1.6929 -1.6949); LABEL="bbox"; BUDGET=100000 ;;
  *) usage ;;
esac

OUT="$(mktemp -d -t bsp-smoke-XXXXXX)"
trap 'rm -rf "$OUT"' EXIT
PYBIN="${PYTHON:-python}"

echo "[$0] running: $PYBIN -m bike_rl.cli ${GEO[*]} --budget $BUDGET --timesteps 2048 --eval-episodes 2 --n-envs 2 --seed 0 --out-dir $OUT"
"$PYBIN" -m bike_rl.cli "${GEO[@]}" --budget "$BUDGET" \
  --timesteps 2048 --eval-episodes 2 --n-envs 2 --seed 0 --out-dir "$OUT"

# A run dir named <label>_<timestamp>/ sits under --out-dir
RUN_DIR="$(find "$OUT" -maxdepth 2 -mindepth 1 -type d | head -n1)"
echo "[$0] run_dir=$RUN_DIR"
missing=0
for f in final_model.zip run_summary.txt training_rewards.png \
         evaluation_metrics.png evaluation_rewards.png best_solution_map.png; do
  if [[ ! -f "$RUN_DIR/$f" ]]; then
    echo "[$0] MISSING artefact: $f" >&2; missing=1
  fi
done
grep -q '^best_reward:' "$RUN_DIR/run_summary.txt" || { echo "[$0] run_summary missing best_reward" >&2; missing=1; }
test "$missing" -eq 0
echo "[$0] OK: $LABEL smoke passed, 6 artefacts present"
```

Make it executable: `chmod +x scripts/smoke_e2e.sh`.

Why these flags: `--timesteps 2048 --n-envs 2` → `train_model` rounds up to
`ppo_n_steps(2048)*n_envs(2)=4096` = one PPO update (cheapest real pass);
`--eval-episodes 2` keeps evaluation short; `MPLBACKEND=Agg` + no `--show`
keeps it headless (§5.10). The city budget `200000` matches the recon run that
succeeded; the tiny-bbox budget `100000` is plenty for a few-block network.

**Verify**:
- `bash -n scripts/smoke_e2e.sh` → exit 0 (syntax check).
- `ruff`/`mypy` are unaffected (shell script).
- Do **not** add the script to CI gating.

### Step 5: Run both smoke paths and confirm artefacts

Run each; each may take a few minutes and may hit the network the first time
(Otley should be served from `osm_cache/`; the bbox will download a small
graph on first run).

**Verify (city)**:
- `bash scripts/smoke_e2e.sh --city` → exit 0, last line
  `[<path>/smoke_e2e.sh] OK: otley smoke passed, 6 artefacts present`.

**Verify (bbox — confirms the Step 2 fix)**:
- `bash scripts/smoke_e2e.sh --bbox` → exit 0, last line
  `[<path>/smoke_e2e.sh] OK: bbox smoke passed, 6 artefacts present`.
- Crucially, stderr must **not** contain
  `This area is .* times your configured Overpass max query area size`. If it
  does, the bbox axis-order fix is wrong — go back to Step 2 before continuing.

If the bbox run emits a `UserWarning` about an empty/walk network or produces
a run dir whose `run_summary.txt` lists `best_added_edges: 0`, that is
**acceptable** (a few-block area can have very few candidates) as long as all
6 artefacts exist and exit code is 0. Do not tune budgets to force non-zero
edges.

### Step 6: Re-run the fast gates and update the index

**Verify**:
- `ruff check . && ruff format --check .` → exit 0.
- `mypy --strict bike_rl` → exit 0.
- `python -m pytest -q` → all pass, coverage ≥80%.
- `git status --porcelain bike_rl tests scripts requirements.txt pyproject.toml`
  lists exactly: `bike_rl/graph_utils.py`, `tests/conftest.py`,
  `tests/test_graph_utils.py`, `requirements.txt`, `pyproject.toml`,
  `scripts/smoke_e2e.sh` (+ `plans/009-...md` and `plans/README.md`). Nothing
  under `.agents/`.

Then update `plans/README.md`: set the row-009 entry (currently
`009 | end-to-end smoke test on a tiny bbox (depends on 008)`) to a full status
row like the others:
```
| 009  | End-to-end headless smoke test (small city + tiny bbox); fix `load_bbox_graph` osmnx-2.x axis order | P1 | S | 008 | DONE |
```
and add a one-line note under "Dependency notes" recording that 009 is DONE and
that `osmnx>=2.0` is now the realistic lower bound (so a future executor
doesn't try to support 1.x).

## Test plan

- **New/changed unit test**: `tests/test_graph_utils.py::test_load_bbox_graph_uses_nsew_order`
  — now asserts `call["osmnx_bbox"] == (-1.7, 53.8, -1.6, 53.9)` for input
  `(N=53.9, S=53.8, E=-1.6, W=-1.7)`, pinning the osmnx-2.x translation. This
  is the regression guard that runs in CI on every push (fast, mocked, no
  network). Pattern: same test, same fixture, just corrected expectations.
- **New integration harness**: `scripts/smoke_e2e.sh` (off the gating path).
  Covered cases: (a) `--city` end-to-end headless success with 6 artefacts;
  (b) `--bbox` end-to-end headless success with 6 artefacts **and no Overpass
  area-inflation warning** — this is the live confirmation that the §3.1
  bbox branch works, which no unit test exercises end-to-end.
- No new pytest test should spin `SubprocVecEnv` or hit the network — that
  belongs in the script, not the fast suite (keeps CI quick and deterministic).

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `python -c "import osmnx;print(osmnx.__version__)"` prints a `2.x` version.
- [ ] `python -m pytest -q tests/test_graph_utils.py` exits 0 with the rewritten bbox-order assertion.
- [ ] `python -m pytest -q` exits 0 with coverage ≥80% (CI gate unchanged).
- [ ] `ruff check .` and `ruff format --check .` exit 0.
- [ ] `mypy --strict bike_rl` exits 0.
- [ ] `bash scripts/smoke_e2e.sh --city` exits 0 and prints the OK line.
- [ ] `bash scripts/smoke_e2e.sh --bbox` exits 0, prints the OK line, and its
      stderr contains **no** `Overpass max query area size` warning.
- [ ] `grep -n "graph_from_bbox((north" bike_rl/graph_utils.py` returns **no** matches;
      `grep -n "graph_from_bbox((west, south, east, north)" bike_rl/graph_utils.py` returns 1 match.
- [ ] `requirements.txt` and `pyproject.toml` both pin `osmnx>=2.0`.
- [ ] `git status --porcelain` shows no modified files under `.agents/` or `bike_rl/optim/`.
- [ ] `plans/README.md` row 009 reads DONE.

## STOP conditions

Stop and report back (do not improvise) if:

- The installed `osmnx` is **not** 2.x (Step 1 print shows `<2.0`). The fix in
  Step 2 assumes the 2.x `(left, bottom, right, top)` signature; supporting
  1.x is a separate decision — report it instead.
- The "Current state" excerpts don't match the live code (the codebase has
  drifted since this plan was written), e.g. `load_bbox_graph` already passes
  `(W,S,E,N)`, or `fake_bbox` no longer records `N/S/E/W`.
- A step's verification fails twice after a reasonable fix attempt —
  particularly, if `smoke_e2e.sh --bbox` still emits the Overpass
  area-inflation warning, the axis translation is still wrong; stop rather than
  paper over it.
- The bbox smoke returns **zero** artefacts or non-zero exit for a reason other
  than the documented axis-order bug (e.g. a real OSMnx/plotting crash) — that
  is a different defect; report it, do not expand scope to fix it here.
- `pytest` coverage drops below 80% because of the conftest change — the
  change should be isomorphic in coverage; if it drops, re-add captured keys
  rather than removing tests.
- You are tempted to edit `bike_rl/cli.py`, `training.py`, `evaluation.py`,
  `plotting.py`, or `env.py` to make the smoke pass — they are out of scope;
  report instead.

## Maintenance notes

- **What future changes interact with this**: any osmnx upgrade (e.g. to a
  hypothetical 3.x) is the most likely thing to break
  `load_bbox_graph` again — osmnx has already flipped bbox ordering once. The
  `test_load_bbox_graph_uses_nsew_order` guard is the early-warning system;
  keep it. If osmnx reintroduces a 4-positional `graph_from_bbox`, the
  tuple-call will TypeError and the unit test will catch it.
- **CI posture**: `scripts/smoke_e2e.sh` is deliberately **not** in the gating
  suite. If an operator later wants it in CI, gate it behind a manual
  `smoke:` job (needs `actions: [manual]`) and ensure `osm_cache/` is either
  pre-seeded or the job tolerates a network call. Don't make it part of the
  per-push `pytest -q` run — it spins `SubprocVecEnv` and is slow/non-hermetic.
- **Reviewer focus**: (1) the `(W,S,E,N)` translation is the only behavioural
  change in `graph_utils.py` — confirm no other caller of
  `load_bbox_graph` re-orders coordinates (grep says there is none besides
  `cli.py`'s `--bbox` branch, which passes user `(N,S,E,W)` straight through
  and is correct). (2) The conftest `fake_bbox` key change (`N/S/E/W` →
  `osmnx_bbox` tuple) breaks any external reader — confirm via grep that only
  `test_graph_utils.py` reads those keys (it does).
- **Deferred out of this plan**: making the smoke hermetic-by-graph-fixtures
  (replay a pickled graph via `cache_graph` so the script never needs network)
  is a worthwhile follow-up but not required for §11 item 9; tagged as a
  maintenance TODO, not a deliverable here. The `--export-geojson` artefact
  (`suggested_bike_paths.geojson`) is exercised by its own unit tests in plan
  007; the smoke script intentionally omits `--export-geojson` to keep the
  artefact list a stable 6.
- **Note**: plans 001–008 are all marked DONE; this is the final plan in the
  RECREATE_SPEC §11 sequence. After 009 lands, the §12 Definition of Done is
  fully satisfied on both `--city` and `--bbox`.