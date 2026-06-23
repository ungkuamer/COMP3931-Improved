# Plan 007: Implement `cli.py` (RL pipeline end-to-end) + `test_cli.py`

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**: `git diff --stat 4b06659..HEAD -- bike_rl/cli.py bike_rl/training.py bike_rl/config.py bike_rl/run_context.py tests/test_cli.py`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: MED
- **Depends on**: plans/006-evaluation-and-plotting.md (DONE)
- **Category**: dx | tech-debt
- **Planned at**: commit `4b06659`, 2026-06-23
- **Issue**: (omit — not published via `--issues`)

## Why this matters

`bike_rl/cli.py` is currently a stub that parses args and raises
`NotImplementedError("Full CLI in plan 007")` (last line of `main()`). None of
the implemented modules (`graph_utils`, `training`, `evaluation`, `plotting`)
are wired together, so `python -m bike_rl.cli` cannot run a training→evaluation
→artefact pipeline. RECREATE_SPEC §9 and §12 (Definition of Done) require the
CLI to run end-to-end headless, write `final_model.zip`, `training_rewards.png`,
`evaluation_metrics.png`, `evaluation_rewards.png`, the best-solution map,
`suggested_bike_paths.geojson` (opt-in), and `run_summary.txt` into a per-run
directory, and print total runtime as `Hh Mm Ss`. This plan implements that.
The SLURM scripts (RECREATE_SPEC §10, §11 item 7's second half) are
**intentionally out of scope** — the operator has confirmed they will no longer
run on an HPC queue; they are deferred (see "Out of scope").

## Current state

The facts the executor needs, inlined.

### Files and their roles

- `bike_rl/cli.py` — the file to implement (currently a stub). Full current
  contents:
  ```python
  """Command-line entry point for bike-path RL and optimiser experiments.

  See RECREATE_SPEC.md §9.
  """

  from __future__ import annotations

  import argparse


  def main() -> int:
      """Entry point: parse args, delegate to training or optimiser."""
      parser = argparse.ArgumentParser(description="Bike-path expansion RL / optimiser experiments")
      group = parser.add_mutually_exclusive_group(required=True)
      group.add_argument("--city", type=str, help="City name for OSMnx download")
      group.add_argument(
          "--bbox",
          type=float,
          nargs=4,
          metavar=("N", "S", "E", "W"),
          help="Bounding box (N S E W)",
      )
      parser.add_argument("--budget", type=float, default=10000.0)
      parser.add_argument("--timesteps", type=int, default=50000)
      parser.add_argument("--eval-episodes", type=int, default=10)
      parser.add_argument("--n-envs", type=int, default=4)
      parser.add_argument("--device", type=str, default="cpu")
      parser.add_argument("--skip-training", action="store_true")
      parser.add_argument("--model-path", type=str)
      parser.add_argument("--no-plots", action="store_true")
      parser.add_argument("--show", action="store_true")
      parser.add_argument("--export-geojson", action="store_true")
      parser.add_argument("--seed", type=int, default=0)
      parser.add_argument("--config", type=str, help="Path to YAML/TOML config file")
      parser.add_argument("--out-dir", type=str, default="output")
      parser.parse_args()
      raise NotImplementedError("Full CLI in plan 007")
  ```
  Note three spec mismatches to correct: `--budget` default should be `100000.0`
  (RECREATE_SPEC §3.9, not `10000.0`); `--timesteps` default `10240` (§3.9, not
  `50000`); `--eval-episodes` default `5` (§3.9, not `10`); `--out-dir` default
  `"bike_path_figures"` (§9, not `"output"`). `--n-envs` should default to
  `max(1, (os.cpu_count() or 2) - 1)` (§3.7 default is `cpu_count - 1`) —
  compute this at call time, not as a static argparse default.

- `bike_rl/training.py` — `make_vec_env`, `train_model`,
  `TrainingProgressCallback` (all DONE in plan 005). `train_model(envs, cfg,
  run_context, total_timesteps) -> MaskablePPO` currently creates its own
  `TrainingProgressCallback` internally (line 249) and does **not** return it.
  The CLI needs the callback's `.episode_rewards` to write
  `training_rewards.png` (§9). This plan makes one **additive, backward-
  compatible** change: add an optional `progress_callback` parameter so the
  CLI can pass its own instance and read it after training. Current relevant
  excerpt (lines 193–257):
  ```python
  def train_model(
      envs: VecEnv,
      cfg: Config,
      run_context: RunContext,
      total_timesteps: int,
  ) -> MaskablePPO:
      ...
      progress_cb = TrainingProgressCallback(rounded, progress_bar=False)
      ckpt_cb = CheckpointCallback(
          save_freq=save_freq,
          save_path=str(ckpt_dir),
          name_prefix="bike_path_ppo",
          verbose=0,
      )
      model.learn(total_timesteps=rounded, callback=[progress_cb, ckpt_cb])
      return model
  ```

- `bike_rl/config.py` — frozen `Config` dataclass (DONE). The CLI builds a
  `Config` (default or loaded from `--config`) and overrides `seed` and
  `device` from CLI flags via `dataclasses.replace(cfg, seed=args.seed,
  device=args.device)`. `Config` has all PPO hyperparameters and
  `cfg.seed`/`cfg.device` already.

- `bike_rl/run_context.py` — `RunContext.create(base_dir, label, timestamp)`
  → `RunContext(run_id=f"{label}_{ts}", output_dir=base_dir/run_id,
  timestamp=ts)`; `ensure_output_dir()` creates the dir (DONE).

- `bike_rl/graph_utils.py` — `load_city_graph(name, cfg, network_type)`,
  `load_bbox_graph(n, s, e, w, cfg, network_type)`, `configure_osm_cache(cfg)`
  (DONE). The CLI loads the bike graph (`network_type="bike"`) and walk graph
  (`network_type="walk"`) **once in the parent process** (§6.1) before building
  the vec env.

- `bike_rl/evaluation.py` — `evaluate_and_visualize(model, bike_graph,
  walk_graph, cfg, run_context, num_evaluations, budget, seed, show,
  export_geojson, no_plots) -> EvaluationTracker` (DONE). Returns a tracker
  with `.best_reward`, `.best_index`, `.n_episodes`, `.best_added_edges`, and
  per-episode lists. Already writes the map + `evaluation_metrics.png` +
  `evaluation_rewards.png` + GeoJSON into `run_context.output_dir` when
  `no_plots=False`.

- `bike_rl/plotting.py` — `plot_rewards(rewards, run_context, show, filename)`
  (DONE). The CLI calls this with the training callback's `episode_rewards`
  and `filename="training_rewards.png"` to produce the training-rewards plot
  (§9). Headless-safe: `matplotlib.use("Agg")` is set at import.

- `bike_rl/__init__.py` — just `__version__`. No change needed.

### Repo conventions to match (with exemplars)

- **Logging over print** (RECREATE_SPEC §8): every module uses
  `logger = logging.getLogger(__name__)` and logs via `logger.info(...)`. The
  CLI is the **one exception** allowed a final banner print (§8: "Remove all
  `print` calls except the final CLI banner"). Use `logging` for progress;
  reserve `print` for the final summary line and the `Hh Mm Ss` runtime line.
  Exemplar: `bike_rl/training.py:23-24` (`logger = logging.getLogger(__name__)`
  then `logger.info("training total_timesteps=%d rounded=%d ...", ...)`).
- **Type hints + Google docstrings** on every public function. Exemplar:
  `bike_rl/training.py` `make_vec_env` (full arg docstrings, `Args:`/
  `Returns:`). Match that style.
- **No global mutable state** (§8): the CLI creates one `RunContext` in
  `main()` and passes it explicitly to `make_vec_env`/`train_model`/
  `evaluate_and_visualize`. Never set module-level globals.
- **`from __future__ import annotations`** as the first code line in every
  module (all `bike_rl/*.py` do this).
- **Ruff** (`line-length=100`, `select E,F,I,UP,B,SIM,D`, google pydocstyle;
  ignored `D100,D102,D104,D105,D107`). Run `ruff check` and `ruff format
  --check`.
- **Mypy** `python_version="3.12"` (see `pyproject.toml`). Keep the CLI
  mypy-clean.
- **Test style**: `tests/test_training.py` and `tests/test_evaluation.py`
  use `monkeypatch` to stub heavy collaborators (`MaskablePPO.learn`,
  `BikePathEnv.reset`), `tmp_path` for output dirs, a `run_context` fixture,
  and docstrings on every test. Model new `tests/test_cli.py` on
  `tests/test_evaluation.py` (it stubs the model and asserts artefact files).
- **Commit style** (from `git log --oneline`): Conventional Commits,
  e.g. `feat(cli): implement RL pipeline CLI (§9)`, `test(cli): add CLI tests`.
  Subject ≤72 chars.

### Documented vocabulary / design constraints to honor

- RECREATE_SPEC §9 CLI signature (verbatim):
  ```
  python -m bike_rl.cli --city "Otley, UK" --budget 300000 \
      --timesteps 51200 --eval_episodes 10 --n_envs 7 --device cpu \
      [--bbox N S E W] [--skip_training --model_path PATH] \
      [--no_plots] [--show] [--export_geojson] [--out_dir ./bike_path_figures]
  ```
  Plus `--seed` and `--config` (§9 "Behaviours to add"). Argparse dest names
  use hyphens → underscores (`--eval-episodes` → `args.eval_episodes`).
- §9 "Behaviours to preserve": round timesteps up to `n_steps * n_envs` per
  update (already done inside `train_model` — the CLI just passes
  `args.timesteps` through); per-run output dir `<out_dir>/<safe_city>_<ts>/`;
  save `final_model.zip`, `training_rewards.png`, `evaluation_metrics.png`,
  `evaluation_rewards.png`, best-solution map, `run_summary.txt`; print total
  runtime as `Hh Mm Ss`.
- §9 "Behaviours to add": `--show` (opt-in `plt.show()`), `--export_geojson`,
  `--seed` (seed Python, NumPy, torch, SB3, envs), `--config` (YAML/TOML
  `Config` override).
- The CLI dispatches the **RL pipeline only** in this plan. The optimiser
  dispatch (`--compare`, OPTIMIZER_SPEC §11 item 5/6) is a future optimiser
  plan — `bike_rl.optim.*` are still `NotImplementedError` stubs. Do **not**
  wire optimisers here.

## Commands you will need

| Purpose   | Command                              | Expected on success |
|-----------|--------------------------------------|---------------------|
| Install   | `pip install -e ".[dev]"`            | exit 0              |
| Typecheck | `mypy bike_rl/cli.py`                | exit 0, no errors   |
| Lint      | `ruff check bike_rl/cli.py tests/test_cli.py` | exit 0      |
| Format    | `ruff format --check bike_rl/cli.py tests/test_cli.py` | exit 0 |
| Tests     | `pytest tests/test_cli.py -q`        | all pass            |
| Full suite| `pytest -q`                          | all pass            |

(These commands are verified from `pyproject.toml`: ruff config under
`[tool.ruff]`, mypy under `[tool.mypy]`, pytest is the dev test runner.)

## Scope

**In scope** (the only files you should modify or create):
- `bike_rl/cli.py` — full rewrite of `main()` + helpers + `__main__` guard.
- `bike_rl/training.py` — **one additive change only**: add optional
  `progress_callback: TrainingProgressCallback | None = None` parameter to
  `train_model` and use it when provided (see Step 2). No other change to this
  file.
- `tests/test_cli.py` — create.

**Out of scope** (do NOT touch, even though they look related):
- `scripts/run_bike_path_slurm.sh` / `scripts/submit_jobs.sh` (RECREATE_SPEC
  §10) — **SLURM is deferred by operator request**; do not create or modify
  any SLURM scripts. (The `scripts/` directory does not currently exist; do
  not create it.)
- `bike_rl/optim/*` and `bike_rl/objective.py` — still stubs; optimiser CLI
  dispatch (`--compare`) is a future plan. Do not import `bike_rl.optim` from
  `cli.py`.
- `bike_rl/evaluation.py`, `bike_rl/plotting.py`, `bike_rl/graph_utils.py`,
  `bike_rl/env.py`, `bike_rl/metrics.py`, `bike_rl/candidates.py`,
  `bike_rl/config.py`, `bike_rl/run_context.py` — all DONE; do not modify
  (the only exception is `training.py` per Step 2).
- `pyproject.toml` — the `bike-rl = "bike_rl.cli:main"` entry point already
  exists; no change needed. Do not add dependencies (pyyaml is already a dep).
- `tests/conftest.py` — reuse the existing `mock_osm`, `tiny_bike_graph`, and
  `tiny_walk_graph` fixtures; do not modify conftest.

## Git workflow

- Branch: `advisor/007-cli-and-tests`
- Commit per logical unit (training.py tweak, cli.py, tests). Conventional
  Commits, e.g.:
  - `feat(cli): implement RL pipeline CLI (§9)`
  - `refactor(training): accept optional progress_callback for CLI reward plot`
  - `test(cli): add CLI argparse + end-to-end tests`
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Make `train_model` accept an optional `progress_callback`

Edit `bike_rl/training.py` only as follows.

1. Change the `train_model` signature from
   ```python
   def train_model(
       envs: VecEnv,
       cfg: Config,
       run_context: RunContext,
       total_timesteps: int,
   ) -> MaskablePPO:
   ```
   to
   ```python
   def train_model(
       envs: VecEnv,
       cfg: Config,
       run_context: RunContext,
       total_timesteps: int,
       progress_callback: TrainingProgressCallback | None = None,
   ) -> MaskablePPO:
   ```
2. Update the docstring `Args:` block to document the new param:
   ```
   progress_callback: Optional pre-built callback. If given, it is used in
       place of the internally-created one so the caller (the CLI, plan 007)
       can read ``.episode_rewards`` after training to plot
       ``training_rewards.png`` (§9). If None, a fresh callback is created
       internally (existing behaviour, unchanged).
   ```
3. Replace the line
   ```python
       progress_cb = TrainingProgressCallback(rounded, progress_bar=False)
   ```
   with
   ```python
       progress_cb = (
           progress_callback
           if progress_callback is not None
           else TrainingProgressCallback(rounded, progress_bar=False)
       )
   ```
   No other line in the function changes.

This is additive and backward-compatible: existing callers (and
`tests/test_training.py`) that omit the param get identical behaviour.

**Verify**:
- `ruff check bike_rl/training.py` → exit 0
- `mypy bike_rl/training.py` → exit 0
- `pytest tests/test_training.py -q` → all pass (confirms backward compat)

### Step 2: Implement `bike_rl/cli.py`

Rewrite `bike_rl/cli.py` to wire the RL pipeline. Structure (match the
conventions above — `from __future__ import annotations` first, logger, Google
docstrings, type hints):

1. **Imports**: `argparse`, `dataclasses.replace`, `logging`, `os`, `random`,
   `sys`, `time` (for `perf_counter`), `pathlib.Path`, `datetime.datetime`
   /`timezone`. From `bike_rl`: `Config`, `RunContext`. From
   `bike_rl.graph_utils`: `load_city_graph`, `load_bbox_graph`,
   `configure_osm_cache`. From `bike_rl.training`: `make_vec_env`,
   `train_model`, `TrainingProgressCallback`. From `bike_rl.evaluation`:
   `evaluate_and_visualize`. From `bike_rl.plotting`: `plot_rewards`. Import
   `MaskablePPO` lazily inside the functions that need it (so `--help` and
   argparse-error paths don't pay the SB3 import cost and so mypy can see it).
2. **`logger = logging.getLogger(__name__)`**.
3. **`_safe_label(s: str) -> str`**: lowercase, replace any non
   `[a-z0-9]` run with `_`, collapse repeated `_`, strip leading/trailing `_`.
   For `"Otley, UK"` → `"otley_uk"`. For a bbox, the caller passes
   `f"bbox_{n}_{s}_{e}_{w}"` through this too.
4. **`load_config(path: str | None, seed: int, device: str) -> Config`**:
   - Start from `Config()` (defaults).
   - If `path` is not None: detect by extension. `.yaml`/`.yml` →
     `import yaml; yaml.safe_load(open(path))`. `.toml` → `import tomllib`
     (stdlib, Python 3.11+); if `tomllib` import fails on 3.10, raise
     `SystemExit` with a clear message ("TOML config requires Python 3.11+;
     use YAML"). Read `tomllib.load(open(path, "rb"))`. For YAML, the whole
     file is the dict; for TOML, use the `[tool.bike_rl]` table if present
     else the top-level dict. Flatten to a flat `str→value` dict of Config
     field names.
   - Filter the dict to keys that are fields of `Config` (use
     `dataclasses.fields(Config)`); ignore unknown keys with a
     `logger.warning("ignoring unknown config key: %s", k)`.
   - Return `dataclasses.replace(cfg, **overrides, seed=seed, device=device)`.
     `seed` and `device` always come from CLI flags (they win over file).
5. **`_seed_everything(seed: int) -> None`**: seed `random`, `numpy` (`import
   numpy as np; np.random.seed(seed)`), and `torch` (`import torch;
   torch.manual_seed(seed)` — guard with `try/except ImportError` so a
   CPU-only env without torch still works; SB3 depends on torch so this is
   effectively always present, but the guard is defensive). Do not seed SB3
   globally — the `MaskablePPO(seed=cfg.seed)` constructor (already in
   `train_model`) handles that.
6. **`_build_parser() -> argparse.ArgumentParser`**: move the parser
   construction out of `main` into this helper so tests can inspect the
   parser without running `main`. Keep all existing args, but fix the
   defaults per "Current state": `--budget` default `100000.0`,
   `--timesteps` default `10240`, `--eval-episodes` default `5`,
   `--out-dir` default `"bike_path_figures"`. Remove the static `--n-envs`
   default of `4` and instead set `default=None` (Step 7 resolves None to
   `cpu_count-1`). Keep `--city`/`--bbox` mutually exclusive and required.
   Keep `--device` default `"cpu"`, `--seed` default `0`, `--skip-training`,
   `--model-path` (default None), `--no-plots`, `--show`, `--export-geojson`,
   `--config` (default None). Use `dest` names matching the hyphen→underscore
   convention.
7. **`main(argv: list[str] | None = None) -> int`** (accept optional `argv`
   so tests can call `main(["--city", ..., ...])` without `sys.argv`
   monkeypatching):
   - `args = _build_parser().parse_args(argv)`.
   - `cfg = load_config(args.config, seed=args.seed, device=args.device)`.
   - `_seed_everything(cfg.seed)`.
   - Resolve `n_envs = args.n_envs if args.n_envs is not None else max(1,
     (os.cpu_count() or 2) - 1)`.
   - Determine the OSM label: if `args.city` → `label = _safe_label(args.city)`;
     else `label = _safe_label(f"bbox_{args.bbox[0]}_{args.bbox[1]}_
     {args.bbox[2]}_{args.bbox[3]}")`.
   - `run_context = RunContext.create(Path(args.out_dir), label)`;
     `run_context.ensure_output_dir()`.
   - Configure a `logging.StreamHandler` at WARNING level (keep it quiet
     during normal runs; the spec reserves stdout for the final banner). Do
     not add a file handler (SLURM is out of scope).
   - `configure_osm_cache(cfg)`.
   - Load graphs **once** in the parent process (§6.1):
     - If `args.city`: `bike_graph = load_city_graph(args.city, cfg,
       network_type="bike")`; `walk_graph = load_city_graph(args.city, cfg,
       network_type="walk")`.
     - Else: `n, s, e, w = args.bbox`;
       `bike_graph = load_bbox_graph(n, s, e, w, cfg, network_type="bike")`;
       `walk_graph = load_bbox_graph(n, s, e, w, cfg, network_type="walk")`.
   - `start = time.perf_counter()`.
   - **Training path** (not `--skip-training`):
     - `envs = make_vec_env(bike_graph, walk_graph, cfg, run_context,
       n_envs=n_envs, seed=cfg.seed, budget=args.budget,
       vec_env_cls=SubprocVecEnv)`. Import `SubprocVecEnv` from
       `stable_baselines3.common.vec_env` at function top.
     - `progress_cb = TrainingProgressCallback(args.timesteps,
       progress_bar=False)`.
     - `model = train_model(envs, cfg, run_context, args.timesteps,
       progress_callback=progress_cb)`.
     - `model.save(str(run_context.output_dir / "final_model.zip"))`.
     - `envs.close()` (in a `finally` so subprocesses are reaped).
     - If not `args.no_plots` and `progress_cb.episode_rewards`:
       `plot_rewards(progress_cb.episode_rewards, run_context,
       show=args.show, filename="training_rewards.png")`.
   - **Skip-training path** (`--skip-training`):
     - Require `args.model_path` (else `parser.error("--model-path is
       required with --skip-training")` which prints usage and exits 2).
     - `from sb3_contrib import MaskablePPO; model =
       MaskablePPO.load(args.model_path)`.
   - **Evaluation** (both paths): `tracker = evaluate_and_visualize(model,
     bike_graph, walk_graph, cfg, run_context, num_evaluations=args.eval_
     episodes, budget=args.budget, seed=cfg.seed, show=args.show,
     export_geojson=args.export_geojson, no_plots=args.no_plots)`.
   - **`run_summary.txt`**: write to `run_context.output_dir /
     "run_summary.txt"` a plain-text summary including: city/bbox, budget,
     timesteps (or "skipped"), n_envs, eval_episodes, seed, device,
     `tracker.n_episodes`, `tracker.best_reward`, `tracker.best_index`,
     `len(tracker.best_added_edges)`, and the runtime string. Use a small
     helper `_format_runtime(seconds: float) -> str` returning `f"{h}h {m}m
     {s}s"` (zero-pad not required by spec; match §9 "Hh Mm Ss").
   - **Final banner** (the one allowed `print`): print a one-line summary to
     stdout, e.g. `print(f"Run {run_context.run_id} complete in
     {_format_runtime(elapsed)} — best reward {tracker.best_reward:.4f},
     artefacts in {run_context.output_dir}")`.
   - `return 0`.
   - Wrap the body so that `SystemExit`/argparse errors propagate (do not
     catch them). Catch other `Exception` only to log and `return 1` — but
     prefer letting exceptions propagate during development; a single
     top-level `try/except Exception as e: logger.exception(...); return 1`
     is acceptable. Do **not** use a bare `except:` (§5.11).
8. **`__main__` guard** at the bottom:
   ```python
   if __name__ == "__main__":
       sys.exit(main())
   ```
   This makes `python -m bike_rl.cli ...` work (the `bike-rl` console-script
   entry point in `pyproject.toml` already calls `main`).

Target shape of the file: ~160–220 lines, all functions with Google
docstrings, `from __future__ import annotations` first, no `print` outside
`main`.

**Verify**:
- `ruff check bike_rl/cli.py` → exit 0
- `ruff format --check bike_rl/cli.py` → exit 0 (run `ruff format
  bike_rl/cli.py` if it complains, then re-check)
- `mypy bike_rl/cli.py` → exit 0
- `python -m bike_rl.cli --help` → prints usage with all flags, exit 0
- `python -m bike_rl.cli --city "Otley, UK" --bbox 1 2 3 4` → argparse error
  "argument --bbox: not allowed with argument --city", exit 2
- `python -m bike_rl.cli` → argparse error "one of the arguments --city
  --bbox is required", exit 2

### Step 3: Write `tests/test_cli.py`

Create `tests/test_cli.py` modelled on `tests/test_evaluation.py` (same import
style, `from __future__ import annotations`, docstring per test, `tmp_path`,
`monkeypatch`). Cover these cases (all must avoid network access by stubbing
the OSM loaders and the heavy training/eval collaborators):

1. **`test_parser_city_bbox_mutually_exclusive`** — `_build_parser()` (or
   `main`) with `["--city", "X", "--bbox", "1 2 3 4"]` raises
   `SystemExit` (argparse exit code 2). Use `pytest.raises(SystemExit)`.
2. **`test_parser_requires_city_or_bbox`** — `main([])` raises `SystemExit`
   (code 2).
3. **`test_load_config_yaml_overrides`** — write a tiny YAML to `tmp_path`
   with e.g. `budget_efficiency_cap: 99.0` and `w_connectivity: 0.5`; call
   `load_config(str(path), seed=7, device="cpu")`; assert the returned
   `Config` has those fields overridden and `seed == 7`, `device == "cpu"`.
   Also assert an unknown key is ignored (log warning) without error.
4. **`test_safe_label`** — `assert _safe_label("Otley, UK") == "otley_uk"`;
   `assert _safe_label("bbox_1.0_2.0_3.0_4.0") == "bbox_1_0_2_0_3_0_4_0"`
   (dots are non-`[a-z0-9]` → `_`).
5. **`test_main_skip_training_loads_model_and_evaluates`** — the main
   end-to-end test for the skip path:
   - Monkeypatch `bike_rl.cli.load_city_graph` (or the `graph_utils`
     loaders) to return the `tiny_bike_graph` / `tiny_walk_graph` fixtures
     (use `monkeypatch.setattr` on the `bike_rl.cli` module's reference, not
     on `graph_utils`, since `cli.py` imports the names into its own
     namespace).
   - Monkeypatch `MaskablePPO.load` (via `monkeypatch.setattr` on the
     `sb3_contrib.MaskablePPO` class or on `bike_rl.cli`'s usage) to return a
     `_FakeModel` like the one in `tests/test_evaluation.py` (copy that
     class: `predict` returns the first legal action from the mask).
   - Monkeypatch `bike_rl.cli.evaluate_and_visualize` is **not** needed — let
     it run for real on the tiny graphs with `no_plots=True` and
     `num_evaluations=1` so the integration is exercised. (The tiny graphs
     are cheap.)
   - Call `main(["--city", "Tiny", "--skip-training", "--model-path",
     "dummy.zip", "--no-plots", "--out-dir", str(tmp_path), "--budget",
     "100000", "--eval-episodes", "1"])`.
   - Assert return value `== 0`.
   - Assert `run_summary.txt` exists under the created run dir (glob
     `tmp_path / "tiny_*" / "run_summary.txt"`).
   - Assert no `final_model.zip` was written in the skip path (training was
     skipped).
6. **`test_main_training_path_writes_artefacts`** — the main end-to-end test
   for the training path:
   - Monkeypatch the OSM loaders as above to return tiny graphs.
   - Monkeypatch `bike_rl.cli.make_vec_env` to return a `DummyVecEnv` over a
     single `BikePathEnv` built from the tiny graphs (so no subprocesses are
     spawned in tests). Alternatively monkeypatch
     `bike_rl.cli.SubprocVecEnv` to `DummyVecEnv` and `make_vec_env` to pass
     `vec_env_cls=DummyVecEnv` — simplest: monkeypatch
     `bike_rl.cli.SubprocVecEnv` to `DummyVecEnv` (import `DummyVecEnv` from
     `stable_baselines3.common.vec_env`). Confirm `make_vec_env` honors
     `vec_env_cls` (it does — see `training.py`).
   - Monkeypatch `MaskablePPO.learn` to a no-op that returns `self` (so no
     real training runs), and `MaskablePPO.save` to a no-op (or let it write
     — but `save` requires a valid path; a no-op is cleaner and avoids
     writing a zip in tests). Monkeypatch
     `bike_rl.training.MaskablePPO.learn` and `MaskablePPO.save` on the
     class.
   - Call `main(["--city", "Tiny", "--timesteps", "1024", "--n-envs", "1",
     "--no-plots", "--out-dir", str(tmp_path), "--budget", "100000",
     "--eval-episodes", "1"])`.
   - Assert return `== 0`; assert `run_summary.txt` exists; assert
     `final_model.zip` existence is **not** asserted (save is stubbed) —
     instead assert the run dir was created and `run_summary.txt` mentions
     "skipped: False" or the timesteps value.
   - To keep the test deterministic, set `--no-plots` so no PNGs are written
     (the `plot_rewards` training-rewards path is then skipped because
     `no_plots` is true; verify this branch is covered separately — see
     next test).
7. **`test_main_training_writes_training_rewards_png`** — same setup as (6)
   but **without** `--no-plots`, with `learn` stubbed and `MaskablePPO.save`
   stubbed to no-op. Monkeypatch `bike_rl.cli.plot_rewards` to record its
   call args (a spy) instead of writing a real PNG (avoids matplotlib in this
   assertion). Assert `plot_rewards` was called with
   `filename="training_rewards.png"` and a non-empty `rewards` list (seed the
   callback by having `learn` invoke the callback once with an
   `info["episode"]` dict — see "Escape hatches" if your `learn` stub makes
   this awkward). **Simpler alternative** (preferred): assert that when
   `progress_cb.episode_rewards` is empty, `plot_rewards` is **not** called
   (guards the `if progress_cb.episode_rewards:` branch). Cover the non-empty
   branch with a unit test on the guard logic rather than driving a fake
   `learn` — i.e. factor the "should I plot training rewards?" decision into
   a tiny pure helper `_should_plot_training_rewards(no_plots, rewards) ->
   bool` and test it directly. This avoids fragile `learn` stubbing.
8. **`test_main_no_plots_skips_plotting`** — with `--no-plots`, assert no
   `.png` or `.geojson` files appear in the run dir (mirror
   `test_evaluate_no_plots_writes_no_figures` in `tests/test_evaluation.py`).
9. **`test_main_export_geojson_flag`** — `--export-geojson` without
   `--no-plots` results in `suggested_bike_paths.geojson` in the run dir
   (mirror `test_evaluate_export_geojson_writes_file`). Use the skip-training
   path with the `_FakeModel` to keep it cheap.
10. **`test_format_runtime`** — `assert _format_runtime(3661) == "1h 1m 1s"`
    and `_format_runtime(0) == "0h 0m 0s"`.

Reuse the `tiny_bike_graph` / `tiny_walk_graph` fixtures from `conftest.py`
(do not redefine them). Use a local `run_context`-free approach: the CLI
creates its own `RunContext` under `tmp_path`, so tests just pass
`--out-dir str(tmp_path)` and glob for the run dir.

For monkeypatching OSM loaders: `cli.py` will do
`from bike_rl.graph_utils import load_city_graph, load_bbox_graph`, so the
test must patch `bike_rl.cli.load_city_graph` (the name in cli's namespace),
**not** `bike_rl.graph_utils.load_city_graph`. Document this in a comment in
the test file.

**Verify**:
- `ruff check tests/test_cli.py` → exit 0
- `ruff format --check tests/test_cli.py` → exit 0
- `mypy tests/test_cli.py` → exit 0 (or no errors attributed to this file)
- `pytest tests/test_cli.py -q` → all pass

### Step 4: Run the full suite and quality gates

**Verify**:
- `pytest -q` → all pass (no regressions in tests/test_training.py from
  Step 1)
- `ruff check bike_rl/ tests/` → exit 0
- `ruff format --check bike_rl/ tests/` → exit 0
- `mypy bike_rl/` → exit 0

## Test plan

- New tests live in `tests/test_cli.py` (cases 1–10 above).
- Structural pattern: model after `tests/test_evaluation.py` (`_FakeModel`
  class, `monkeypatch` for heavy collaborators, `tmp_path`, docstring per
  test, `from __future__ import annotations`).
- The key regression this plan guards: the CLI runs end-to-end without
  network access (OSM loaders stubbed) and writes `run_summary.txt` plus the
  evaluation artefacts; `--skip-training` loads a model instead of training;
  `--no-plots` skips all figure output; `--export-geojson` writes the
  GeoJSON.
- Coverage note: `cli.py`'s `if __name__ == "__main__":` guard is excluded
  from coverage by convention (RECREATE_SPEC §7 excludes `cli.py`'s
  `__main__`). Do not write a test that execs the guard.
- Verification: `pytest tests/test_cli.py -q` → all pass; `pytest --cov=
  bike_rl --cov-report=term-missing -q` → `cli.py` line coverage ≥80%
  (excluding the `__main__` guard).

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `mypy bike_rl/` exits 0
- [ ] `ruff check bike_rl/ tests/` exits 0
- [ ] `ruff format --check bike_rl/ tests/` exits 0
- [ ] `pytest -q` exits 0; the 10 new `tests/test_cli.py` cases exist and pass
- [ ] `python -m bike_rl.cli --help` exits 0 and lists all §9 flags
- [ ] `python -m bike_rl.cli --city "X" --bbox 1 2 3 4` exits 2 (mutual
  exclusivity)
- [ ] `python -m bike_rl.cli` exits 2 (city/bbox required)
- [ ] `grep -rn "print(" bike_rl/cli.py` returns matches only inside `main`
  (the final banner) — no `print` in helpers
- [ ] `grep -rn "NotImplementedError\|Full CLI in plan 007" bike_rl/cli.py`
  returns no matches
- [ ] `grep -rn "import bike_rl.optim\|from bike_rl.optim" bike_rl/cli.py`
  returns no matches (optimiser dispatch is out of scope)
- [ ] No files outside the in-scope list are modified (`git status` shows only
  `bike_rl/cli.py`, `bike_rl/training.py`, `tests/test_cli.py`)
- [ ] `bike_rl/training.py` diff is exactly the additive `progress_callback`
  change (verify with `git diff bike_rl/training.py` — no other lines moved)
- [ ] No SLURM scripts created (`ls scripts/ 2>/dev/null` returns nothing or
  "No such file")
- [ ] `plans/README.md` status row for 007 updated to DONE (or IN PROGRESS if
  handing off mid-way)

## STOP conditions

Stop and report back (do not improvise) if:

- The code at the locations in "Current state" doesn't match the excerpts
  (e.g. `train_model` signature or line numbers have drifted since
  `4b06659`).
- `make_vec_env` does **not** honor `vec_env_cls` (Step 3 case 6 depends on
  this; `training.py` currently does — verify before relying on it).
- `evaluate_and_visualize`'s signature does not match the one quoted in
  "Current state" (if params were renamed, the CLI call in Step 2 will be
  wrong — report rather than guessing the new names).
- `MaskablePPO.save` / `MaskablePPO.load` are not available on the installed
  `sb3-contrib` version (report the version; do not invent a fallback).
- Adding `progress_callback` to `train_model` breaks any existing
  `tests/test_training.py` test after a reasonable fix attempt (report the
  failing assertion).
- You find that `Config` cannot be constructed via `dataclasses.replace` for
  the fields the CLI needs to override (e.g. a field is not a valid
  `dataclasses` field) — report which field.
- The `pyyaml` import for `--config` YAML loading fails (it is a declared
  dependency, so this should not happen; if it does, report rather than
  adding a dependency).

## Maintenance notes

For whoever owns this code after the change lands:

- **Optimiser dispatch is deferred.** When the optimiser plans
  (OPTIMIZER_SPEC §11) land, `cli.py` will need a `--method` / `--compare`
  subcommand to dispatch `bike_rl.optim.*`. Add it as a new subparser; do not
  bloat `main` with more flags. The current `main` returns `int` and accepts
  `argv` precisely so a subcommand dispatch can wrap it later.
- **SLURM is deferred.** If HPC support is ever revived, add `scripts/`
  per RECREATE_SPEC §10 (install from `requirements.txt`, `python -m
  bike_rl.cli`, `MPLBACKEND=Agg`, `--out_dir $RUN_DIR`). Do not re-add it to
  this plan's scope.
- **Training-rewards plot coupling.** `train_model` now accepts a
  `progress_callback` so the CLI can read `.episode_rewards`. Any future
  change to `TrainingProgressCallback`'s public attributes (`episode_rewards`,
  `episode_lengths`) must keep them as documented lists — the CLI and its
  tests depend on them.
- **`--config` TOML on 3.10.** The CLI raises `SystemExit` for TOML on Python
  3.10 (no `tomllib`). If 3.10 support is dropped (pyproject currently says
  `>=3.10` but mypy targets 3.12), reconsider adding `tomli` as a dep instead
  of the hard exit.
- **Reviewer focus in the PR:** (1) confirm `training.py` diff is *only* the
  additive param; (2) confirm no `bike_rl.optim` import leaked in; (3)
  confirm the CLI loads graphs once in the parent (§6.1) — both bike and walk
  graphs, before `make_vec_env`; (4) confirm `run_summary.txt` is always
  written even on the skip-training path; (5) confirm no bare `except:`
  (§5.11) and no `plt.show()` unless `--show` (§5.10 — `evaluate_and_visualize`
  and `plot_rewards` already honor `show`).
- **Follow-up explicitly deferred:** the §12 "Definition of Done" smoke test
  on a tiny bbox (§11 item 9) and the ≥80% coverage gate (§11 item 8) are
  separate plans (008/009 in `plans/README.md`); this plan only ensures
  `cli.py` itself is ≥80% covered and the unit tests pass headless.
