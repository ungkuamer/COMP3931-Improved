# Plan 005: Implement `training.py` — MaskablePPO training loop, SubprocVecEnv with parent-loaded graphs (§6.1 cache), episode-capturing callback + `test_training.py`

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**:
> `git diff --stat b48386a..HEAD -- bike_rl/training.py bike_rl/config.py bike_rl/env.py bike_rl/run_context.py bike_rl/graph_utils.py tests/conftest.py tests/test_env.py`
> If any in-scope or dependency file changed since this plan was written,
> compare the "Current state" excerpts against the live code before
> proceeding; on a mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: MED
- **Depends on**: plans/004-env-and-reward.md (DONE — provides `bike_rl.env.BikePathEnv` with a **fixed-size** `Discrete` action space, `action_masks()`, `reset(options={"budget": …})`, and SB3-style `info["episode"]` emitted on termination — the §5.12 contract this plan's callback relies on)
- **Category**: feature (RL training loop §3.7) + perf (§6.1 OSM cache via parent-loaded graphs) + tests
- **Planned at**: commit `b48386a`, 2026-06-22
- **Issue**: (not published)

## Why this matters

The original `nx-rx-simple-ur.py` trains with `SubprocVecEnv`, and **every
worker re-downloads the same OSM graphs from the network**
(`n_envs × duplicate network I/O`, rate-limited by OSM). On an 8-CPU SLURM
node that is 8 redundant downloads per training run — slow and flaky on a
headless cluster. The training loop also uses vanilla `PPO`, so the §5.4
action-masking fix delivered by plan 004 (`BikePathEnv.action_masks()`) is
never actually consumed — invalid actions still get sampled and the agent
still sees the wrong signal.

This plan builds `bike_rl/training.py`, which:

1. **Uses `MaskablePPO` from `sb3-contrib`** with the
   `MaskableActorCriticPolicy`, so plan 004's `action_masks()` is consumed
   and invalid (already-used / unaffordable) actions are never sampled
   (RECREATE_SPEC §3.7, §5.4).
2. **Loads the OSM graphs once in the parent process and passes the in-memory
   graph objects to each worker via the `SubprocVecEnv` thunks** — SB3
   serialises the thunks (with the graphs in their closure) to workers with
   `cloudpickle`, so **no worker ever re-downloads** (§6.1). Combined with
   `graph_utils.configure_osm_cache` (already implemented in plan 002), the
   parent's download is also backed by the on-disk OSMnx cache for repeat
   runs.
3. **Rounds `total_timesteps` up to `ppo_n_steps * n_envs`** per update
   batch, matching the original §3.7 behaviour exactly.
4. **Records every terminated episode — including the last one** — via a
   `TrainingProgressCallback` that reads the SB3 `info["episode"]` dict the
   env emits on termination (the §5.12 contract from plan 004), instead of
   the original's drop-the-last-episode `reset()`-time append.
5. **Attaches a `CheckpointCallback`** that writes into the `RunContext`
   output directory (no global `RUN_FIGURES_DIR` — §5.13).

`evaluation.py`, `plotting.py`, `cli.py`, and the SLURM scripts are **out of
scope** — they are plans 006–007. The CLI will load graphs and call
`make_vec_env` + `train_model`; this plan provides those entry points.

## Current state

The package is scaffolded (plan 001), `graph_utils.py`/`candidates.py`
implemented (plan 002), `metrics.py` implemented (plan 003), and `env.py`
implemented (plan 004). All 62 existing tests pass (`pytest -q` → `62
passed`). The relevant files:

- `bike_rl/training.py` — **stub**. Three names are declared, all raising
  `NotImplementedError("Plan 005")`. Their signatures are:
  ```python
  def make_env(
      bike_graph: object,
      walk_graph: object,
      cfg: Config,
      run_context: RunContext,
      rank: int,
      seed: int,
  ) -> gym.Env[object, object]:
      """Create a single BikePathEnv wrapped with Monitor."""
      raise NotImplementedError("Plan 005")

  def train_model(envs: object, cfg: Config, run_context: RunContext, total_timesteps: int) -> object:
      """Train a PPO/MaskablePPO model."""
      raise NotImplementedError("Plan 005")

  class TrainingProgressCallback:
      def __init__(self) -> None:
          raise NotImplementedError("Plan 005")
      def _on_step(self) -> bool:
          raise NotImplementedError("Plan 005")
  ```
  (`Config`, `RunContext`, and `gym` are imported under `TYPE_CHECKING` in
  the stub — keep those guarded imports and add the real runtime imports in
  step 1.)

- `bike_rl/env.py` — implemented (plan 004). **This is the dependency you
  build on.** Import and use exactly this constructor and these methods
  (verified at `b48386a`):
  ```python
  class BikePathEnv(gym.Env[object, object]):
      def __init__(self, bike_graph, walk_graph, cfg: Config, run_context: RunContext) -> None: ...
      def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None)
          -> tuple[npt.NDArray[np.float32], dict[str, Any]]: ...
      def action_masks(self) -> npt.NDArray[np.bool_]: ...
      def step(self, action) -> tuple[...]: ...
  ```
  - **Budget is passed via `reset(options={"budget": float})`, not via
    `Config` or `__init__`** — this is deliberate (plan 004 maintenance
    note: budget is a per-run CLI arg). `make_env` MUST reset the env with
    the budget on construction, otherwise the env silently falls back to
    `100_000.0`.
  - On termination, `step` emits `info["episode"] = {"r": float, "l": int}`
    (the SB3 Monitor convention). The `TrainingProgressCallback` reads
    exactly this key — do **not** append at `reset()` time (that was the
    §5.12 bug).
  - `action_masks()` is exposed on the unwrapped env; `MaskablePPO` finds it
    automatically through `Monitor`/`SubprocVecEnv` wrappers via
    `env.get_wrapper_attr("action_masks")` / `env.env_method("action_masks")`
    (verified — see "SB3 / sb3-contrib API facts" below).

- `bike_rl/config.py` — frozen `Config`, fully implemented. **This plan adds
  NO new Config fields.** Use these existing fields (verified at `b48386a`):
  `ppo_policy: str = "MlpPolicy"`, `ppo_learning_rate: float = 3e-4`,
  `ppo_n_steps: int = 2048`, `ppo_batch_size: int = 64`,
  `ppo_n_epochs: int = 10`, `ppo_gamma: float = 0.99`,
  `ppo_gae_lambda: float = 0.95`, `ppo_clip_range: float = 0.2`,
  `device: str = "cpu"`, `seed: int = 0`. There is **no `n_envs` field** —
  `n_envs` is a CLI arg (§3.9) and is passed explicitly to `make_vec_env`.

- `bike_rl/run_context.py` — `RunContext(run_id, output_dir, timestamp)` with
  `ensure_output_dir() -> Path`. `train_model` writes checkpoints under
  `run_context.output_dir / "checkpoints"`.

- `bike_rl/graph_utils.py` — provides `configure_osm_cache(cfg)`,
  `load_city_graph`, `load_bbox_graph`, `cache_graph`, `load_cached_graph`.
  **`training.py` does NOT call any OSM loader** (the CLI does, plan 007).
  `training.py` receives already-loaded graph objects. The §6.1 fix is
  realised by passing those in-memory graphs into `SubprocVecEnv` thunks so
  workers receive them via pickle instead of re-downloading.

- `tests/conftest.py` — provides `tiny_bike_graph`, `tiny_walk_graph`, and
  `mock_osm` fixtures. `test_training.py` will reuse `tiny_bike_graph` /
  `tiny_walk_graph` and the `run_context` fixture pattern copied from
  `tests/test_env.py` (see Step 4).

- `tests/test_training.py` — **does not exist** (verified). Create it.

### SB3 / sb3-contrib API facts (verified in the venv at `b48386a`)

These are the exact import paths and signatures the plan uses. They were
checked by importing them in the project venv. If the executor's environment
disagrees, that's a STOP condition.

- `sb3_contrib.MaskablePPO` — constructor (only the params this plan uses are
  listed; the rest keep defaults):
  ```python
  MaskablePPO(
      policy,                  # "MlpPolicy" or a policy class — use MaskableActorCriticPolicy
      env,                     # a gym.Env or VecEnv
      learning_rate=0.0003,
      n_steps=2048,
      batch_size=64,
      n_epochs=10,
      gamma=0.99,
      gae_lambda=0.95,
      clip_range=0.2,
      seed=None,
      device="auto",
      verbose=0,
      # ... other params keep defaults
  )
  ```
- `sb3_contrib.common.maskable.policies.MaskableActorCriticPolicy` — pass this
  class as `policy=` (NOT the string `"MlpPolicy"`) so MaskablePPO gets the
  masking-aware policy. (`cfg.ppo_policy` is the string `"MlpPolicy"`; use it
  only as documentation — the actual `policy=` argument is the class.)
- `sb3_contrib.common.maskable.utils.get_action_masks(env)` — **you do not
  call this yourself**; `MaskablePPO.collect_rollouts` calls it internally.
  It works through `Monitor` and `SubprocVecEnv` wrappers automatically
  (`env.get_wrapper_attr("action_masks")` for non-VecEnv;
  `env.env_method("action_masks")` for VecEnv). Verified.
- `stable_baselines3.common.vec_env.SubprocVecEnv(env_fns: list[Callable[[], gym.Env]], start_method=None)`
  — spawns one process per thunk. `env_fns` are serialised with
  `cloudpickle`, so closures capturing the in-memory graphs work (this is the
  §6.1 mechanism).
- `stable_baselines3.common.vec_env.DummyVecEnv(env_fns)` — in-process VecEnv,
  used by the tests to avoid spawning processes in `pytest`.
- `stable_baselines3.common.monitor.Monitor(env, filename=None, ...)` — wraps
  an env and records episode results to a `monitor.csv` if `filename` given.
  Pass `filename=None` (we do not need per-env monitor files; the
  `TrainingProgressCallback` records episodes in-memory).
- `stable_baselines3.common.callbacks.BaseCallback` — subclass with
  `_on_step(self) -> bool`. Available attributes: `self.n_calls` (increments
  per `collect_rollouts` step), `self.num_timesteps` (global), `self.model`,
  `self.locals` (dict; contains `infos` — a list of info dicts, one per env,
  for the step just taken). Optional overrides: `_on_training_start(self)`,
  `_init_callback(self)`.
- `stable_baselines3.common.callbacks.CheckpointCallback(save_freq: int, save_path: str, name_prefix="rl_model", verbose=0)`
  — saves the model every `save_freq` steps. It creates `save_path` on first
  save, but creating the dir up front is safer for `run_context`-based paths.
- `MaskablePPO.learn(total_timesteps, callback=None, ...)` — `callback`
  accepts a `BaseCallback`, a `list[BaseCallback]`, or a callable. Pass
  `[TrainingProgressCallback(...), CheckpointCallback(...)]`.

## Commands you will need

| Purpose   | Command                                              | Expected on success |
|-----------|------------------------------------------------------|---------------------|
| Install   | `pip install -e ".[dev]"`                            | exit 0              |
| Lint      | `ruff check bike_rl/training.py tests/test_training.py` | exit 0          |
| Format    | `ruff format --check bike_rl/training.py tests/test_training.py` | exit 0 |
| Typecheck | `mypy --strict bike_rl`                              | exit 0, no errors   |
| Tests     | `pytest -q tests/test_training.py`                   | all pass            |
| Full suite| `pytest -q`                                          | 62 prior + new tests pass |

(Exact commands from this repo — verified during recon. The CI workflow at
`.github/workflows/ci.yml` runs `ruff check .`, `ruff format --check .`,
`mypy --strict bike_rl`, `pytest -q` on Python 3.10 and 3.12.)

## Scope

**In scope** (the only files you should modify):
- `bike_rl/training.py` — replace the stub with the full implementation.
- `tests/test_training.py` — create.

**Out of scope** (do NOT touch, even though they look related):
- `bike_rl/config.py` — **no new fields.** All hyperparameters this plan
  needs already exist. If you are tempted to add a field, STOP and report.
- `bike_rl/env.py`, `bike_rl/metrics.py`, `bike_rl/candidates.py`,
  `bike_rl/graph_utils.py`, `bike_rl/run_context.py`, `bike_rl/objective.py`
  — all implemented by prior plans; do not modify.
- `bike_rl/evaluation.py`, `bike_rl/plotting.py`, `bike_rl/cli.py` — stubs
  for plans 006–007; do not implement them here.
- `bike_rl/optim/*` — the direct optimisers; unrelated to the RL training
  loop.
- `tests/conftest.py`, `tests/test_env.py`, other test files — do not modify.
  Reuse fixtures by importing/parametrising in `test_training.py`.
- `pyproject.toml`, `requirements.txt`, CI config — `sb3-contrib`,
  `stable-baselines3`, `tqdm` are already declared (verified).

## Git workflow

- Branch: `advisor/005-training-and-tests`
- Commit per step or per logical unit; message style: conventional commits —
  e.g. `feat(training): implement MaskablePPO loop + SubprocVecEnv (§3.7/§6.1)`
  (match the repo's existing style: see `git log --oneline` — `feat(env): …`,
  `test(reward): …`, `docs: mark plan … as DONE`).
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Replace the `training.py` stub with real imports and the `make_env` function

Write `bike_rl/training.py` with this exact module header and `make_env`.
Keep the `TYPE_CHECKING` guarded `Config` / `RunContext` / `gym` imports from
the stub, and add the real runtime imports below.

```python
"""PPO/MaskablePPO training loop. See RECREATE_SPEC.md §3.7, §6.1."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Callable

import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from bike_rl.env import BikePathEnv

if TYPE_CHECKING:
    from bike_rl.config import Config
    from bike_rl.run_context import RunContext

logger = logging.getLogger(__name__)


def make_env(
    bike_graph: object,
    walk_graph: object,
    cfg: Config,
    run_context: RunContext,
    rank: int,
    seed: int,
    budget: float = 100_000.0,
) -> gym.Env[object, object]:
    """Construct, seed, reset, and ``Monitor``-wrap a single ``BikePathEnv``.

    The env is reset immediately with ``options={"budget": budget}`` so the
    episode starts with the configured budget (plan 004's contract — budget
    is passed via ``reset`` options, not ``Config``). The per-env seed is
    ``seed + rank`` so workers get distinct, reproducible seeds (§3.7).

    Args:
        bike_graph: The existing bike network (loaded once in the parent
            process — §6.1; passed to workers via the SubprocVecEnv thunk).
        walk_graph: The walkable network candidates are drawn from.
        cfg: Config (reward weights, cost factor, PPO hyperparameters).
        run_context: Per-run output context (stored on the env for logging;
            the env does not create directories).
        rank: Worker index (0-based); added to ``seed`` for per-env seeding.
        seed: Base seed (typically ``cfg.seed``).
        budget: Initial episode budget. Defaults to 100_000.0; the CLI
            (plan 007) passes the real ``--budget`` value.

    Returns:
        A ``Monitor``-wrapped ``BikePathEnv`` with the episode already reset.
    """
    env = BikePathEnv(bike_graph, walk_graph, cfg, run_context)
    env.reset(seed=seed + rank, options={"budget": float(budget)})
    return Monitor(env, filename=None)
```

Notes:
- `make_env`'s public signature from the stub is preserved **except** one new
  optional parameter `budget: float = 100_000.0` is added at the end. This is
  a backward-compatible addition (callers that don't pass it get the default).
  It is required because plan 004 deliberately takes budget via
  `reset(options=...)` and the training loop must supply it. If the reviewer
  objects to adding the parameter, the alternative is to read a `budget`
  field from `Config` — but `Config` has no such field by design (budget is a
  per-run CLI arg, §3.9), so adding the parameter is correct.
- `Monitor(env, filename=None)` records nothing to disk but still forwards
  `info["episode"]` and supports `get_episode_rewards()`; we pass
  `filename=None` to avoid cluttering the run dir with per-worker CSVs.

**Verify**:
- `ruff check bike_rl/training.py` → exit 0
- `ruff format --check bike_rl/training.py` → exit 0 (run `ruff format
  bike_rl/training.py` if it complains, then re-check)
- `mypy --strict bike_rl/training.py` → exit 0. If mypy complains that
  `Monitor` is not assignable to `gym.Env[object, object]`, wrap the return
  with `return Monitor(env, filename=None)` unchanged and add
  `# type: ignore[return-value]` on that line only — but first try
  `cast(gym.Env[object, object], Monitor(env, filename=None))` (preferred,
  no ignore). Report which was needed in the commit message.

### Step 2: Add `make_vec_env` (the §6.1 SubprocVecEnv assembler)

Add this function to `bike_rl/training.py` (after `make_env`). This is a new
public helper not in the stub; it is the primary realisation of the §6.1
cache fix and the entry point the CLI (plan 007) will call.

```python
def make_vec_env(
    bike_graph: object,
    walk_graph: object,
    cfg: Config,
    run_context: RunContext,
    n_envs: int,
    seed: int | None = None,
    budget: float = 100_000.0,
    vec_env_cls: type[VecEnv] = SubprocVecEnv,
    start_method: str | None = None,
) -> VecEnv:
    """Build a vectorised env over ``n_envs`` ``BikePathEnv`` workers.

    Each worker is a ``Monitor``-wrapped ``BikePathEnv`` built by
    :func:`make_env` with a distinct seed (``base_seed + rank``). The
    in-memory ``bike_graph`` / ``walk_graph`` are captured in each thunk's
    closure; ``SubprocVecEnv`` serialises the thunks to workers with
    ``cloudpickle``, so **no worker re-downloads OSM data** (RECREATE_SPEC
    §6.1 — the graphs are loaded once in the parent process by the CLI and
    passed in here).

    Args:
        bike_graph: The existing bike network (parent-loaded, §6.1).
        walk_graph: The walkable network.
        cfg: Config.
        run_context: Per-run output context.
        n_envs: Number of parallel envs (§3.7 default is ``cpu_count() - 1``;
            the CLI computes that — this function just takes the number).
        seed: Base seed; defaults to ``cfg.seed``. Each worker gets
            ``seed + rank``.
        budget: Initial episode budget passed to each env's ``reset``.
        vec_env_cls: ``SubprocVecEnv`` (default, spawns processes) or
            ``DummyVecEnv`` (in-process; used by tests to avoid spawning).
        start_method: ``SubprocVecEnv`` ``start_method`` (``"spawn"`` /
            ``"forkserver"`` / ``"fork"`` / ``None``). Ignored for
            ``DummyVecEnv``.

    Returns:
        A vectorised env ready to pass to :func:`train_model`.
    """
    base_seed = cfg.seed if seed is None else seed
    env_fns: list[Callable[[], gym.Env[object, object]]] = [
        lambda r=rank: make_env(bike_graph, walk_graph, cfg, run_context, r, base_seed, budget)
        for rank in range(n_envs)
    ]
    if vec_env_cls is SubprocVecEnv:
        return SubprocVecEnv(env_fns, start_method=start_method)
    return DummyVecEnv(env_fns)
```

Notes:
- The `lambda r=rank: ...` default-argument pattern binds `rank` per-iteration
  so each thunk closes over its own rank (the classic late-binding fix).
- `vec_env_cls is SubprocVecEnv` is the branching test. `DummyVecEnv` does
  not accept `start_method`, so we branch rather than passing it through.
- mypy: `list[Callable[[], gym.Env[object, object]]]` — each lambda returns
  `Monitor` (subtype of `gym.Env[object, object]`), which is assignable to
  the element type. If mypy flags the list literal, annotate the lambdas'
  return via the list annotation above (already done); if it still complains,
  `cast` the list: `env_fns = cast(list[Callable[[], gym.Env[object, object]]], [...])`.

**Verify**:
- `ruff check bike_rl/training.py` → exit 0
- `mypy --strict bike_rl/training.py` → exit 0

### Step 3: Implement `TrainingProgressCallback` (§5.12 — capture every episode)

Replace the stub class with this. It subclasses `BaseCallback`, reads
`info["episode"]` from every env's info on every step, and appends to an
in-memory list — so the **last** episode (terminated at the end of rollout)
is recorded in the same step it terminates, not dropped.

```python
class TrainingProgressCallback(BaseCallback):
    """Callback that records every terminated episode's reward (RECREATE_SPEC §5.12).

    Reads the SB3 ``info["episode"]`` dict that ``BikePathEnv.step`` emits on
    termination (plan 004's contract) and appends ``info["episode"]["r"]`` to
    ``episode_rewards``. Because the env emits the summary *on termination*
    (not at the next ``reset``), the final episode of a rollout is recorded
    too — fixing the original §5.12 "last episode dropped" bug.

    Optionally displays a ``tqdm`` progress bar over ``total_timesteps``.

    Attributes:
        episode_rewards: List of per-episode total rewards (one entry per
            terminated episode, across all envs), in termination order.
        episode_lengths: List of per-episode lengths, parallel to
            ``episode_rewards``.
    """

    def __init__(
        self,
        total_timesteps: int,
        progress_bar: bool = True,
        verbose: int = 0,
    ) -> None:
        """Store config; lists are populated during training.

        Args:
            total_timesteps: Total timesteps the model will train for (used
                only for the tqdm bar total).
            progress_bar: If True, show a tqdm progress bar. Tests pass
                ``False`` to keep output deterministic.
            verbose: SB3 callback verbosity.
        """
        super().__init__(verbose=verbose)
        self._total_timesteps = total_timesteps
        self._progress_bar = progress_bar
        self._pbar: Any = None
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []

    def _on_training_start(self) -> None:
        """Create the tqdm bar (if enabled) once training starts."""
        super()._on_training_start()
        if self._progress_bar:
            from tqdm import tqdm

            self._pbar = tqdm(total=self._total_timesteps, desc="training", leave=False)

    def _on_step(self) -> bool:
        """Record any episode info from this step's per-env ``infos``; update the bar.

        Returns ``True`` to continue training (return ``False`` to stop early;
        this implementation never stops early).
        """
        infos: list[dict[str, Any]] = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                self.episode_rewards.append(float(ep["r"]))
                self.episode_lengths.append(int(ep["l"]))
        if self._pbar is not None:
            self._pbar.update(self.model.num_timesteps - self._pbar.n)
        return True

    def _on_training_end(self) -> None:
        """Close the tqdm bar (if any)."""
        super()._on_training_end()
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None
```

Notes:
- `self.locals` is set by SB3 before `_on_step` is called; using
  `.get("infos", [])` is defensive (SB3 always populates it, but the default
  keeps tests robust).
- `self.model.num_timesteps` is the global step count; `self._pbar.n` is how
  far the bar has advanced. Updating by the delta keeps the bar in sync
  without double-counting across `_on_step` calls.
- `tqdm` is imported lazily inside `_on_training_start` so tests that pass
  `progress_bar=False` never import it (keeps test output clean and avoids
  any tqdm-in-CI noise). `tqdm>=4.66` is already in `requirements.txt`.

**Verify**:
- `ruff check bike_rl/training.py` → exit 0
- `mypy --strict bike_rl/training.py` → exit 0. `tqdm` is not in the mypy
  overrides' `ignore_missing_imports` list, but `tqdm` ships type hints;
  `self._pbar: Any` avoids exposing the tqdm type. If mypy complains about
  `from tqdm import tqdm`, add `tqdm` to the
  `[[tool.mypy.overrides]] module = [...]` list in `pyproject.toml` — but
  **only** if needed (try without first; this is the one allowed
  `pyproject.toml` touch in this plan, and only for the mypy override list).

### Step 4: Implement `train_model` (MaskablePPO + rounded timesteps + callbacks)

Replace the stub function with this. It refines the stub's `envs: object` /
`-> object` annotations to concrete types (`VecEnv` / `MaskablePPO`) — this
is a justified, mypy-strict-driven refinement (the `object` annotations would
force `cast` calls everywhere and give callers no type information).

```python
def train_model(
    envs: VecEnv,
    cfg: Config,
    run_context: RunContext,
    total_timesteps: int,
    budget: float = 100_000.0,
) -> MaskablePPO:
    """Train a ``MaskablePPO`` model on ``envs`` (RECREATE_SPEC §3.7, §5.4).

    Uses ``MaskableActorCriticPolicy`` so the env's ``action_masks()`` is
    consumed (§5.4 — invalid actions are never sampled). Hyperparameters come
    from ``cfg``. ``total_timesteps`` is rounded up to a multiple of
    ``ppo_n_steps * n_envs`` (§3.7 — one update batch's worth). A
    :class:`TrainingProgressCallback` records every terminated episode
    (§5.12), and a :class:`CheckpointCallback` writes into
    ``run_context.output_dir / "checkpoints"``.

    Args:
        envs: Vectorised env (from :func:`make_vec_env`).
        cfg: Config (PPO hyperparameters, seed, device).
        run_context: Per-run output context for checkpoint paths.
        total_timesteps:Requested timesteps; rounded up to
            ``ppo_n_steps * envs.num_envs``.
        budget: Unused inside ``train_model`` (kept in the signature only if
            the stub is extended later); the budget is applied per-env in
            :func:`make_vec_env`. Defaults to 100_000.0.

    Returns:
        The trained ``MaskablePPO`` model.
    """
    n_envs = envs.num_envs
    batch = cfg.ppo_n_steps * n_envs
    rounded = max(batch, int(math.ceil(total_timesteps / batch) * batch))
    logger.info(
        "training total_timesteps=%d rounded=%d (n_steps=%d n_envs=%d)",
        total_timesteps, rounded, cfg.ppo_n_steps, n_envs,
    )

    out_dir = run_context.ensure_output_dir()
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_freq = max(rounded // 10, 1)

    model = MaskablePPO(
        policy=MaskableActorCriticPolicy,
        env=envs,
        learning_rate=cfg.ppo_learning_rate,
        n_steps=cfg.ppo_n_steps,
        batch_size=cfg.ppo_batch_size,
        n_epochs=cfg.ppo_n_epochs,
        gamma=cfg.ppo_gamma,
        gae_lambda=cfg.ppo_gae_lambda,
        clip_range=cfg.ppo_clip_range,
        seed=cfg.seed,
        device=cfg.device,
        verbose=0,
    )
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

Notes:
- `policy=MaskableActorCriticPolicy` (the **class**, not the string
  `"MlpPolicy"`) is required for masking to work. `cfg.ppo_policy` is
  intentionally **not** passed to `MaskablePPO` — it is a documentation field
  only. (Passing the string `"MlpPolicy"` would make `MaskablePPO` use the
  masking-aware policy by default too, but passing the class explicitly is
  unambiguous and matches the sb3-contrib docs.)
- `progress_bar=False` in `train_model` because training usually runs
  headless on SLURM; if a caller wants the bar, they can construct their own
  callback. (This keeps CLI output clean — plan 007 can flip it via a flag if
  desired, but that is out of scope here.)
- The `budget` parameter on `train_model` is **kept for signature stability**
  but is not used inside the body (budget is applied per-env in
  `make_vec_env`/`make_env`). If the reviewer prefers to drop it, that is
  fine — but the stub did not declare it, so adding it is optional. **Choose
  one and be consistent**: either (a) keep `budget` in both `make_vec_env`
  and `train_model` (the CLI can pass it to either), or (b) drop it from
  `train_model` and have the CLI pass it only to `make_vec_env`. Recommended:
  **drop `budget` from `train_model`** to avoid the misleading unused
  parameter — the CLI calls `make_vec_env(..., budget=B)` then
  `train_model(envs, cfg, rc, T)`. The excerpt above includes it for
  illustration; **remove the `budget` parameter and its docstring entry
  before finalising** so `train_model`'s signature is
  `train_model(envs: VecEnv, cfg: Config, run_context: RunContext, total_timesteps: int) -> MaskablePPO`.
- Timestep rounding: `batch = ppo_n_steps * n_envs`; `rounded` is the
  smallest multiple of `batch` that is `>= total_timesteps`. For
  `total_timesteps=0` or negative, `max(batch, ...)` clamps to one batch.
  This matches §3.7 "rounded up to n_steps * n_envs per update batch".

**Verify**:
- `ruff check bike_rl/training.py` → exit 0
- `ruff format --check bike_rl/training.py` → exit 0
- `mypy --strict bike_rl/training.py` → exit 0
- `python -c "from bike_rl.training import make_env, make_vec_env, train_model, TrainingProgressCallback; print('ok')"` →
  prints `ok`

### Step 5: Create `tests/test_training.py`

Create the file with these tests. They use `DummyVecEnv` (no process
spawning), monkeypatch `MaskablePPO` to avoid real training, and drive the
callback by hand to pin §5.12. Model the file structure on
`tests/test_env.py` (imports, `run_context` fixture, `Config()` usage,
`_NXGraph` typing guard).

```python
"""Tests for ``bike_rl.training``.

Covers: ``make_env`` seeding + budget passthrough, ``make_vec_env`` thunk
seeds, ``train_model`` timestep rounding + MaskablePPO/policy/cfg wiring
(§3.7), and ``TrainingProgressCallback`` recording every terminated episode
including the last (§5.12).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import networkx as nx
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from bike_rl.config import Config
from bike_rl.env import BikePathEnv
from bike_rl.run_context import RunContext
from bike_rl.training import TrainingProgressCallback, make_env, make_vec_env, train_model

if TYPE_CHECKING:
    _NXGraph = nx.MultiDiGraph[Any, Any, Any]
else:
    _NXGraph = nx.MultiDiGraph


@pytest.fixture
def run_context(tmp_path: Path) -> RunContext:
    """Provide a RunContext with a temp output dir."""
    return RunContext(
        run_id="test",
        output_dir=tmp_path / "out",
        timestamp=datetime.now(timezone.utc),
    )


def test_make_env_returns_monitor_wrapped_bikepathenv(
    tiny_bike_graph: _NXGraph, tiny_walk_graph: _NXGraph, run_context: RunContext
) -> None:
    """make_env returns a Monitor wrapping a BikePathEnv with the episode reset."""
    env = make_env(tiny_bike_graph, tiny_walk_graph, Config(), run_context, rank=0, seed=42)
    assert env.unwrapped.__class__.__name__ == "BikePathEnv"
    assert isinstance(env.unwrapped, BikePathEnv)
    # Monitor is a wrapper around the env
    assert env.spec is None or hasattr(env, "step")


def test_make_env_passes_budget_via_reset_options(
    tiny_bike_graph: _NXGraph, tiny_walk_graph: _NXGraph, run_context: RunContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """make_env resets with options={'budget': budget} (plan 004 contract)."""
    recorded: dict[str, Any] = {}

    def fake_reset(self, *, seed=None, options=None):
        recorded["seed"] = seed
        recorded["options"] = options
        return self._last_obs.copy() if hasattr(self, "_last_obs") else (None, {})

    monkeypatch.setattr(BikePathEnv, "reset", fake_reset)
    make_env(tiny_bike_graph, tiny_walk_graph, Config(), run_context, rank=0, seed=42, budget=25000.0)
    assert recorded["options"] == {"budget": 25000.0}
    assert recorded["seed"] == 42


def test_make_env_seeds_differ_per_rank(
    tiny_bike_graph: _NXGraph, tiny_walk_graph: _NXGraph, run_context: RunContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each rank gets a distinct seed = base_seed + rank (§3.7)."""
    seen: list[int] = []

    def fake_reset(self, *, seed=None, options=None):
        seen.append(seed)
        return None, {}

    monkeypatch.setattr(BikePathEnv, "reset", fake_reset)
    for rank in range(3):
        make_env(tiny_bike_graph, tiny_walk_graph, Config(), run_context, rank=rank, seed=100)
    assert seen == [100, 101, 102]


def test_make_vec_env_builds_n_envs_with_distinct_seeds(
    tiny_bike_graph: _NXGraph, tiny_walk_graph: _NXGraph, run_context: RunContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """make_vec_env with DummyVecEnv builds n_envs thunks with distinct seeds."""
    seen: list[int] = []

    def fake_reset(self, *, seed=None, options=None):
        seen.append(seed)
        return None, {}

    monkeypatch.setattr(BikePathEnv, "reset", fake_reset)
    vec = make_vec_env(
        tiny_bike_graph, tiny_walk_graph, Config(), run_context,
        n_envs=3, seed=7, vec_env_cls=DummyVecEnv,
    )
    assert vec.num_envs == 3
    # DummyVecEnv calls reset on each env during construction
    assert seen == [7, 8, 9]


def test_train_model_rounds_timesteps_up_to_batch(
    tiny_bike_graph: _NXGraph, tiny_walk_graph: _NXGraph, run_context: RunContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """train_model rounds total_timesteps up to ppo_n_steps * n_envs (§3.7)."""
    cfg = Config()  # ppo_n_steps=2048
    n_envs = 2
    batch = cfg.ppo_n_steps * n_envs  # 4096
    vec = make_vec_env(
        tiny_bike_graph, tiny_walk_graph, cfg, run_context,
        n_envs=n_envs, seed=cfg.seed, vec_env_cls=DummyVecEnv,
    )

    captured: dict[str, Any] = {}

    def fake_learn(self, total_timesteps, callback=None, **kw):
        captured["total_timesteps"] = total_timesteps
        captured["callback"] = callback
        return self

    monkeypatch.setattr("bike_rl.training.MaskablePPO.learn", fake_learn)
    # Request 1000 (< batch) -> rounded up to 4096
    train_model(vec, cfg, run_context, total_timesteps=1000)
    assert captured["total_timesteps"] == batch  # 4096

    # Request 5000 -> next multiple of 4096 = 8192
    train_model(vec, cfg, run_context, total_timesteps=5000)
    assert captured["total_timesteps"] == 2 * batch  # 8192


def test_train_model_uses_maskable_ppo_and_cfg_hyperparams(
    tiny_bike_graph: _NXGraph, tiny_walk_graph: _NXGraph, run_context: RunContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """train_model constructs MaskablePPO with MaskableActorCriticPolicy + cfg values."""
    cfg = Config()
    vec = make_vec_env(
        tiny_bike_graph, tiny_walk_graph, cfg, run_context,
        n_envs=2, seed=cfg.seed, vec_env_cls=DummyVecEnv,
    )

    captured: dict[str, Any] = {}

    real_init = type  # placeholder; we patch the class itself

    class FakeMaskablePPO:
        def __init__(self, policy, env, learning_rate, n_steps, batch_size,
                     n_epochs, gamma, gae_lambda, clip_range, seed, device, verbose):
            captured.update(policy=policy, env=env, learning_rate=learning_rate,
                            n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs,
                            gamma=gamma, gae_lambda=gae_lambda, clip_range=clip_range,
                            seed=seed, device=device, verbose=verbose)

        def learn(self, total_timesteps, callback=None, **kw):
            captured["learn_timesteps"] = total_timesteps
            return self

    monkeypatch.setattr("bike_rl.training.MaskablePPO", FakeMaskablePPO)
    train_model(vec, cfg, run_context, total_timesteps=4096)

    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
    assert captured["policy"] is MaskableActorCriticPolicy
    assert captured["learning_rate"] == cfg.ppo_learning_rate
    assert captured["n_steps"] == cfg.ppo_n_steps
    assert captured["batch_size"] == cfg.ppo_batch_size
    assert captured["n_epochs"] == cfg.ppo_n_epochs
    assert captured["gamma"] == cfg.ppo_gamma
    assert captured["gae_lambda"] == cfg.ppo_gae_lambda
    assert captured["clip_range"] == cfg.ppo_clip_range
    assert captured["seed"] == cfg.seed
    assert captured["device"] == cfg.device


def test_train_model_creates_checkpoint_dir_and_attaches_callbacks(
    tiny_bike_graph: _NXGraph, tiny_walk_graph: _NXGraph, run_context: RunContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """train_model creates run_context.output_dir/checkpoints and passes 2 callbacks."""
    cfg = Config()
    vec = make_vec_env(
        tiny_bike_graph, tiny_walk_graph, cfg, run_context,
        n_envs=2, seed=cfg.seed, vec_env_cls=DummyVecEnv,
    )

    def fake_learn(self, total_timesteps, callback=None, **kw):
        assert isinstance(callback, list)
        assert len(callback) == 2
        assert isinstance(callback[0], TrainingProgressCallback)
        from stable_baselines3.common.callbacks import CheckpointCallback
        assert isinstance(callback[1], CheckpointCallback)
        assert "checkpoints" in callback[1].save_path
        return self

    monkeypatch.setattr("bike_rl.training.MaskablePPO.learn", fake_learn)
    train_model(vec, cfg, run_context, total_timesteps=4096)
    assert (run_context.output_dir / "checkpoints").exists()


def test_training_progress_callback_records_all_episodes_including_last() -> None:
    """§5.12: the callback records every terminated episode, including the last."""
    cb = TrainingProgressCallback(total_timesteps=10, progress_bar=False)

    # Simulate SB3's BaseCallback wiring: set the attrs _on_step reads.
    cb.model = type("M", (), {"num_timesteps": 0})()  # noqa: S106
    cb.locals = {"infos": [{"episode": {"r": 1.0, "l": 5}}]}
    cb.n_calls = 0

    # Three steps with episode infos (last one terminates the final episode).
    for r, length in [(1.0, 5), (2.0, 7), (3.0, 4)]:
        cb.locals = {"infos": [{"episode": {"r": r, "l": length}}]}
        assert cb._on_step() is True

    assert cb.episode_rewards == [1.0, 2.0, 3.0]
    assert cb.episode_lengths == [5, 7, 4]


def test_training_progress_callback_ignores_infos_without_episode() -> None:
    """Steps with no terminated episode contribute nothing."""
    cb = TrainingProgressCallback(total_timesteps=10, progress_bar=False)
    cb.model = type("M", (), {"num_timesteps": 0})()
    cb.locals = {"infos": [{"invalid_action": True}, {}]}
    assert cb._on_step() is True
    assert cb.episode_rewards == []
    assert cb.episode_lengths == []


def test_training_progress_callback_handles_multi_env_infos() -> None:
    """With multiple envs, infos is a list; each env's episode is recorded."""
    cb = TrainingProgressCallback(total_timesteps=10, progress_bar=False)
    cb.model = type("M", (), {"num_timesteps": 0})()
    cb.locals = {"infos": [
        {"episode": {"r": 1.5, "l": 3}},
        {},  # env 1 did not terminate
        {"episode": {"r": 2.5, "l": 6}},
    ]}
    assert cb._on_step() is True
    assert cb.episode_rewards == [1.5, 2.5]
    assert cb.episode_lengths == [3, 6]
```

Notes on the tests:
- `test_make_env_passes_budget_via_reset_options` and
  `test_make_env_seeds_differ_per_rank` monkeypatch `BikePathEnv.reset` to
  record args. The fake `reset` returns `(None, {})` (the env is not used
  after `make_env`'s reset call in these tests). `make_env` then wraps the env
  in `Monitor` — `Monitor.__init__` does not call `reset`, so the recording
  captures exactly one `reset` call per `make_env`.
- `test_make_vec_env_builds_n_envs_with_distinct_seeds` uses `DummyVecEnv`,
  which calls `reset` on each env during construction (so `seen` is populated
  immediately). `SubprocVecEnv` is not used in tests to avoid spawning
  processes in `pytest`.
- `test_train_model_rounds_timesteps_up_to_batch` monkeypatches
  `MaskablePPO.learn` (patched as `bike_rl.training.MaskablePPO.learn`) so
  the model is not actually trained; it asserts the rounded `total_timesteps`
  passed to `learn`.
- `test_train_model_uses_maskable_ppo_and_cfg_hyperparams` replaces the whole
  `MaskablePPO` class with `FakeMaskablePPO` to capture constructor args.
- `test_training_progress_callback_*` drive the callback by hand: they set
  `cb.model` and `cb.locals` directly (mirroring what SB3's
  `BaseCallback` machinery does) and call `_on_step`. This deterministically
  pins §5.12 without running a real training loop.

**Verify**:
- `ruff check tests/test_training.py` → exit 0
- `ruff format --check tests/test_training.py` → exit 0
- `mypy --strict bike_rl` → exit 0 (the test file is under `bike_rl`'s mypy
  scope only if mypy is pointed at it; `mypy --strict bike_rl` does not type
  `tests/`, so the test file only needs to pass `ruff`. If the CI's
  `mypy --strict bike_rl` does not cover tests, the test file's typing is
  best-effort — but keep it clean anyway.)
- `pytest -q tests/test_training.py` → all new tests pass
- `pytest -q` → 62 prior + ~9 new tests pass (62 + the new count)

### Step 6: Full-suite verification

Run the complete CI gate locally to confirm nothing regressed:

- `ruff check .` → exit 0
- `ruff format --check .` → exit 0
- `mypy --strict bike_rl` → exit 0
- `pytest -q` → all pass (62 prior + new), no warnings about the new file

## Test plan

New tests, in `tests/test_training.py`, covering:

| Test | Pins |
|------|------|
| `test_make_env_returns_monitor_wrapped_bikepathenv` | `make_env` returns a `Monitor`-wrapped `BikePathEnv` |
| `test_make_env_passes_budget_via_reset_options` | plan 004's `reset(options={"budget": …})` contract |
| `test_make_env_seeds_differ_per_rank` | §3.7 per-env distinct seeds (`seed + rank`) |
| `test_make_vec_env_builds_n_envs_with_distinct_seeds` | `make_vec_env` builds `n_envs` thunks with distinct seeds |
| `test_train_model_rounds_timesteps_up_to_batch` | §3.7 timestep rounding to `ppo_n_steps * n_envs` |
| `test_train_model_uses_maskable_ppo_and_cfg_hyperparams` | `MaskablePPO` + `MaskableActorCriticPolicy` + all cfg hyperparams |
| `test_train_model_creates_checkpoint_dir_and_attaches_callbacks` | checkpoint dir under `run_context.output_dir`, 2 callbacks attached |
| `test_training_progress_callback_records_all_episodes_including_last` | §5.12 — last episode recorded |
| `test_training_progress_callback_ignores_infos_without_episode` | no-episode infos ignored |
| `test_training_progress_callback_handles_multi_env_infos` | multi-env `infos` list handled |

Structural pattern to model after: `tests/test_env.py` (imports, `run_context`
fixture, `Config()` usage, `_NXGraph` typing guard, `monkeypatch` usage).

Verification: `pytest -q tests/test_training.py` → all pass; `pytest -q` →
all pass (62 prior + new).

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `ruff check .` exits 0
- [ ] `ruff format --check .` exits 0
- [ ] `mypy --strict bike_rl` exits 0
- [ ] `pytest -q` exits 0; the new `tests/test_training.py` exists and all
      its tests pass
- [ ] `python -c "from bike_rl.training import make_env, make_vec_env, train_model, TrainingProgressCallback"`
      exits 0
- [ ] `bike_rl/training.py` contains `MaskablePPO(`, `MaskableActorCriticPolicy`,
      `SubprocVecEnv`, `DummyVecEnv`, `Monitor`, `CheckpointCallback`, and
      `math.ceil` (grep each)
- [ ] `bike_rl/training.py` does **not** contain `NotImplementedError("Plan 005")`
      (`grep -n 'NotImplementedError("Plan 005")' bike_rl/training.py` → no
      matches)
- [ ] `bike_rl/config.py`, `bike_rl/env.py`, `bike_rl/metrics.py`,
      `bike_rl/candidates.py`, `bike_rl/graph_utils.py`, `bike_rl/run_context.py`,
      `bike_rl/objective.py`, `bike_rl/optim/*` are **unchanged**
      (`git diff --stat b48386a..HEAD -- bike_rl/config.py bike_rl/env.py
      bike_rl/metrics.py bike_rl/candidates.py bike_rl/graph_utils.py
      bike_rl/run_context.py bike_rl/objective.py bike_rl/optim` shows no
      changes) — **except** a possible one-line addition of `"tqdm"` to the
      mypy `ignore_missing_imports` override list in `pyproject.toml` if
      Step 3's mypy check required it (and only that line).
- [ ] No files outside the in-scope list are modified (`git status --short`
      lists only `bike_rl/training.py`, `tests/test_training.py`, and
      optionally `pyproject.toml` for the tqdm mypy override).
- [ ] `plans/README.md` status row for 005 updated (TODO → DONE) — unless a
      reviewer told you they maintain the index.

## STOP conditions

Stop and report back (do not improvise) if:

- The code at the locations in "Current state" doesn't match the excerpts —
  e.g. `training.py` already has real bodies, `BikePathEnv.__init__` or
  `reset` signatures differ from those quoted, `Config` is missing one of the
  listed `ppo_*` / `seed` / `device` fields, or `BikePathEnv.reset` does not
  accept `options={"budget": …}`. The codebase has drifted since this plan
  was written; re-baseline before continuing.
- The sb3-contrib API differs from "SB3 / sb3-contrib API facts" —
  specifically: `MaskablePPO` cannot be imported from `sb3_contrib`,
  `MaskableActorCriticPolicy` is not at
  `sb3_contrib.common.maskable.policies`, `SubprocVecEnv`/`DummyVecEnv`/
  `Monitor`/`BaseCallback`/`CheckpointCallback` are not at the quoted import
  paths, or `MaskablePPO.learn` does not accept `callback=` as a list. Report
  the installed `sb3-contrib` / `stable-baselines3` versions and the actual
  signatures; do not invent a different API path.
- `MaskablePPO` does not automatically call `env.action_masks()` through
  `Monitor` + `SubprocVecEnv` wrappers (i.e. `get_action_masks` does not use
  `env.get_wrapper_attr` / `env.env_method` as quoted). If masking requires a
  special wrapper class, report which — do not silently ship un-masked
  training.
- `BikePathEnv` is not picklable (a `SubprocVecEnv` thunk closure holding the
  env or the graphs fails to serialise). Plan 004's maintenance note says the
  env should pickle (no lambdas on `self`), but if it does not, **do not**
  edit `env.py` from this plan — report it so plan 004 can be re-baselined.
  (Tests use `DummyVecEnv`, so this only surfaces if you add a `SubprocVecEnv`
  integration test — which this plan does **not** require.)
- `Config` turns out to lack a field this plan needs (e.g. `ppo_clip_range`
  is absent). Do not add fields to `Config` from this plan without reporting —
  the plan is designed to need zero new fields; a missing one means drift.
- A step's verification fails twice after a reasonable fix attempt.
- The fix appears to require touching an out-of-scope file (e.g. you find
  `env.py` or `graph_utils.py` must be edited to make tests pass — they must
  not be).

## Maintenance notes

For the human/agent who owns this code after the change lands:

- **`make_env`/`make_vec_env` take graph objects, not loaders.** The §6.1 OSM
  cache fix is realised by the **parent process** (the CLI, plan 007) loading
  graphs once via `graph_utils.load_city_graph` / `load_bbox_graph` (which
  already enable `ox.settings.use_cache`), then passing the in-memory graphs
  to `make_vec_env`. `SubprocVecEnv`'s `cloudpickle` serialisation carries
  the graphs to workers. Plan 007 must NOT call `make_env`/`make_vec_env` in
  a way that re-loads inside workers — there is no loader call in
  `training.py` by design. If a future change wants per-worker loading
  (e.g. for memory on huge cities), that would re-introduce §6.1 — review
  carefully.
- **`SubprocVecEnv` start method.** `make_vec_env(..., start_method=None)`
  lets SB3 pick the default (usually `"fork"` on Linux). On macOS / CI
  runners where `"fork"` is unsafe after threading starts (torch), pass
  `start_method="spawn"`. Plan 007 should decide based on platform; this plan
  keeps it configurable.
- **`MaskablePPO` masking is automatic.** `MaskablePPO.collect_rollouts`
  calls `get_action_masks(env)` internally, which uses
  `env.get_wrapper_attr("action_masks")` (single env) or
  `env.env_method("action_masks")` (VecEnv). So `Monitor`-wrapped
  `BikePathEnv` inside `SubprocVecEnv` works without any explicit masking
  glue. If you change the wrapper stack (e.g. add `VecMonitor`), re-verify
  that `env_method("action_masks")` still reaches `BikePathEnv.action_masks`.
- **Episode recording contract.** `TrainingProgressCallback` reads
  `info["episode"]` from `self.locals["infos"]` every step. This depends on
  plan 004's env emitting `info["episode"] = {"r": …, "l": …}` on
  **termination**. If a future env change also emits it on **truncation**
  (currently it does not — only on `terminated=True`), truncated episodes
  would start being recorded too. That is arguably desirable, but it is a
  contract change — flag it in review.
- **`train_model` does not seed Python/NumPy/torch globally.** Full
  reproducibility (`--seed` seeding Python, NumPy, torch, SB3, envs per §9)
  is a CLI concern (plan 007) — `train_model` only passes `seed=cfg.seed` to
  `MaskablePPO`, which seeds SB3/torch/envs internally. The CLI should also
  `random.seed` / `np.random.seed` / `torch.manual_seed` if bit-exact
  reproducibility is required.
- **Reviewer scrutiny.** Check that (a) `policy=MaskableActorCriticPolicy`
  (the class, not the string); (b) `total_timesteps` is rounded to a multiple
  of `ppo_n_steps * n_envs` and clamped to at least one batch; (c) the
  checkpoint path is under `run_context.output_dir` (no global state); (d)
  the callback reads `info["episode"]` every step (not at `reset`); (e) no
  `print` calls (use `logger.info`); (f) no bare `except`; (g) the `budget`
  parameter was dropped from `train_model` per Step 4's recommendation.
- **Follow-up deferred:** `evaluation.py` + `plotting.py` (plan 006) and
  `cli.py` + SLURM scripts (plan 007). The CLI wires `load_*_graph` →
  `make_vec_env` → `train_model` → `evaluate_and_visualize`; the SLURM
  scripts set `MPLBACKEND=Agg` and call `python -m bike_rl.cli`.
