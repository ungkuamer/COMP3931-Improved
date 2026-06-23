# Plan 006: Implement `evaluation.py` (deterministic rollouts + best-solution tracking, §3.8) + `plotting.py` (headless-safe §5.10) + `test_evaluation.py` / `test_plotting.py`

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**:
> `git diff --stat 289e069..HEAD -- bike_rl/evaluation.py bike_rl/plotting.py bike_rl/env.py bike_rl/training.py bike_rl/metrics.py bike_rl/candidates.py bike_rl/config.py bike_rl/run_context.py tests/conftest.py`
> If any in-scope or dependency file changed since this plan was written,
> compare the "Current state" excerpts against the live code before
> proceeding; on a mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: MED
- **Depends on**: plans/005-training-and-tests.md (DONE — provides
  `bike_rl.training.train_model` returning a `MaskablePPO`, and
  `bike_rl.env.BikePathEnv` with `action_masks()`, `reset(options={"budget": …})`,
  and a 4-component observation `[connectivity, efficiency, coverage, budget/initial]`).
- **Category**: feature (evaluation §3.8) + fix (headless plotting §5.10) + tests
- **Planned at**: commit `289e069`, 2026-06-23
- **Issue**: (not published)

## Why this matters

The original `nx-rx-simple-ur.py` has an `evaluate_and_visualize` step that
runs deterministic rollouts of the trained policy, picks the best solution by
reward, and renders a map + metric plots (RECREATE_SPEC §3.8). Two problems
make the current package unable to complete a run:

1. **`bike_rl/evaluation.py` and `bike_rl/plotting.py` are stubs** raising
   `NotImplementedError("Plan 006")`. There is no way to evaluate a trained
   model or produce any artefact. The CLI (plan 007) cannot be wired without
   these.
2. **Plotting must be headless-safe (§5.10).** The original calls
   `plt.show()` unconditionally, which breaks / spams logs on SLURM. The
   `plotting.py` stub already sets `matplotlib.use("Agg")` at import — this
   plan honours that and gates every `plt.show()` behind an explicit
   `show: bool = False` parameter (the `--show` CLI flag, plan 007).

This plan delivers: a deterministic evaluation loop that consumes the
`MaskablePPO` model from plan 005 + the `BikePathEnv` action-mask contract
from plan 004; an `EvaluationTracker` recording per-episode
reward/connectivity/efficiency/coverage/budget plus the best solution's
added edges; and headless plotting functions that write
`best_solution_map.png`, `evaluation_metrics.png`, `evaluation_rewards.png`
(and `training_rewards.png`), plus optional GeoJSON export (§8).

**Reconciliation note (important):** RECREATE_SPEC §11 item 6 lists
"`evaluation.py` + `plotting.py` + `test_cli.py`". `test_cli.py` is
**deferred to plan 007** in this plan, because `cli.py` itself is plan 007's
deliverable and does not exist yet (it is a stub raising
`NotImplementedError("Full CLI in plan 007")`). Writing `test_cli.py` now
would produce tests that fail against a stub. This plan instead writes
`tests/test_evaluation.py` and `tests/test_plotting.py`; plan 007 writes
`cli.py` and `tests/test_cli.py` together. The `plans/README.md` index is
updated to reflect this.

**Scoring note:** RECREATE_SPEC §3.8 selects the best solution by **reward**.
This plan follows that. The *canonical-objective* scoring of an RL policy
(OPTIMIZER_SPEC §3.2 / §8 `evaluate_rl_policy`) is a separate, optimiser-plan
concern that depends on `bike_rl.objective`, which is still a stub. This plan
does **not** import `bike_rl.objective` — evaluation reports the env's reward
and the four observation-derived metrics, exactly as §3.8 specifies.

## Current state

The package is scaffolded (plan 001), `graph_utils.py`/`candidates.py`
implemented (002), `metrics.py` implemented (003), `env.py` implemented (004),
`training.py` implemented (005). All 72 existing tests pass
(`pytest -q` → `72 passed`). The relevant files:

- `bike_rl/evaluation.py` — **stub**. `EvaluationTracker` is a dataclass with
  fields `rewards, paths, connectivity, efficiency, population, budget,
  best_reward`; `evaluate_and_visualize(model, bike_graph, walk_graph, cfg,
  run_context, num_evaluations=10)` raises `NotImplementedError("Plan 006")`.
  `Config` / `RunContext` are imported under `TYPE_CHECKING` only — keep those
  guarded imports and add the real runtime imports in step 1.
- `bike_rl/plotting.py` — **stub**. Already does `matplotlib.use("Agg")` at
  import (keep this). Three functions raise `NotImplementedError("Plan 006")`:
  ```python
  def render_solution_map(graph, added_paths, run_context, show: bool = False) -> Path: ...
  def plot_metrics(tracker: EvaluationTracker, run_context, show: bool = False) -> None: ...
  def plot_rewards(rewards: list[float], run_context, show: bool = False) -> None: ...
  ```
  `EvaluationTracker` / `RunContext` are imported under `TYPE_CHECKING` — keep
  guarded.

- `bike_rl/env.py` — implemented (plan 004). **This is the dependency you
  build on.** Verified at `289e069`:
  ```python
  class BikePathEnv(gym.Env[object, object]):
      def __init__(self, bike_graph, walk_graph, cfg: Config, run_context: RunContext) -> None: ...
      def reset(self, *, seed=None, options=None) -> tuple[npt.NDArray[np.float32], dict]: ...
      def action_masks(self) -> npt.NDArray[np.bool_]: ...
      def step(self, action) -> tuple[obs, float, bool, bool, dict]: ...
  ```
  - Observation is `[connectivity, path_efficiency, coverage, budget/initial_budget]`,
    all in `[0, 1]`, dtype `float32`. **Evaluation reads the final observation
    of each episode to get the four metrics — no private-attribute access
    needed for metrics.**
  - `reset(options={"budget": float})` sets the episode budget (plan 004
    contract). Evaluation MUST pass the budget via `options`; otherwise the
    env falls back to `100_000.0`.
  - `action_masks()` is on the **unwrapped** env. Evaluation builds
    `BikePathEnv` directly (NOT via `training.make_env`, to avoid
    `Monitor`/`VecEnv` wrapper indirection in the rollout loop), so
    `env.action_masks()` is called directly.
  - The env does **not** expose a public added-edges accessor. To collect the
    added paths for the best-solution map, evaluation reads
    `env._graph` (the env's live `nx.MultiDiGraph`) and diffs against the
    original `bike_graph` — see "Added-edge collection" below. This is the
    **only** private-attribute access in this plan and it is read-only.

- `bike_rl/training.py` — implemented (plan 005). Provides
  `train_model(...) -> MaskablePPO`. Evaluation receives the model object
  produced by `train_model` (or loaded from disk by the CLI via
  `MaskablePPO.load`). Evaluation only calls `model.predict`.

- `bike_rl/metrics.py` — implemented (plan 003). Pure functions
  `connectivity(graph, cfg)`, `path_efficiency(graph, cfg)`,
  `coverage(graph, cfg)`, `fragmentation(graph, cfg)`. Evaluation does NOT
  need to call these for per-episode metrics (the observation already carries
  them), but `render_solution_map`'s stats annotation may use
  `connectivity`/`coverage` on the best graph for the annotation text —
  optional, see step 3.

- `bike_rl/config.py` — frozen `Config`, fully implemented. **This plan adds
  NO new Config fields.** Fields used: `seed` (for deterministic eval seeds),
  `coverage_radius_m` / `coverage_mode` (only if you call `coverage` for
  annotation — optional), `edge_cost_factor` (not used here). If you are
  tempted to add a field, STOP and report.

- `bike_rl/run_context.py` — `RunContext(run_id, output_dir, timestamp)` with
  `ensure_output_dir() -> Path`. All plots/GeoJSON write under
  `run_context.output_dir`.

- `bike_rl/objective.py` and `bike_rl/optim/*` — **stubs** (optimiser plans).
  Do NOT import or implement them here.

- `tests/conftest.py` — provides `tiny_bike_graph`, `tiny_walk_graph`,
  `mock_osm`. `test_evaluation.py` / `test_plotting.py` will reuse
  `tiny_bike_graph` / `tiny_walk_graph` and the `run_context` fixture pattern
  copied from `tests/test_env.py` (see step 4).

- `tests/test_evaluation.py`, `tests/test_plotting.py` — **do not exist**
  (verified). Create them.

### sb3-contrib / matplotlib API facts (verified in the venv at `289e069`)

- `MaskablePPO.predict(observation, state=None, episode_start=None,
  deterministic=False, action_masks=None) -> tuple[np.ndarray, state]`.
  For a single (non-Vec) env, pass `action_masks=env.action_masks()` and
  `deterministic=True`; the returned `action` is a 0-d / 1-element array —
  call `int(action)` before `env.step`. Verified.
- `matplotlib.use("Agg")` must be called **before** `import matplotlib.pyplot
  as plt`. The `plotting.py` stub already does this at module top — keep it
  exactly there and import `pyplot` below it.
- `matplotlib` 3.11, `geopandas` 1.1, `shapely` 2.1 are installed (verified).
  `geopandas` / `shapely` are in `pyproject.toml` dependencies and in the
  mypy `ignore_missing_imports` override list — so `import geopandas` /
  `import shapely` need no `type: ignore`.

### Added-edge collection (the one private-attr read)

`BikePathEnv` stores its live graph on `self._graph` (an `nx.MultiDiGraph`)
and never mutates the original `bike_graph` passed to `__init__`. After a
rollout, the added bike-lane edges are exactly the edges in `env._graph` that
are tagged `bike_lane == "yes"` and are **not** present in the original
`bike_graph`. Evaluation collects them as:

```python
raw = env._graph  # read-only access to the env's live graph
added = [
    (u, v, dict(d))
    for u, v, d in raw.edges(data=True)
    if d.get("bike_lane") == "yes" and not bike_graph.has_edge(u, v)
]
```

`bike_graph` is the original graph passed into `evaluate_and_visualize` (the
same object the env copied at construction), so `has_edge` correctly
identifies pre-existing edges. This is read-only and touches exactly one
private attribute. The STOP condition covers the case where `env._graph` is
renamed/removed; a future plan should add a public `added_edges()` accessor
to `BikePathEnv` (recorded in Maintenance notes) — do NOT add it from this
plan (`env.py` is out of scope).

## Commands you will need

| Purpose   | Command                                              | Expected on success |
|-----------|------------------------------------------------------|---------------------|
| Install   | `pip install -e ".[dev]"`                            | exit 0              |
| Lint      | `ruff check bike_rl/evaluation.py bike_rl/plotting.py tests/test_evaluation.py tests/test_plotting.py` | exit 0 |
| Format    | `ruff format --check bike_rl/evaluation.py bike_rl/plotting.py tests/test_evaluation.py tests/test_plotting.py` | exit 0 |
| Typecheck | `mypy --strict bike_rl`                              | exit 0, no errors   |
| Tests     | `pytest -q tests/test_evaluation.py tests/test_plotting.py` | all pass   |
| Full suite| `pytest -q`                                          | 72 prior + new pass |

(Exact commands from this repo — verified during recon. The CI workflow at
`.github/workflows/ci.yml` runs `ruff check .`, `ruff format --check .`,
`mypy --strict bike_rl`, `pytest -q` on Python 3.10 and 3.12.)

## Scope

**In scope** (the only files you should modify):
- `bike_rl/evaluation.py` — replace the stub with the full implementation.
- `bike_rl/plotting.py` — replace the stub with the full implementation
  (keep the existing `matplotlib.use("Agg")` line at module top).
- `tests/test_evaluation.py` — create.
- `tests/test_plotting.py` — create.
- `plans/README.md` — update the 006 status row and the dependency-notes
  bullets for 006/007 (test_cli.py moved to 007).

**Out of scope** (do NOT touch, even though they look related):
- `bike_rl/env.py`, `bike_rl/training.py`, `bike_rl/metrics.py`,
  `bike_rl/candidates.py`, `bike_rl/graph_utils.py`, `bike_rl/run_context.py`,
  `bike_rl/config.py` — all implemented by prior plans; do not modify. If a
  test needs an env behaviour that is missing, STOP and report — do not edit
  `env.py`.
- `bike_rl/cli.py` — stub for plan 007; do not implement here.
- `bike_rl/objective.py`, `bike_rl/optim/*` — stubs for the optimiser plans.
  Do NOT import `objective` from evaluation; evaluation scores by reward
  (RECREATE_SPEC §3.8), not the canonical objective.
- `tests/test_cli.py` — **deferred to plan 007** (with `cli.py`). Do not
  create it here.
- `tests/conftest.py`, other existing test files — do not modify. Reuse
  fixtures by importing them in the new test files.
- `pyproject.toml`, `requirements.txt`, CI config — `matplotlib`,
  `geopandas`, `shapely` are already declared (verified); no changes needed.

## Git workflow

- Branch: `advisor/006-evaluation-and-plotting`
- Commit per step or per logical unit; message style: conventional commits —
  e.g. `feat(eval): implement deterministic rollouts + EvaluationTracker (§3.8)`
  (match the repo's existing style: see `git log --oneline` —
  `feat(training): …`, `feat(env): …`, `test(reward): …`).
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Replace `evaluation.py` stub with imports and `EvaluationTracker`

Write `bike_rl/evaluation.py` with this exact module header and the
`EvaluationTracker` dataclass. Keep the `TYPE_CHECKING`-guarded `Config` /
`RunContext` imports from the stub; add `MaskablePPO` under `TYPE_CHECKING`
too (evaluation calls `model.predict` at runtime but only needs the type for
annotations). Add real runtime imports for `logging`, `Any`, `gym` (not
needed — do not import gym), `networkx`, `numpy`, and `BikePathEnv`.

```python
"""Deterministic policy evaluation + best-solution tracking. See RECREATE_SPEC.md §3.8."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import networkx as nx

from bike_rl.env import BikePathEnv

if TYPE_CHECKING:
    from bike_rl.config import Config
    from bike_rl.run_context import RunContext
    from sb3_contrib import MaskablePPO

    _NXGraph = nx.MultiDiGraph[Any, Any, Any]
else:
    _NXGraph = nx.MultiDiGraph

logger = logging.getLogger(__name__)


@dataclass
class EvaluationTracker:
    """Per-episode evaluation metrics + the best solution's added edges (§3.8).

    Attributes:
        rewards: Total reward per episode (sum of step rewards).
        connectivity: Final connectivity per episode (from the final obs).
        efficiency: Final path efficiency per episode.
        population: Final coverage (population-served proxy) per episode.
        budget_remaining: Remaining budget at episode end, per episode.
        n_episodes: Number of evaluation episodes run.
        best_reward: The highest ``rewards[i]``; ``-inf`` if none ran.
        best_index: Index into the per-episode lists of the best episode,
            or ``-1`` if none ran.
        best_added_edges: Edges added in the best episode, as
            ``(u, v, data_dict)`` triples, for rendering the solution map.
    """

    rewards: list[float] = field(default_factory=list)
    connectivity: list[float] = field(default_factory=list)
    efficiency: list[float] = field(default_factory=list)
    population: list[float] = field(default_factory=list)
    budget_remaining: list[float] = field(default_factory=list)
    n_episodes: int = 0
    best_reward: float = float("-inf")
    best_index: int = -1
    best_added_edges: list[tuple[Any, Any, dict[str, Any]]] = field(default_factory=list)
```

Notes:
- The stub's `paths` / `budget` field names are replaced by `best_added_edges`
  / `budget_remaining` (clearer; the stub is meant to be replaced). The
  `best_added_edges` triple type is `tuple[Any, Any, dict[str, Any]]` because
  node ids are `int | str` and `Any` keeps mypy strict happy without a union
  gymnastic.
- Do NOT import `gymnasium` — evaluation builds `BikePathEnv` directly and
  does not need the `gym` type at runtime.

**Verify**:
- `ruff check bike_rl/evaluation.py` → exit 0
- `ruff format --check bike_rl/evaluation.py` → exit 0
- `mypy --strict bike_rl/evaluation.py` → exit 0

### Step 2: Implement `_run_episode` and `evaluate_and_visualize`

Add these to `bike_rl/evaluation.py`. The evaluation loop builds a fresh
`BikePathEnv` per episode, runs `MaskablePPO.predict` deterministically with
action masks, accumulates reward, and reads the final observation for
metrics. The best episode (by reward) has its added edges captured for
rendering. Plotting is delegated to `bike_rl.plotting` and is skipped when
`no_plots=True`.

```python
def _run_episode(
    model: MaskablePPO,
    bike_graph: _NXGraph,
    walk_graph: _NXGraph,
    cfg: Config,
    run_context: RunContext,
    budget: float,
    seed: int,
) -> tuple[float, float, float, float, float, list[tuple[Any, Any, dict[str, Any]]]]:
    """Run one deterministic rollout and return metrics + added edges.

    Returns ``(total_reward, connectivity, efficiency, coverage,
    budget_remaining, added_edges)``. The four metric values are read from
    the final observation (plan 004's observation contract); ``added_edges``
    is the diff of the env's live graph against ``bike_graph`` (see "Added-edge
    collection" in the plan).
    """
    env = BikePathEnv(bike_graph, walk_graph, cfg, run_context)
    obs, _ = env.reset(seed=seed, options={"budget": float(budget)})
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        mask = env.action_masks()
        if not bool(mask.any()):
            break  # no legal action — episode is over
        action, _ = model.predict(obs, deterministic=True, action_masks=mask)
        obs, reward, terminated, truncated, _info = env.step(int(action))
        total_reward += float(reward)
        steps += 1
        if steps > 10_000:  # hard safety cap; env terminates by budget normally
            logger.warning("eval episode hit 10_000-step safety cap at seed=%d", seed)
            break

    conn = float(obs[0])
    eff = float(obs[1])
    pop = float(obs[2])
    norm_budget = float(obs[3])
    budget_remaining = norm_budget * float(budget)

    raw = env._graph  # noqa: SLF001 — read-only; see plan "Added-edge collection"
    added: list[tuple[Any, Any, dict[str, Any]]] = [
        (u, v, dict(d))
        for u, v, d in raw.edges(data=True)
        if d.get("bike_lane") == "yes" and not bike_graph.has_edge(u, v)
    ]
    return total_reward, conn, eff, pop, budget_remaining, added


def evaluate_and_visualize(
    model: object,
    bike_graph: _NXGraph,
    walk_graph: _NXGraph,
    cfg: Config,
    run_context: RunContext,
    num_evaluations: int = 10,
    budget: float = 100_000.0,
    seed: int | None = None,
    show: bool = False,
    export_geojson: bool = False,
    no_plots: bool = False,
) -> EvaluationTracker:
    """Evaluate a trained policy with deterministic rollouts (RECREATE_SPEC §3.8).

    Runs ``num_evaluations`` fresh episodes, records per-episode reward and
    final metrics (connectivity, efficiency, coverage, remaining budget),
    selects the best episode by **reward** (§3.8), and — unless
    ``no_plots`` — renders the best-solution map and metric/reward plots into
    ``run_context.output_dir`` via :mod:`bike_rl.plotting`.

    Args:
        model: A trained ``MaskablePPO`` (from :func:`bike_rl.training.train_model`
            or ``MaskablePPO.load``). Typed ``object`` to avoid a hard
            runtime import of ``sb3_contrib``; only ``model.predict`` is used.
        bike_graph: The original bike network (candidates are drawn from the
            walk graph; the env copies this and never mutates it).
        walk_graph: The walkable network.
        cfg: Config (uses ``cfg.seed`` when ``seed`` is None).
        run_context: Per-run output context for plot/GeoJSON paths.
        num_evaluations: Number of deterministic episodes.
        budget: Episode budget (passed via ``reset(options={"budget": …})``).
        seed: Base seed; episode ``i`` uses ``seed + i``. Defaults to ``cfg.seed``.
        show: If True, call ``plt.show()`` on each figure (headless default False — §5.10).
        export_geojson: If True, write ``suggested_bike_paths.geojson`` for the
            best solution (RECREATE_SPEC §8).
        no_plots: If True, skip all rendering (RECREATE_SPEC §9 ``--no_plots``).

    Returns:
        The populated :class:`EvaluationTracker`.
    """
    base_seed = cfg.seed if seed is None else seed
    tracker = EvaluationTracker()
    masked_model = cast("MaskablePPO", model)  # public type is `object`; cast for the internal helper

    for i in range(num_evaluations):
        total_r, conn, eff, pop, brem, added = _run_episode(
            masked_model, bike_graph, walk_graph, cfg, run_context, budget, base_seed + i
        )
        tracker.rewards.append(total_r)
        tracker.connectivity.append(conn)
        tracker.efficiency.append(eff)
        tracker.population.append(pop)
        tracker.budget_remaining.append(brem)
        if total_r > tracker.best_reward:
            tracker.best_reward = total_r
            tracker.best_index = i
            tracker.best_added_edges = added

    tracker.n_episodes = num_evaluations
    logger.info(
        "evaluation: %d episodes, best_reward=%.4f (episode %d, %d added edges)",
        tracker.n_episodes, tracker.best_reward, tracker.best_index,
        len(tracker.best_added_edges),
    )

    if no_plots or tracker.best_index < 0:
        return tracker

    from bike_rl.plotting import (
        export_geojson as export_geojson_fn,
        plot_metrics,
        plot_rewards,
        render_solution_map,
    )

    out_dir = run_context.ensure_output_dir()
    best_graph = bike_graph.copy()
    for u, v, d in tracker.best_added_edges:
        best_graph.add_edge(u, v, **d)
    render_solution_map(best_graph, tracker.best_added_edges, run_context, show=show)
    plot_metrics(tracker, run_context, show=show)
    plot_rewards(tracker.rewards, run_context, show=show, filename="evaluation_rewards.png")
    if export_geojson:
        export_geojson_fn(tracker.best_added_edges, run_context)
    _ = out_dir  # ensure dir created even if a plotting impl skips it
    return tracker
```

Notes:
- `model: object` with `model.predict(...)` — mypy strict will complain that
  `object` has no `predict`. Use a `cast` inside `_run_episode`:
  ```python
  from typing import cast
  from sb3_contrib import MaskablePPO  # TYPE_CHECKING only at top; for cast use:
  ```
  The public `evaluate_and_visualize(model: object, ...)` is intentionally
  loosely typed (callers should not have to import `sb3_contrib`), so it
  casts to the precise internal type once: `cast("MaskablePPO", model)`.
  `_run_episode`'s `model` parameter is typed `MaskablePPO` (a forward ref
  under `from __future__ import annotations`, resolved from the
  `TYPE_CHECKING` import). `from typing import cast` is added in step 1's
  import block.
- The `env._graph` access (one line, read-only) is the documented private-attr
  read. Keep the `# noqa: SLF001` comment so ruff does not flag it (the repo's
  ruff select includes `SIM` but not `SLF` — verify with `ruff check`; if
  `SLF001` is not in the selected rules, the noqa is harmless and can stay).
- `int(action)`: `MaskablePPO.predict` returns a numpy array; for a single env
  it is a 0-d/1-element int array. `int(action)` converts it. `BikePathEnv.step`
  accepts `np.integer` and `int` (verified in env.py).
- Plotting imports are **lazy** (inside the `if not no_plots` block) so
  `evaluation.py` does not require matplotlib at import time — tests that only
  exercise the rollout (with `no_plots=True`) do not pull in matplotlib.
- `best_graph` is built by copying `bike_graph` and adding the best episode's
  edges, so `render_solution_map` gets a coherent graph (original + added) to
  draw. This avoids passing the env's private graph to plotting.
- `steps > 10_000` safety cap: the env terminates by budget exhaustion or
  candidate depletion normally; this cap only guards against a pathological
  masked-action loop. It is not expected to trigger in tests.

**Verify**:
- `ruff check bike_rl/evaluation.py` → exit 0
- `ruff format --check bike_rl/evaluation.py` → exit 0
- `mypy --strict bike_rl/evaluation.py` → exit 0. If mypy complains about
  `env._graph` (private access is fine for mypy; it does not flag that), the
  likely issue is the `model` cast — ensure `cast("MaskablePPO", model)` uses
  the string form (forward ref under `from __future__ import annotations`).

### Step 3: Implement `plotting.py` (headless-safe rendering + plots + GeoJSON)

Replace the `plotting.py` stub. Keep `matplotlib.use("Agg")` as the **first**
matplotlib-related line (before `import pyplot`). All functions gate
`plt.show()` behind `show` and save to `run_context.output_dir / filename`.

```python
"""Headless-safe plotting for bike-path solutions. See RECREATE_SPEC.md §5.10, §3.8."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")  # headless-safe (§5.10); must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402

if TYPE_CHECKING:
    from bike_rl.evaluation import EvaluationTracker
    from bike_rl.run_context import RunContext

    _NXGraph = Any
else:
    _NXGraph = Any

logger = logging.getLogger(__name__)


def _node_xy(graph: object, n: Any) -> tuple[float, float]:
    """Return ``(x, y)`` for node ``n`` from its attribute dict (x=lon, y=lat)."""
    data = graph.nodes[n]  # type: ignore[union-attr]
    return float(data.get("x", 0.0)), float(data.get("y", 0.0))


def render_solution_map(
    graph: object,
    added_paths: list[tuple[Any, Any, dict[str, Any]]],
    run_context: RunContext,
    show: bool = False,
    stats: dict[str, float] | None = None,
    filename: str = "best_solution_map.png",
) -> Path:
    """Render the bike network with ``added_paths`` highlighted in red (§3.8).

    Draws every edge of ``graph`` in grey and each edge in ``added_paths`` in
    red, using node ``x``/``y`` attributes. Optionally annotates ``stats``
    (e.g. connectivity/coverage) in the corner. Saves to
    ``run_context.output_dir / filename`` and returns the path.

    Args:
        graph: The full bike network (original + added edges).
        added_paths: ``(u, v, data)`` triples of the added edges to highlight.
        run_context: Per-run output context.
        show: If True, call ``plt.show()`` (default False — headless, §5.10).
        stats: Optional ``{label: value}`` dict rendered as an annotation.
        filename: Output filename within ``run_context.output_dir``.

    Returns:
        The path the figure was written to.
    """
    out_dir = run_context.ensure_output_dir()
    fig, ax = plt.subplots(figsize=(8, 8))
    # Base edges (grey).
    for u, v in graph.edges():  # type: ignore[union-attr]
        x1, y1 = _node_xy(graph, u)
        x2, y2 = _node_xy(graph, v)
        ax.plot([x1, x2], [y1, y2], color="#cccccc", linewidth=1.0, zorder=1)
    # Added edges (red).
    added_set = {(u, v) for u, v, _d in added_paths}
    for u, v in added_paths:
        x1, y1 = _node_xy(graph, u)
        x2, y2 = _node_xy(graph, v)
        ax.plot([x1, x2], [y1, y2], color="red", linewidth=2.5, zorder=3)
    _ = added_set  # retained for future parallel-edge handling
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Suggested bike lanes (red)")
    ax.set_aspect("equal", adjustable="datalim")
    if stats:
        text = "\n".join(f"{k}: {v:.4f}" for k, v in stats.items())
        ax.text(
            0.02, 0.98, text, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, family="monospace",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )
    out_path = out_dir / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_metrics(
    tracker: EvaluationTracker,
    run_context: RunContext,
    show: bool = False,
    filename: str = "evaluation_metrics.png",
) -> Path:
    """Plot connectivity/efficiency/coverage over evaluation episodes (§3.8).

    Saves to ``run_context.output_dir / filename`` and returns the path.
    """
    out_dir = run_context.ensure_output_dir()
    fig, ax = plt.subplots(figsize=(8, 5))
    episodes = list(range(1, tracker.n_episodes + 1))
    ax.plot(episodes, tracker.connectivity, marker="o", label="connectivity")
    ax.plot(episodes, tracker.efficiency, marker="s", label="efficiency")
    ax.plot(episodes, tracker.population, marker="^", label="coverage")
    ax.set_xlabel("evaluation episode")
    ax.set_ylabel("metric value [0, 1]")
    ax.set_title("Evaluation metrics per episode")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="best")
    out_path = out_dir / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_rewards(
    rewards: list[float],
    run_context: RunContext,
    show: bool = False,
    filename: str = "training_rewards.png",
) -> Path:
    """Plot a reward history (training or evaluation) and save it (§3.8/§9).

    ``filename`` selects between ``training_rewards.png`` (default, training
    callback rewards) and ``evaluation_rewards.png`` (evaluation rewards).
    """
    out_dir = run_context.ensure_output_dir()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(range(1, len(rewards) + 1)), rewards, marker="o", color="tab:blue")
    ax.set_xlabel("episode")
    ax.set_ylabel("total reward")
    ax.set_title("Reward history")
    out_path = out_dir / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def export_geojson(
    added_edges: list[tuple[Any, Any, dict[str, Any]]],
    run_context: RunContext,
    graph: object | None = None,
    filename: str = "suggested_bike_paths.geojson",
) -> Path:
    """Write the added bike-lane edges to a GeoJSON file (RECREATE_SPEC §8).

    Each added edge becomes a LineString feature between its endpoint
    ``x``/``y`` coordinates. If ``graph`` is given, coordinates are read from
    its node attributes; otherwise the edge ``data`` dict's ``x``/``y`` are
    used if present (else 0/0). Uses geopandas + shapely.

    Args:
        added_edges: ``(u, v, data)`` triples of added edges.
        run_context: Per-run output context.
        graph: Optional graph to read node coordinates from.
        filename: Output filename within ``run_context.output_dir``.

    Returns:
        The path the GeoJSON was written to.
    """
    import geopandas as gpd
    from shapely.geometry import LineString

    out_dir = run_context.ensure_output_dir()
    geometries = []
    properties: list[dict[str, Any]] = []
    for u, v, d in added_edges:
        if graph is not None:
            x1, y1 = _node_xy(graph, u)
            x2, y2 = _node_xy(graph, v)
        else:
            x1, y1 = float(d.get("x", 0.0)), float(d.get("y", 0.0))
            x2, y2 = x1, y1
        geometries.append(LineString([(x1, y1), (x2, y2)]))
        props = {k: v for k, v in d.items() if isinstance(v, (str, int, float, bool))}
        props["u"] = str(u)
        props["v"] = str(v)
        properties.append(props)
    gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs="EPSG:4326")
    out_path = out_dir / filename
    gdf.to_file(out_path, driver="GeoJSON")
    return out_path
```

Notes:
- `# noqa: E402` on `import matplotlib.pyplot as plt` is required because
  `matplotlib.use("Agg")` must run before the pyplot import, so the import is
  not at the true top of the file (ruff `E402` = module-level import not at
  top). This is the canonical pattern for Agg-first plotting.
- `_NXGraph = Any` (both branches) — plotting does not need the real graph
  type; `Any` keeps mypy strict quiet without a `TYPE_CHECKING` union.
  `graph.edges()` / `graph.nodes[n]` are guarded with `# type: ignore[union-attr]`
  where needed (on `object`). If you prefer, type `graph: Any` directly in the
  signatures instead of `object` — but the stubs use `object`, and `object`
  with `type: ignore[union-attr]` on the two call sites is cleaner for the
  public API. Pick one and be consistent; the excerpt above uses `object` +
  ignores.
- `render_solution_map`'s `added_set` is currently unused for drawing (it is
  retained for clarity / future parallel-edge dedup). If ruff flags it as
  unused (`F841`), remove the `added_set` line and the `_ = added_set` line
  entirely — they are optional. (The plan includes them for readability; the
  executor may drop them if ruff complains.)
- `export_geojson` is a **new** function not in the stub. It realises the
  §8 `--export_geojson` behaviour. It is imported lazily-name in
  `evaluation.py` (`from bike_rl.plotting import export_geojson as
  export_geojson_fn`) — keep that alias consistent.
- `plot_metrics` / `plot_rewards` return `Path` (refined from the stub's
  `-> None`); a `Path` return is more useful and tests assert on it. Adding a
  `filename` param to `plot_rewards` and `plot_metrics` (with defaults) is
  backward-compatible.

**Verify**:
- `ruff check bike_rl/plotting.py` → exit 0 (the `E402` noqa silences the
  pyplot-after-use warning; if ruff still complains, re-check the noqa
  spelling)
- `ruff format --check bike_rl/plotting.py` → exit 0
- `mypy --strict bike_rl/plotting.py` → exit 0. `geopandas` / `shapely` are
  in the mypy `ignore_missing_imports` override list, so the lazy imports
  inside `export_geojson` need no `type: ignore`. If mypy flags
  `gpd.GeoDataFrame(...)` typing, add a targeted `# type: ignore[call-overload]`
  on that line only — but try without first.

### Step 4: Create `tests/test_evaluation.py`

Create the file. It builds a `BikePathEnv`-compatible fake model (so no real
PPO is trained), drives `evaluate_and_visualize` on the tiny fixtures, and
pins: per-episode recording, best-by-reward selection, added-edge capture,
`no_plots` skipping, and the budget-passthrough via `reset(options=...)`.

Model the file structure on `tests/test_env.py` (imports, `run_context`
fixture, `Config()` usage, `_NXGraph` typing guard, `monkeypatch` usage).

```python
"""Tests for ``bike_rl.evaluation``. See RECREATE_SPEC.md §3.8."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
import pytest

from bike_rl.config import Config
from bike_rl.evaluation import EvaluationTracker, evaluate_and_visualize

if TYPE_CHECKING:
    _NXGraph = nx.MultiDiGraph[Any, Any, Any]
else:
    _NXGraph = nx.MultiDiGraph


@pytest.fixture
def run_context(tmp_path: Path) -> Any:
    """RunContext with a temp output dir."""
    from bike_rl.run_context import RunContext

    return RunContext(
        run_id="test",
        output_dir=tmp_path / "out",
        timestamp=datetime.now(timezone.utc),
    )


class _FakeModel:
    """Stand-in for MaskablePPO: always picks the first legal action.

    ``evaluate_and_visualize`` only calls ``model.predict(obs, deterministic=,
    action_masks=)``; this fake returns the first unmasked index so the rollout
    is deterministic and exercises the env without training a real PPO.
    """

    def predict(self, observation: Any, deterministic: bool = False,
                action_masks: Any = None) -> tuple[np.ndarray, None]:
        mask = np.asarray(action_masks) if action_masks is not None else None
        if mask is not None and bool(mask.any()):
            idx = int(np.argmax(mask))  # first True
        else:
            idx = 0
        return np.array(idx, dtype=np.int64), None


def test_evaluate_records_per_episode_metrics(
    tiny_bike_graph: _NXGraph, tiny_walk_graph: _NXGraph, run_context: Any,
) -> None:
    """evaluate_and_visualize records one entry per episode across all lists."""
    tracker = evaluate_and_visualize(
        _FakeModel(), tiny_bike_graph, tiny_walk_graph, Config(), run_context,
        num_evaluations=3, budget=100_000.0, no_plots=True,
    )
    assert tracker.n_episodes == 3
    assert len(tracker.rewards) == 3
    assert len(tracker.connectivity) == len(tracker.efficiency) == 3
    assert len(tracker.population) == len(tracker.budget_remaining) == 3


def test_evaluate_selects_best_by_reward(
    tiny_bike_graph: _NXGraph, tiny_walk_graph: _NXGraph, run_context: Any,
) -> None:
    """best_index points at the argmax of rewards; best_reward matches."""
    tracker = evaluate_and_visualize(
        _FakeModel(), tiny_bike_graph, tiny_walk_graph, Config(), run_context,
        num_evaluations=4, budget=100_000.0, no_plots=True,
    )
    assert tracker.best_index == int(np.argmax(tracker.rewards))
    assert tracker.best_reward == max(tracker.rewards)


def test_evaluate_captures_added_edges_for_best(
    tiny_bike_graph: _NXGraph, tiny_walk_graph: _NXGraph, run_context: Any,
) -> None:
    """best_added_edges is non-empty when the policy adds at least one edge."""
    tracker = evaluate_and_visualize(
        _FakeModel(), tiny_bike_graph, tiny_walk_graph, Config(), run_context,
        num_evaluations=1, budget=100_000.0, no_plots=True,
    )
    # The tiny walk graph has affordable candidates; the fake model picks the
    # first legal one until budget exhaustion, so >=1 edge should be added.
    assert len(tracker.best_added_edges) >= 1
    for u, v, d in tracker.best_added_edges:
        assert d.get("bike_lane") == "yes"
        assert not tiny_bike_graph.has_edge(u, v)


def test_evaluate_no_plots_writes_no_figures(
    tiny_bike_graph: _NXGraph, tiny_walk_graph: _NXGraph, run_context: Any,
) -> None:
    """no_plots=True writes no image/geojson files into the output dir."""
    tracker = evaluate_and_visualize(
        _FakeModel(), tiny_bike_graph, tiny_walk_graph, Config(), run_context,
        num_evaluations=2, budget=100_000.0, no_plots=True, export_geojson=True,
    )
    # ensure_output_dir may create the dir, but no figures inside.
    if run_context.output_dir.exists():
        files = [p.name for p in run_context.output_dir.iterdir()]
        assert not any(f.endswith((".png", ".geojson")) for f in files), files
    _ = tracker


def test_evaluate_writes_plots_when_enabled(
    tiny_bike_graph: _NXGraph, tiny_walk_graph: _NXGraph, run_context: Any,
) -> None:
    """With plots enabled, the map + metric + reward PNGs are written."""
    evaluate_and_visualize(
        _FakeModel(), tiny_bike_graph, tiny_walk_graph, Config(), run_context,
        num_evaluations=2, budget=100_000.0, no_plots=False, export_geojson=False,
    )
    names = {p.name for p in run_context.output_dir.iterdir()}
    assert "best_solution_map.png" in names
    assert "evaluation_metrics.png" in names
    assert "evaluation_rewards.png" in names


def test_evaluate_export_geojson_writes_file(
    tiny_bike_graph: _NXGraph, tiny_walk_graph: _NXGraph, run_context: Any,
) -> None:
    """export_geojson=True writes suggested_bike_paths.geojson."""
    evaluate_and_visualize(
        _FakeModel(), tiny_bike_graph, tiny_walk_graph, Config(), run_context,
        num_evaluations=1, budget=100_000.0, no_plots=False, export_geojson=True,
    )
    assert (run_context.output_dir / "suggested_bike_paths.geojson").exists()


def test_evaluate_passes_budget_via_reset_options(
    tiny_bike_graph: _NXGraph, tiny_walk_graph: _NXGraph, run_context: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Evaluation resets each env with options={'budget': budget} (plan 004)."""
    from bike_rl.env import BikePathEnv

    seen: list[float] = []
    real_reset = BikePathEnv.reset

    def spy_reset(self, *, seed=None, options=None):
        if options and "budget" in options:
            seen.append(float(options["budget"]))
        return real_reset(self, seed=seed, options=options)

    monkeypatch.setattr(BikePathEnv, "reset", spy_reset)
    evaluate_and_visualize(
        _FakeModel(), tiny_bike_graph, tiny_walk_graph, Config(), run_context,
        num_evaluations=2, budget=25_000.0, no_plots=True,
    )
    assert seen == [25_000.0, 25_000.0]


def test_evaluation_tracker_defaults() -> None:
    """A fresh tracker has empty lists and -inf best_reward."""
    t = EvaluationTracker()
    assert t.rewards == []
    assert t.best_reward == float("-inf")
    assert t.best_index == -1
    assert t.best_added_edges == []
    assert t.n_episodes == 0
```

Notes on the tests:
- `_FakeModel.predict` returns `(np.array(idx, dtype=np.int64), None)`, matching
  `MaskablePPO.predict`'s return shape. `int(action)` in `_run_episode`
  converts it. `np.argmax(mask)` gives the first `True` index deterministically.
- `test_evaluate_no_plots_writes_no_figures` asserts no `.png`/`.geojson` files
  are written when `no_plots=True` (even with `export_geojson=True`, since
  `no_plots` short-circuits before the geojson call — this pins that
  `no_plots` fully disables all rendering, which is the §9 `--no_plots`
  semantics).
- `test_evaluate_passes_budget_via_reset_options` monkeypatches `BikePathEnv.reset`
  with a spy that calls through to the real reset, so the rollout still runs
  and exactly the budget-passing behaviour is pinned. Two episodes → two
  `25_000.0` entries.
- The fake model + tiny fixtures run the full rollout without importing
  `sb3_contrib` at test time (only `bike_rl.env`), keeping the tests fast.

**Verify**:
- `ruff check tests/test_evaluation.py` → exit 0
- `ruff format --check tests/test_evaluation.py` → exit 0
- `pytest -q tests/test_evaluation.py` → all pass

### Step 5: Create `tests/test_plotting.py`

Create the file. It calls each plotting function with the tiny fixtures and a
hand-built `EvaluationTracker`, and asserts the expected files appear in the
run dir. No `plt.show()` (headless). Model after `tests/test_evaluation.py`'s
fixture/imports.

```python
"""Tests for ``bike_rl.plotting`` (headless-safe). See RECREATE_SPEC.md §5.10."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx
import pytest

from bike_rl.evaluation import EvaluationTracker
from bike_rl.plotting import export_geojson, plot_metrics, plot_rewards, render_solution_map

if TYPE_CHECKING:
    _NXGraph = nx.MultiDiGraph[Any, Any, Any]
else:
    _NXGraph = nx.MultiDiGraph


@pytest.fixture
def run_context(tmp_path: Path) -> Any:
    from bike_rl.run_context import RunContext

    return RunContext(
        run_id="test",
        output_dir=tmp_path / "out",
        timestamp=datetime.now(timezone.utc),
    )


def test_render_solution_map_writes_png(
    tiny_bike_graph: _NXGraph, run_context: Any,
) -> None:
    """render_solution_map writes best_solution_map.png under the run dir."""
    added = [(1, 2, {"bike_lane": "yes", "length": 120.0})]
    out = render_solution_map(tiny_bike_graph, added, run_context, show=False)
    assert out.name == "best_solution_map.png"
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_solution_map_with_stats_annotation(
    tiny_bike_graph: _NXGraph, run_context: Any,
) -> None:
    """The optional stats dict is rendered without error."""
    added = [(1, 2, {"bike_lane": "yes", "length": 120.0})]
    out = render_solution_map(
        tiny_bike_graph, added, run_context, show=False,
        stats={"connectivity": 0.123, "coverage": 0.456},
    )
    assert out.exists()


def test_plot_metrics_writes_png(run_context: Any) -> None:
    """plot_metrics writes evaluation_metrics.png."""
    t = EvaluationTracker(
        rewards=[1.0, 2.0, 3.0],
        connectivity=[0.1, 0.2, 0.3],
        efficiency=[0.5, 0.5, 0.6],
        population=[0.2, 0.3, 0.4],
        budget_remaining=[100.0, 80.0, 60.0],
        n_episodes=3,
        best_reward=3.0,
        best_index=2,
    )
    out = plot_metrics(t, run_context, show=False)
    assert out.name == "evaluation_metrics.png"
    assert out.exists()


def test_plot_rewards_default_filename(run_context: Any) -> None:
    """plot_rewards defaults to training_rewards.png."""
    out = plot_rewards([10.0, 20.0, 15.0], run_context, show=False)
    assert out.name == "training_rewards.png"
    assert out.exists()


def test_plot_rewards_custom_filename(run_context: Any) -> None:
    """plot_rewards accepts a custom filename (evaluation_rewards.png)."""
    out = plot_rewards([1.0, 2.0], run_context, show=False, filename="evaluation_rewards.png")
    assert out.name == "evaluation_rewards.png"
    assert out.exists()


def test_export_geojson_writes_file(run_context: Any) -> None:
    """export_geojson writes a valid GeoJSON with one LineString per edge."""
    added = [(1, 2, {"bike_lane": "yes", "length": 200.0, "x": 0.0, "y": 0.0})]
    out = export_geojson(added, run_context)
    assert out.name == "suggested_bike_paths.geojson"
    assert out.exists()
    text = out.read_text()
    assert "LineString" in text or "FeatureCollection" in text


def test_plotting_does_not_call_show_by_default(
    tiny_bike_graph: _NXGraph, run_context: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§5.10: plt.show() is NOT called when show=False (headless default)."""
    import matplotlib.pyplot as plt

    called = {"show": False}
    monkeypatch.setattr(plt, "show", lambda *a, **k: called.__setitem__("show", True))
    added = [(1, 2, {"bike_lane": "yes", "length": 120.0})]
    render_solution_map(tiny_bike_graph, added, run_context, show=False)
    plot_rewards([1.0], run_context, show=False)
    assert called["show"] is False
```

Notes:
- `test_plotting_does_not_call_show_by_default` monkeypatches `plt.show` to
  prove the §5.10 headless default. It must monkeypatch **after** the
  `bike_rl.plotting` module is imported (plotting captured `plt` at import
  time via `import matplotlib.pyplot as plt`), so the test patches
  `matplotlib.pyplot.show` — which is the same object `plotting.py` calls.
  Verify this works; if `plotting.py`'s `plt.show()` is not the patched one,
  the test would still pass (it asserts the flag is False, i.e. show was not
  *called*) — but the monkeypatch makes the assertion meaningful. If the
  import-time binding means the patch does not intercept, switch the test to
  assert `show=False` is honoured by checking no GUI backend is invoked — but
  the Agg backend never blocks anyway, so the simplest robust assertion is
  that the functions return a path and exit without error when `show=False`.
  Keep the monkeypatch version; it is the clearest intent.

**Verify**:
- `ruff check tests/test_plotting.py` → exit 0
- `ruff format --check tests/test_plotting.py` → exit 0
- `pytest -q tests/test_plotting.py` → all pass

### Step 6: Full-suite verification + update `plans/README.md`

Run the complete CI gate locally to confirm nothing regressed:

- `ruff check .` → exit 0
- `ruff format --check .` → exit 0
- `mypy --strict bike_rl` → exit 0
- `pytest -q` → all pass (72 prior + new), no warnings about the new files

Then update `plans/README.md`:
- Add a row for 006 to the execution-order table:
  `| 006 | evaluation.py + plotting.py (headless §5.10) + test_evaluation/test_plotting | P1 | M | 005 | TODO |`
  (set to DONE when this plan is complete).
- Update the dependency-notes bullets: change the `006:` bullet to say
  `evaluation.py` + `plotting.py` + `test_evaluation.py`/`test_plotting.py`
  (§3.8/§5.10/§8), and note that **`test_cli.py` is moved to plan 007**
  (with `cli.py`), because `cli.py` does not exist yet. Update the `007:`
  bullet to say it includes `test_cli.py`.

## Test plan

New tests, covering:

| File | Test | Pins |
|------|------|------|
| `test_evaluation.py` | `test_evaluate_records_per_episode_metrics` | one entry per episode in every list (§3.8) |
| | `test_evaluate_selects_best_by_reward` | best by argmax(reward) (§3.8) |
| | `test_evaluate_captures_added_edges_for_best` | added edges captured, tagged `bike_lane='yes'`, not in original |
| | `test_evaluate_no_plots_writes_no_figures` | `no_plots=True` writes no `.png`/`.geojson` (§9 `--no_plots`) |
| | `test_evaluate_writes_plots_when_enabled` | map + metrics + rewards PNGs written |
| | `test_evaluate_export_geojson_writes_file` | `suggested_bike_paths.geojson` written (§8) |
| | `test_evaluate_passes_budget_via_reset_options` | budget passed via `reset(options=...)` (plan 004 contract) |
| | `test_evaluation_tracker_defaults` | fresh tracker defaults |
| `test_plotting.py` | `test_render_solution_map_writes_png` | map PNG written, non-empty |
| | `test_render_solution_map_with_stats_annotation` | stats annotation renders |
| | `test_plot_metrics_writes_png` | `evaluation_metrics.png` written |
| | `test_plot_rewards_default_filename` | default `training_rewards.png` |
| | `test_plot_rewards_custom_filename` | custom `evaluation_rewards.png` |
| | `test_export_geojson_writes_file` | valid GeoJSON with LineString/FeatureCollection |
| | `test_plotting_does_not_call_show_by_default` | §5.10 headless default |

Structural pattern to model after: `tests/test_env.py` (imports, `run_context`
fixture, `Config()` usage, `_NXGraph` typing guard, `monkeypatch` usage).

Verification: `pytest -q tests/test_evaluation.py tests/test_plotting.py` →
all pass; `pytest -q` → all pass (72 prior + new).

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `ruff check .` exits 0
- [ ] `ruff format --check .` exits 0
- [ ] `mypy --strict bike_rl` exits 0
- [ ] `pytest -q` exits 0; `tests/test_evaluation.py` and
      `tests/test_plotting.py` exist and all their tests pass
- [ ] `python -c "from bike_rl.evaluation import EvaluationTracker, evaluate_and_visualize; from bike_rl.plotting import render_solution_map, plot_metrics, plot_rewards, export_geojson; print('ok')"`
      exits 0 and prints `ok`
- [ ] `bike_rl/evaluation.py` does not contain `NotImplementedError("Plan 006")`
      (`grep -n 'NotImplementedError("Plan 006")' bike_rl/evaluation.py` → no
      matches)
- [ ] `bike_rl/plotting.py` does not contain `NotImplementedError("Plan 006")`
      (`grep -n 'NotImplementedError("Plan 006")' bike_rl/plotting.py` → no
      matches)
- [ ] `bike_rl/plotting.py` contains `matplotlib.use("Agg")` before
      `import matplotlib.pyplot as plt`
- [ ] `bike_rl/evaluation.py` does **not** import `bike_rl.objective` or
      `bike_rl.optim` (`grep -nE 'import (bike_rl\.objective|bike_rl\.optim)'
      bike_rl/evaluation.py` → no matches)
- [ ] `bike_rl/env.py`, `bike_rl/training.py`, `bike_rl/metrics.py`,
      `bike_rl/candidates.py`, `bike_rl/graph_utils.py`, `bike_rl/run_context.py`,
      `bike_rl/config.py`, `bike_rl/objective.py`, `bike_rl/optim/*`,
      `bike_rl/cli.py` are **unchanged**
      (`git diff --stat 289e069..HEAD -- bike_rl/env.py bike_rl/training.py
      bike_rl/metrics.py bike_rl/candidates.py bike_rl/graph_utils.py
      bike_rl/run_context.py bike_rl/config.py bike_rl/objective.py
      bike_rl/optim bike_rl/cli.py` shows no changes)
- [ ] No files outside the in-scope list are modified (`git status --short`
      lists only `bike_rl/evaluation.py`, `bike_rl/plotting.py`,
      `tests/test_evaluation.py`, `tests/test_plotting.py`, and
      `plans/README.md`).
- [ ] `plans/README.md` status row for 006 updated (TODO → DONE) and the
      dependency notes reflect `test_cli.py` moving to plan 007 — unless a
      reviewer told you they maintain the index.

## STOP conditions

Stop and report back (do not improvise) if:

- The code at the locations in "Current state" doesn't match the excerpts —
  e.g. `evaluation.py` / `plotting.py` already have real bodies,
  `BikePathEnv.__init__`/`reset`/`action_masks`/`step` signatures differ from
  those quoted, the observation is not the 4-component
  `[connectivity, efficiency, coverage, budget/initial]`, or `Config` lacks
  `seed`. The codebase has drifted since this plan was written; re-baseline
  before continuing.
- `BikePathEnv` no longer stores its live graph on `self._graph`, or the
  added edges are no longer tagged `bike_lane == "yes"` — the added-edge
  collection strategy breaks. Do **not** edit `env.py` from this plan; report
  it so plan 004 can be re-baselined or a public `added_edges()` accessor
  added there.
- `MaskablePPO.predict` does not accept `action_masks=` as a keyword argument,
  or its return shape is not `(action_array, state)` for a single env. Report
  the installed `sb3-contrib` version and the actual signature; do not invent
  a different API path.
- `matplotlib.use("Agg")` cannot be called before `import matplotlib.pyplot`
  in this environment, or `geopandas.GeoDataFrame.to_file(driver="GeoJSON")`
  fails due to a missing optional dependency (e.g. `fiona`). Report the
  versions; do not switch drivers without sign-off.
- `Config` turns out to lack a field this plan needs (only `seed` is
  expected). Do not add fields to `Config` from this plan without reporting —
  the plan is designed to need zero new fields.
- A step's verification fails twice after a reasonable fix attempt.
- The fix appears to require touching an out-of-scope file (e.g. you find
  `env.py` or `training.py` must be edited to make tests pass — they must
  not be).
- You discover that `evaluate_and_visualize` needs the canonical
  `objective.py` to score episodes — it does **not** (§3.8 scores by reward).
  If a reviewer/requirement insists on objective-based scoring, STOP and
  report; that is an optimiser-plan concern that depends on the (still-stub)
  `objective.py`.

## Maintenance notes

For the human/agent who owns this code after the change lands:

- **Scoring is reward-based (§3.8), not objective-based.** This plan
  deliberately does **not** import `bike_rl.objective`. The canonical-objective
  scoring of an RL policy (for the optimiser comparison table,
  OPTIMIZER_SPEC §3.2/§8) is delivered by `bike_rl.optim.evaluate.evaluate_rl_policy`
  in a later optimiser plan, which depends on `objective.py` being implemented.
  When that plan lands, it can reuse this plan's `_run_episode` rollout logic
  but score the resulting edge set with `objective(...)` instead of the env
  reward. Keep the rollout logic in `evaluation.py` reusable (do not couple it
  to the reward sum in a way that blocks objective rescoring).
- **`env._graph` is a read-only coupling.** `evaluation._run_episode` reads
  `env._graph` to collect added edges. A future refactor of `BikePathEnv`
  should add a public `added_edges() -> list[tuple]` accessor (returning the
  `bike_lane='yes'` edges not in the original) and `evaluation.py` should
  switch to it. Flag this in any env refactor review.
- **`test_cli.py` is deferred to plan 007.** This plan ships
  `test_evaluation.py` / `test_plotting.py` only. Plan 007 implements
  `cli.py` and writes `tests/test_cli.py` covering: `--city`/`--bbox` mutual
  exclusivity, `--skip_training` loads a model, `--no_plots` skips plotting,
  `--show` / `--export_geojson` plumbing — all with OSM + PPO mocked. The
  `plans/README.md` index is updated to reflect this split.
- **Plotting is Agg-first.** `plotting.py` calls `matplotlib.use("Agg")` at
  import. Any module that imports `bike_rl.plotting` gets the Agg backend. If
  a future caller needs an interactive backend, they must set the backend
  **before** importing `bike_rl.plotting` — document this. The SLURM scripts
  (plan 007) also set `MPLBACKEND=Agg` for redundancy.
- **GeoJSON export uses geopandas + shapely.** The `export_geojson` function
  builds LineStrings from node `x`/`y`. For real OSM graphs, edges may carry
  geometries; this plan uses straight lines between endpoints, which is
  sufficient for a "suggested paths" overlay. A future enhancement can read
  OSMnx edge geometries if richer lines are needed.
- **Reviewer scrutiny.** Check that (a) `evaluate_and_visualize` selects best
  by **reward** and records `best_added_edges` only for the best episode;
  (b) the rollout passes `action_masks=env.action_masks()` to `model.predict`
  (masking is honoured during evaluation, not just training); (c) `no_plots`
  short-circuits **all** rendering including GeoJSON; (d) every plotting
  function gates `plt.show()` behind `show` and closes its figure
  (`plt.close(fig)`) to avoid memory leaks across many runs; (e) no `print`
  calls (use `logger.info`); (f) no bare `except`; (g) `evaluation.py` does
  not import `objective` or `optim`.
- **Follow-up deferred:** `cli.py` + SLURM scripts + `tests/test_cli.py`
  (plan 007); the full `ruff`/`mypy`/`pytest --cov` ≥80% pass (plan 008);
  end-to-end smoke test on a tiny bbox (plan 009). The CLI wires
  `load_*_graph` → `make_vec_env` → `train_model` → `evaluate_and_visualize`,
  passing `--budget`, `--show`, `--export_geojson`, `--no_plots` through.
