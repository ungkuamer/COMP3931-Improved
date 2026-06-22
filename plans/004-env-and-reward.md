# Plan 004: Implement `env.py` — MaskablePPO action masking, fixed reward, fixed state updates + `test_env.py` / `test_reward.py`

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**:
> `git diff --stat 06a124d..HEAD -- bike_rl/env.py bike_rl/config.py tests/conftest.py tests/test_env.py tests/test_reward.py`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: L
- **Risk**: MED
- **Depends on**: plans/003-metrics-and-tests.md (DONE — provides `bike_rl.metrics.MetricsState` and the four pure metric functions with the §5.7 coverage fix)
- **Category**: bug (§5.4 spurious termination, §5.5 no-op reward, §5.6 stale state, §5.12 dropped episode) + tech-debt (modular `BikePathEnv`) + tests
- **Planned at**: commit `06a124d`, 2026-06-22
- **Issue**: (not published)

## Why this matters

The original `nx-rx-simple-ur.py` `BikePathEnv` has four bugs that between
them make the RL agent learn the wrong thing or barely learn at all:

1. **§5.4 — spurious termination.** The action space is
   `Discrete(initial_candidate_count)` but candidates shrink every step, so
   any PPO action ≥ the current candidate count triggers
   `return …, -10, True, …`. The agent learns to terminate early. This is the
   core RL bug.
2. **§5.5 — budget-efficiency reward is a no-op.** The original term
   simplifies to ≈0, so the agent gets no signal for spending budget well.
3. **§5.6 — incremental state update leaves connectivity & population stale**
   for the most important (connected) additions, so those reward terms are
   always 0 exactly when they should fire.
4. **§5.12 — the final episode is dropped** from `episode_rewards` because it
   is only appended at the next `reset()`.

This plan builds `bike_rl/env.py` — the Gymnasium environment that plan 005
(`training.py`) wraps in `SubprocVecEnv` and trains with `MaskablePPO`. It
delivers: (1) a **fixed-size action space** with a correct `action_masks()`
so `MaskablePPO` (from `sb3-contrib`) never sees a "terminate on invalid
action" signal — invalid/masked actions cost reward 0 and do **not** end the
episode; (2) a **meaningful budget-efficiency reward term** (Config-weighted,
capped) that is non-zero whenever a positive-gain edge is added; (3)
**correct per-step state** by always reading the four metrics from
`MetricsState` (incremental + version-cached, so "always full" is also
cheap — §5.6/§5.14); (4) **episode info emitted in `step`'s `info` dict** on
termination (the SB3 `info["episode"]` convention) so plan 005's callback
never drops the last episode; (5) logging instead of `print`, no bare
`except`. `training.py`, `evaluation.py`, `plotting.py`, `cli.py` are
**out of scope** — they are plans 005–007.

## Current state

The package was scaffolded (plan 001), `graph_utils.py`/`candidates.py`
implemented (plan 002), and `metrics.py` implemented (plan 003). The relevant
files:

- `bike_rl/env.py` — **stub**. A `BikePathEnv(gym.Env[object, object])` with
  `__init__`, `step`, `reset`, `action_masks` all raising
  `NotImplementedError("Plan 004")`. The `__init__` signature is already
  fixed and **must be preserved** (plan 005's `make_env` will call it):
  ```python
  def __init__(
      self,
      bike_graph: object,
      walk_graph: object,
      cfg: Config,
      run_context: RunContext,
  ) -> None:
  ```
  (`Config` and `RunContext` are imported under `TYPE_CHECKING` — keep that.)
- `bike_rl/config.py` — frozen `Config`, fully implemented. Has every reward
  weight this plan needs **except** the budget-efficiency fields and a step
  cap (see Step 1, which adds exactly three fields). Existing fields used
  here (verified): `w_connectivity=0.9`, `w_efficiency=0.2`,
  `w_population=0.3`, `reward_scale=100.0`, `road_priority_scale=5.0`,
  `continuity_bonus=50.0`, `fragmentation_weight=200.0`,
  `isolation_penalty=-100.0`, `edge_cost_factor=10.0`,
  `max_episode_steps` (to be added), `seed=0`.
- `bike_rl/metrics.py` — implemented. **This is the dependency you build on.**
  Import and use exactly these:
  ```python
  from bike_rl.metrics import MetricsState
  st = MetricsState(graph, cfg)          # copies graph; builds PyGraph + UnionFind
  st.add_edge(u, v, data: dict)          # record an added edge; bumps internal version (§5.14)
  st.connectivity()      -> float in [0,1]   # cached on version
  st.path_efficiency()   -> float in (0,1]   # cached on version
  st.fragmentation()     -> float in [0,1]   # incremental via UnionFind (§6.3)
  st.coverage()          -> float in [0,1]   # §5.7 fix (sensitive to additions)
  ```
  `MetricsState.__init__` **copies** the graph (`graph.copy()`); it does not
  take ownership. `add_edge` mutates `MetricsState`'s internal copy only —
  to keep the env's own `self.graph` consistent, the env must also
  `self.graph.add_edge(u, v, **data)` itself (see Step 3). Do not re-implement
  metrics in `env.py`; call `MetricsState`.
- `bike_rl/candidates.py` — implemented. Use:
  ```python
  from bike_rl.candidates import Candidate, candidate_cost, extract_candidates
  candidates = extract_candidates(bike_graph, walk_graph, cfg)  # sorted, fresh dataclasses, no source mutation (§5.8)
  cost = candidate_cost(c, cfg)    # c.length * cfg.edge_cost_factor
  ```
  `Candidate` is a **frozen dataclass**; identity is `(u, v, length,
  road_priority)` (the `connects_to_bike_path` flag and `data` dict are
  excluded from equality — `field(compare=False)`). This means you can
  replace a slot's `Candidate` with one whose `connects_to_bike_path` flag
  differs and it stays "the same" candidate for set/list membership. The env
  does **not** need to mutate candidates' flags: re-derive `connects_to_bike_path`
  at step time from `self._bike_nodes` (§5.8) — see Step 3.
- `bike_rl/run_context.py` — `RunContext` (frozen dataclass: `run_id`,
  `output_dir: Path`, `timestamp`). The env receives it but does not need to
  write files (that's evaluation/plotting, plans 006). Store it on `self`
  for future use and for logging context; do not create directories here.
- `tests/conftest.py` — has `tiny_bike_graph` (4 nodes, edges `(1,2)` and
  `(3,4)` both `bike_lane='yes'`, lengths 120/130, `x`/`y` ~0.001 apart),
  `tiny_walk_graph` (5 nodes; candidates `(2,5)` primary len 200, `(3,5)`
  secondary len 150, `(4,5)` tertiary len 600, plus skipped edges), and
  `mock_osm`. **Do not modify these fixtures.** The env tests will
  construct envs directly from `tiny_bike_graph` + `tiny_walk_graph`.
- `bike_rl/objective.py` — stub, raises `NotImplementedError`. **Do not
  touch.** The env's reward is its own (derived from the spec's §3.5
  formula), not `objective()`; `objective.py` is the optimiser-side single
  source of truth and is implemented in an optimiser plan.
- `bike_rl/training.py`, `evaluation.py`, `plotting.py`, `cli.py` — stubs.
  Out of scope. Do not wire `env` into them.

### Existing tests (must stay green)

`pytest -q` currently passes 38 tests: `test_smoke.py` (3), `test_graph_utils.py`,
`test_candidates.py`, `test_metrics.py`. Do not modify any existing test file.
Only **add** `tests/test_env.py` and `tests/test_reward.py`.

### Repo conventions to match (still in force from plans 001–003)

- **Typing**: `from __future__ import annotations` at top of every module;
  `mypy --strict bike_rl` must stay clean. `pyproject.toml` already sets
  `ignore_missing_imports = true` for `gymnasium`? — **it does not**, but
  gymnasium ships type stubs, so `import gymnasium as gym` type-checks fine.
  `sb3_contrib` has `ignore_missing_imports` set, but **env.py must not
  import `sb3_contrib`** — MaskablePPO is only needed in `training.py`
  (plan 005). The env just exposes `action_masks()`; that is the whole
  contract.
- **Imports**: stdlib → third-party → local, blank-line separated (ruff `I`).
  Guard `Config` / `RunContext` under `TYPE_CHECKING` if only used in
  annotations (the stub already does this — keep it). `MetricsState`,
  `Candidate`, `candidate_cost`, `extract_candidates` are used at runtime →
  import normally.
- **Docstrings**: Google convention (ruff `D`, `convention = "google"`).
  Public class/functions get docstrings. `D100/D102/D104/D105/D107` ignored.
- **No `print`**: use `logging` (`logger = logging.getLogger(__name__)`).
  No prints in `env.py`.
- **No bare `except`** (§5.11): use `except Exception as e:` and log via
  `logger.exception(...)` if you catch at all. Prefer not catching — let
  errors propagate during development. The only justified catch is around
  metric computation if you want the env to survive a transient rustworkx
  error; if you add one, log it. (The reference impl's bare `except:` blocks
  are gone in the modular design — do not reintroduce them.)
- **Frozen dataclasses** for value types. `BikePathEnv` is a `gym.Env`
  subclass (mutable by nature) — plain class, not a dataclass.
- **Tests**: `pytest`, fixtures from `tests/conftest.py`, no network access.
  Model test file structure on `tests/test_candidates.py` (top docstring,
  `from __future__ import annotations`, small focused functions with
  docstrings). Use a local `_make_env()` helper in each test file to build a
  configured env from the conftest fixtures (see Step 4).
- **Commit style**: Conventional Commits with scope — `feat(env): …`,
  `test(env): …`, matching `git log` entries like
  `feat(metrics): implement pure metric functions …`.

### Gymnasium / MaskablePPO API facts (verified — use exactly these)

The executor must use these exact shapes; gymnasium is picky:

- Installed: `gymnasium==1.3.0` (verified via `python -c "import gymnasium;
  print(gymnasium.__version__)"`).
- `gym.Env` subclass contract:
  - Call `super().__init__()` first in `__init__`.
  - Set `self.observation_space` and `self.action_space` as **instance**
    attributes (not class attributes) after `super().__init__()`.
  - `reset(self, *, seed: int | None = None, options: dict | None = None)
    -> tuple[np.ndarray, dict[str, Any]]`. Call `super().reset(seed=seed)`
    first (this seeds `self.np_random`); then rebuild state and return
    `(obs, info)`. Do **not** return `info` with `seed` semantics — just the
    fresh observation and an empty-or-diagnostic info dict.
  - `step(self, action) -> tuple[np.ndarray, float, bool, bool, dict[str,
    Any]]` = `(obs, reward, terminated, truncated, info)`. `terminated` =
    the episode ended naturally (no valid actions / budget exhausted);
    `truncated` = hit the step cap (only if `max_episode_steps > 0`).
    Return a Python `float` for reward (satisfies `SupportsFloat`).
- `gym.spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)` for the
  observation. **dtype must be `np.float32`** and `step`/`reset` must
  return an `np.ndarray` of that dtype (cast with `np.asarray(...,
  dtype=np.float32)`).
- `gym.spaces.Discrete(n)` for the action space. `n` is **fixed at
  `__init__` time** = the initial candidate count (padded to ≥1). `reset`
  must **not** change `self.action_space` — MaskablePPO caches the action
  space shape at env wrapping; changing it across episodes breaks it.
- **MaskablePPO contract** (from `sb3-contrib`, verified importable):
  the env only needs to expose a method
  `action_masks(self) -> np.ndarray` returning a boolean array of shape
  `(self.action_space.n,)` where `True` = action is legal now. No base
  class, no registration, no `gym.register` needed. `env.py` does **not**
  import `sb3_contrib`. The mask dtype should be `bool`; cast with
  `np.asarray(..., dtype=bool)`.
- Do **not** call `gym.envs.registration.register` or add the env to the
  registry — plan 005 instantiates `BikePathEnv` directly.

### Intent docs to honor (quoted so you don't re-read them)

From `RECREATE_SPEC.md` §3.4 (env behaviour to preserve):
> **Action space**: `Discrete(len(candidate_edges))` …
> **`step(action)`**: … If `cost > budget`: pop the action, return reward −1,
> maybe done. … Add edge to `self.graph` with `bike_lane='yes'`; also add to
> `self.rx_graph`. … `done` when budget < cheapest remaining edge cost or no
> candidates left.
> **`reset()`**: restore `graph = original_graph.copy()`, rebuild rx graph,
> rebuild candidates, reset budget/steps, return state.

From `RECREATE_SPEC.md` §3.5 (reward formula — the structure to keep; the
budget term is the §5.5 fix):
> state_improvement = (new-old)*w for (connectivity, efficiency, population)
>   with w = [0.9, 0.2, 0.3]
> reward = sum(state_improvement) * 100
>   + road_priority * 5
>   + 50  if connects_to_bike_path
>   + (old_frag - new_frag) * 200
>   - 100 if would_create_isolated
>   + budget_efficiency term  (currently a no-op — §5.5)

From `RECREATE_SPEC.md` §5.4 (the core RL bug):
> use **`MaskablePPO` from `sb3-contrib`** with `env.action_masks()`
> returning a boolean mask over the *fixed* action space of valid
> (budget-affordable, not-yet-used) candidates. Never terminate on an
> invalid/masked action; instead pad the action space to the initial size
> and mask invalid indices. Invalid actions should yield reward 0 and **not**
> end the episode.

From `RECREATE_SPEC.md` §5.5 (the budget fix):
> define a meaningful cost-effectiveness term, e.g. `reward +=
> connectivity_gain / cost`, or normalise `cost` against `initial_budget`:
> `reward += w * (state_gain - cost/initial_budget)`. Make weights live in
> `Config`.

From `RECREATE_SPEC.md` §5.6 (the stale-state fix):
> maintain incremental connectivity & population updates, **or** always run
> full `_get_state()` but make it cheap via incremental data structures
> (§6.2–6.3). Prefer correctness over the fragile incremental shortcut.

From `RECREATE_SPEC.md` §5.8 (candidate freshness):
> Re-derive `connects_to_bike_path` from current `bike_nodes` at step time.

From `RECREATE_SPEC.md` §5.12 (episode bookkeeping):
> Append in a proper `_on_episode_end` hook (or in `step` when `done`).

From `RECREATE_SPEC.md` §8 (no globals):
> Replace `RUN_ID` / `RUN_FIGURES_DIR` with a `RunContext` … explicitly
> passed to training/evaluation/plotting. SubprocVecEnv workers receive
> serialised config, not globals.

From `RESEARCH_DIRECTION.md` §4.4 (determinism): solvers must be seeded and
deterministic. The env must therefore be deterministic given a fixed
`Config.seed` and action sequence — no unseeded global RNG in `env.py`.
(Metric sampling is already seeded via `cfg.seed` inside `MetricsState`.)

## Commands you will need

| Purpose    | Command                                                                 | Expected on success |
|------------|-------------------------------------------------------------------------|---------------------|
| Install    | `pip install -e ".[dev]"`                                               | exit 0              |
| Ruff lint  | `ruff check bike_rl/env.py bike_rl/config.py tests/test_env.py tests/test_reward.py` | exit 0, no errors |
| Ruff fmt   | `ruff format --check bike_rl tests`                                     | exit 0              |
| Mypy       | `mypy --strict bike_rl`                                                 | exit 0, no errors   |
| Env tests  | `pytest -q tests/test_env.py`                                           | all pass            |
| Reward tests | `pytest -q tests/test_reward.py`                                     | all pass            |
| Full suite | `pytest -q`                                                             | all pass (43 prior + 19 new = 62) |
| Coverage   | `pytest --cov=bike_rl --cov-report=term-missing tests/test_env.py tests/test_reward.py` | exit 0; `bike_rl/env.py` ≥80% line coverage |

## Suggested executor toolkit

- Verify the installed versions once before coding:
  `python -c "import gymnasium; print(gymnasium.__version__)"`
  (expect `1.3.0`); `python -c "from sb3_contrib import MaskablePPO;
  print('ok')"` (expect `ok` — this only confirms the training dependency
  exists; **do not import it in env.py**).
- If a gymnasium signature below doesn't match 1.3.0, STOP and report — do
  not guess a different API.

## Scope

**In scope** (the only files you should modify or create):
- `bike_rl/config.py` — **add exactly three fields** to `Config` (Step 1):
  `w_budget_efficiency`, `budget_efficiency_cap`, `max_episode_steps`. No
  other changes to `config.py`. Do not rename or reorder existing fields.
- `bike_rl/env.py` — full implementation of `BikePathEnv`.
- `tests/test_env.py` — create.
- `tests/test_reward.py` — create.

**Out of scope** (do NOT touch, even though they look related):
- `bike_rl/metrics.py` — done in plan 003. Import `MetricsState` from it; do
  not edit it. If you find a metric misbehaves, STOP and report — do not
  patch `metrics.py` from this plan.
- `bike_rl/candidates.py`, `bike_rl/graph_utils.py` — done in plan 002.
  Import from them; do not edit.
- `bike_rl/run_context.py`, `bike_rl/objective.py` — leave exactly as is.
- `bike_rl/training.py`, `bike_rl/evaluation.py`, `bike_rl/plotting.py`,
  `bike_rl/cli.py`, anything under `bike_rl/optim/` — plans 005+ and the
  optimiser plans. Do not wire `env` into them, do not implement `make_env`,
  `train_model`, or any callback here.
- `tests/conftest.py` and the existing `tests/test_*.py` files — do not
  modify. Add new fixtures **inside** `tests/test_env.py` /
  `tests/test_reward.py` (local fixtures keep conftest stable).
- `pyproject.toml`, `requirements*.txt`, CI, `.gitignore`.

## Git workflow

- Branch: `advisor/004-env-and-reward`
- Commit per logical unit (suggested: one for Config fields, one for the env
  implementation, one for each test file). Conventional Commits with scope:
  `feat(config): add budget-efficiency + step-cap fields`,
  `feat(env): implement BikePathEnv with MaskablePPO action masking (§5.4/§5.5/§5.6/§5.12)`,
  `test(env): add env behaviour tests`,
  `test(reward): pin §5.5 budget-efficiency + continuity + isolation terms`.
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Add three Config fields

Edit `bike_rl/config.py`. Add three fields to the `Config` dataclass and
three lines to its docstring `Attributes:` block. Place the new fields in
the **Reward weights** group (for `w_budget_efficiency`) and in the
**Cost / budget** group (for `budget_efficiency_cap`) and a new line in the
existing `max_episode_steps`-ish area — concretely, add them right after the
existing `isolation_penalty` / `edge_cost_factor` lines, and update the
docstring. The exact insertion points:

After the line `isolation_penalty: float = -100.0` add:

```python
    w_budget_efficiency: float = 0.1
```

After the line `edge_cost_factor: float = 10.0` add:

```python
    budget_efficiency_cap: float = 200.0
    max_episode_steps: int = 0
```

And add three lines to the `Attributes:` docstring, near the matching
existing entries (keep alphabetical-ish / grouped ordering with the fields):

```python
        w_budget_efficiency: Weight on the budget-efficiency reward term (§5.5).
        budget_efficiency_cap: Upper clamp on the budget-efficiency term (§5.5).
        max_episode_steps: Step cap before truncation (0 = disabled; §3.4 done-by-budget only).
```

Rationale (so the executor understands, not improvises):
- `w_budget_efficiency` and `budget_efficiency_cap` make the §5.5 fix
  Config-driven (spec §5.5: "Make weights live in `Config`"). Defaults are
  chosen so the term is a moderate bonus, not a blow-up: with `reward_scale
  = 100`, a positive state gain of ~0.01–0.1 → `gain_scaled ~ 1–10`, and
  `cost/initial_budget ~ 0.02` (a 200 m edge, cost 2000, budget 100000) →
  raw term `0.1 * 10 / 0.02 = 50`, well under the 200 cap. The cap only
  bites for tiny costs on huge budgets.
- `max_episode_steps = 0` means **truncation is disabled by default**; the
  episode ends only when budget is exhausted or no candidate is affordable
  (preserving the original §3.4 `done` semantics). A non-zero value lets
  plan 005 bound episode length for training stability.

**Verify**:
- `ruff check bike_rl/config.py` → exit 0
- `ruff format --check bike_rl/config.py` → exit 0 (run `ruff format bike_rl/config.py` if needed)
- `mypy --strict bike_rl/config.py` → exit 0
- `python -c "from bike_rl.config import Config; c=Config(); print(c.w_budget_efficiency, c.budget_efficiency_cap, c.max_episode_steps)"` → prints `0.1 200.0 0`
- `pytest -q tests/test_smoke.py` → 3 pass (the existing `test_config_defaults` must still pass — it only asserts on pre-existing fields, so adding fields is safe)

### Step 2: Implement `BikePathEnv` core — `__init__`, `reset`, `_get_observation`, `action_masks`, slot management

Replace the entire contents of `bike_rl/env.py` with the implementation
below (adapt import ordering to satisfy ruff `I`; the structure is
load-bearing). Read the comments — they encode the spec references.

```python
"""Gymnasium RL environment for budgeted bike-network expansion.

See RECREATE_SPEC.md §3.4 (behaviour), §5.4 (action masking),
§5.5 (budget-efficiency reward), §5.6 (state updates), §5.11 (logging),
§5.12 (episode bookkeeping), §5.14 (cache versioning via MetricsState).

The env exposes ``action_masks()`` so ``sb3-contrib``'s ``MaskablePPO``
(plan 005) can mask invalid (already-used or unaffordable) actions over a
**fixed-size** action space. This module does **not** import ``sb3_contrib``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, SupportsFloat

import gymnasium as gym
import networkx as nx
import numpy as np

from bike_rl.candidates import Candidate, candidate_cost, extract_candidates
from bike_rl.metrics import MetricsState

if TYPE_CHECKING:
    from bike_rl.config import Config
    from bike_rl.run_context import RunContext

    _NXGraph = nx.MultiDiGraph[Any, Any, Any]
else:
    _NXGraph = nx.MultiDiGraph

logger = logging.getLogger(__name__)


class BikePathEnv(gym.Env[np.ndarray, np.int64]):
    """Bike-path expansion env with MaskablePPO action masking (RECREATE_SPEC §3.4/§5.4).

    Observation: ``Box(0, 1, (4,), float32)`` =
    ``[connectivity, path_efficiency, coverage, budget/initial_budget]``.
    Action space: ``Discrete(n_initial_candidates)`` — **fixed** for the life
    of the instance (§5.4). ``action_masks()`` returns a boolean mask over
    that fixed space; ``True`` = the candidate at that index is still unused
    *and* affordable with the remaining budget. Invalid/masked actions yield
    reward 0 and do **not** terminate the episode (§5.4).
    """

    def __init__(
        self,
        bike_graph: _NXGraph,
        walk_graph: _NXGraph,
        cfg: Config,
        run_context: RunContext,
    ) -> None:
        """Store inputs and build the fixed action space; call :meth:`reset` is NOT done here.

        Args:
            bike_graph: The existing bike network (edges with ``bike_lane='yes'``).
                Copied internally; not mutated.
            walk_graph: The walkable network candidates are drawn from.
                Not mutated (§5.8 — ``extract_candidates`` copies edge data).
            cfg: Config (reward weights, cost factor, step cap, seed).
            run_context: Per-run output context (stored for logging/future
                use by evaluation; the env does not create directories).
        """
        super().__init__()
        self._bike_graph_original = bike_graph.copy()
        self._walk_graph = walk_graph
        self._cfg = cfg
        self._run_context = run_context

        # Fixed action space = initial candidate count (padded to ≥1) — §5.4.
        initial_candidates = extract_candidates(bike_graph, walk_graph, cfg)
        self._n_actions = max(1, len(initial_candidates))
        self.action_space = gym.spaces.Discrete(self._n_actions)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Episode state (initialised properly in reset).
        self._initial_budget: float = 0.0
        self._budget: float = 0.0
        self._steps: int = 0
        self._graph: _NXGraph = nx.MultiDiGraph()
        self._metrics: MetricsState | None = None
        self._slots: list[Candidate | None] = []
        self._bike_nodes: set[int | str] = set()
        self._last_obs: np.ndarray = np.zeros(4, dtype=np.float32)
        self._last_frag: float = 0.0
        self._episode_reward: float = 0.0

    # ── Episode lifecycle ────────────────────────────────────────────────

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the episode: fresh graph copy, full budget, all slots available.

        Args:
            seed: Optional seed (passed to ``super().reset``; the env is
                otherwise deterministic given ``cfg``).
            options: Optional dict; recognised key ``"budget"`` (float)
                overrides the default initial budget = ``cfg`` has no budget
                field, so the default is read from ``options["budget"]`` or
                falls back to ``self._initial_budget`` if already set. The
                CLI/training (plan 005/007) passes the budget via options.

        Returns:
            ``(observation, info)`` with an empty info dict.
        """
        super().reset(seed=seed)
        cfg = self._cfg
        self._graph = self._bike_graph_original.copy()
        self._metrics = MetricsState(self._graph, cfg)
        candidates = extract_candidates(self._bike_graph_original, self._walk_graph, cfg)
        # Pad slots to the fixed action-space size with None.
        self._slots = list(candidates) + [None] * (self._n_actions - len(candidates))
        self._bike_nodes = {
            n
            for u, v, d in self._graph.edges(data=True)
            if d.get("bike_lane") == "yes"
            for n in (u, v)
        }
        if options is not None and "budget" in options:
            self._initial_budget = float(options["budget"])
        if self._initial_budget <= 0.0:
            # Fallback when no budget has been configured: a large default so
            # tiny fixtures still terminate via candidate exhaustion. The
            # CLI always sets a real budget (plan 007).
            self._initial_budget = 100_000.0
        self._budget = self._initial_budget
        self._steps = 0
        self._episode_reward = 0.0
        self._last_obs = self._get_observation()
        self._last_frag = self._metrics.fragmentation()
        return self._last_obs.copy(), {}

    def _get_observation(self) -> np.ndarray:
        """Return ``[connectivity, efficiency, coverage, budget/initial]`` as float32.

        All four components are in ``[0, 1]``. Reads from ``self._metrics``
        (incremental + version-cached — §5.6/§5.14) so a full state read is
        cheap and always consistent.
        """
        assert self._metrics is not None
        cfg = self._cfg
        conn = self._metrics.connectivity()
        eff = self._metrics.path_efficiency()
        pop = self._metrics.coverage()
        norm_budget = self._budget / self._initial_budget if self._initial_budget > 0 else 0.0
        norm_budget = max(0.0, min(1.0, norm_budget))
        obs = np.asarray([conn, eff, pop, norm_budget], dtype=np.float32)
        return obs

    def action_masks(self) -> np.ndarray:
        """Boolean mask of shape ``(action_space.n,)``: legal = unused & affordable.

        ``True`` at index ``i`` iff ``self._slots[i]`` is a candidate that has
        not been used yet and whose cost fits the remaining budget. This is
        the MaskablePPO contract (§5.4).
        """
        cfg = self._cfg
        mask = np.zeros(self._n_actions, dtype=bool)
        for i, slot in enumerate(self._slots):
            if slot is None:
                continue
            if candidate_cost(slot, cfg) <= self._budget + 1e-9:
                mask[i] = True
        return mask
```

Notes for the executor:
- `gym.Env[np.ndarray, np.int64]` — the generic parameters are observation
  and action types. mypy strict accepts this; if it complains about
  `np.int64` not matching the base's `ActType`, use `gym.Env[np.ndarray,
  np.intp]` or fall back to `gym.Env[object, object]` (the stub's choice).
  Prefer the typed version; only downgrade if mypy errors.
- The `1e-9` tolerance in `action_masks` avoids floating-point churn where
  `cost == budget` exactly.
- `reset` reads the budget from `options["budget"]` because `Config` has no
  budget field (the budget is a per-run CLI argument, §3.9). Plan 005's
  `make_env` will pass `options={"budget": cfg_budget}` on reset (or the
  SubprocVecEnv wrapper handles the first reset). For the tests in this
  plan, always pass `options={"budget": …}` explicitly.
- Do **not** call `reset()` inside `__init__` — Gymnasium lets the user
  call `reset` themselves, and SubprocVecEnv calls it on first use. But
  some SB3 helpers call `env.reset()` immediately after construction; to
  be safe, initialise `_initial_budget` lazily in `reset` as shown (the
  fields are set to safe defaults in `__init__`).

**Verify**:
- `ruff check bike_rl/env.py` → exit 0
- `ruff format --check bike_rl/env.py` → exit 0 (run `ruff format bike_rl/env.py` if it reformats)
- `mypy --strict bike_rl/env.py` → exit 0
- `python -c "
import networkx as nx
from bike_rl.config import Config
from bike_rl.run_context import RunContext
from bike_rl.env import BikePathEnv
from pathlib import Path
bg = nx.MultiDiGraph(); bg.add_node(1,x=0.,y=0.); bg.add_node(2,x=0.001,y=0.); bg.add_node(3,x=0.,y=0.001); bg.add_node(4,x=0.001,y=0.001)
bg.add_edge(1,2,length=120.0,highway='residential',bike_lane='yes'); bg.add_edge(3,4,length=130.0,highway='residential',bike_lane='yes')
wg = nx.MultiDiGraph()
for n,(x,y) in {1:(0.,0.),2:(0.001,0.),3:(0.,0.001),4:(0.001,0.001),5:(0.002,0.002)}.items(): wg.add_node(n,x=x,y=y)
wg.add_edge(2,5,length=200.0,highway='primary'); wg.add_edge(3,5,length=150.0,highway='secondary'); wg.add_edge(4,5,length=600.0,highway='tertiary')
rc = RunContext(run_id='t', output_dir=Path('/tmp/t'), timestamp=__import__('datetime').datetime.now(__import__('datetime').timezone.utc))
env = BikePathEnv(bg, wg, Config(), rc)
obs, info = env.reset(options={'budget': 100_000.0})
print('obs', obs, 'mask', env.action_masks(), 'n', env.action_space.n)
"` → prints an obs array of 4 floats in `[0,1]`, a boolean mask with at least one `True`, and `n` = the candidate count (3 for this fixture: `(2,5)`, `(3,5)`, `(4,5)`).

### Step 3: Implement `step` — reward (§5.5), full state (§5.6), masking (§5.4), episode info (§5.12)

Add the `step` method to `BikePathEnv` (and a small `_apply_edge` helper).
The reward formula follows §3.5 with the §5.5 budget-efficiency fix and the
§5.8 step-time `connects_to_bike_path` re-derivation.

```python
    def step(
        self, action: np.int64 | int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply one bike-lane addition (RECREATE_SPEC §3.4/§3.5/§5.4/§5.5/§5.6/§5.12).

        - If ``action`` is masked (used or unaffordable): reward 0, no state
          change, episode continues (§5.4).
        - Otherwise: add the edge to the graph and to ``MetricsState``,
          deduct cost, recompute the full state from ``MetricsState`` (§5.6 —
          always full, cheap via incremental structures + version cache
          §5.14), and compute the reward.
        - ``terminated`` when no action is legal afterwards (budget exhausted
          or all candidates used). ``truncated`` when ``max_episode_steps >
          0`` and the step cap is hit.
        - On termination, emit SB3-style episode info in ``info["episode"]``
          so the training callback (plan 005) records every episode,
          including the last (§5.12).
        """
        cfg = self._cfg
        assert self._metrics is not None

        mask = self.action_masks()
        # ── Invalid / masked action (§5.4): no-op, reward 0, not done ──
        if action < 0 or action >= self._n_actions or not bool(mask[action]):
            logger.debug("masked action %s ignored", action)
            obs = self._get_observation().copy()
            info: dict[str, Any] = {"invalid_action": True}
            self._steps += 1
            truncated = self._truncated_now()
            # An all-masked state should already have terminated last step;
            # but if the env is driven into a no-legal-action state, terminate now.
            terminated = not bool(mask.any())
            if terminated:
                info = {**info, **self._episode_info()}
            return obs, 0.0, terminated, truncated, info

        candidate = self._slots[action]
        assert candidate is not None
        cost = candidate_cost(candidate, cfg)

        # ── State before ──
        old_obs = self._last_obs
        old_frag = self._last_frag

        # ── Apply the edge to the env graph AND to MetricsState ──
        self._apply_edge(candidate)

        # ── State after (full read — §5.6; cheap via MetricsState — §5.14) ──
        new_obs = self._get_observation()
        new_frag = self._metrics.fragmentation()

        # ── Reward (§3.5 structure + §5.5 fix) ──
        wc, we, wp = cfg.reward_weights()
        state_gain = (
            (float(new_obs[0]) - float(old_obs[0])) * wc
            + (float(new_obs[1]) - float(old_obs[1])) * we
            + (float(new_obs[2]) - float(old_obs[2])) * wp
        )
        connects = (candidate.u in self._bike_nodes_before) or (
            candidate.v in self._bike_nodes_before
        )
        would_create_isolated = (
            candidate.u not in self._bike_nodes_before
            and candidate.v not in self._bike_nodes_before
        )

        reward = state_gain * cfg.reward_scale
        reward += candidate.road_priority * cfg.road_priority_scale
        if connects:
            reward += cfg.continuity_bonus
        reward += (old_frag - new_frag) * cfg.fragmentation_weight
        if would_create_isolated:
            reward += cfg.isolation_penalty  # negative (§3.5)

        # Budget-efficiency term (§5.5 fix — non-zero for positive gain + cost).
        gain_scaled = max(0.0, state_gain) * cfg.reward_scale
        if cost > 0 and self._initial_budget > 0:
            frac = cost / self._initial_budget
            budget_term = cfg.w_budget_efficiency * gain_scaled / frac
            reward += min(budget_term, cfg.budget_efficiency_cap)

        # ── Bookkeeping ──
        self._last_obs = new_obs
        self._last_frag = new_frag
        self._episode_reward += reward
        self._steps += 1

        terminated = not bool(self.action_masks().any())
        truncated = self._truncated_now()
        info = {}
        if terminated:
            info = {**info, **self._episode_info()}
        return new_obs.copy(), float(reward), terminated, truncated, info

    def _apply_edge(self, candidate: Candidate) -> None:
        """Add the candidate edge to the env graph + MetricsState; update bike_nodes.

        Records ``connects_to_bike_path`` from the bike-node set *before* the
        add (stored on ``self._bike_nodes_before``) so the reward's continuity
        / isolation tests see the pre-add network state (§5.8/§3.5).
        """
        assert self._metrics is not None
        self._bike_nodes_before = set(self._bike_nodes)
        u, v = candidate.u, candidate.v
        data = dict(candidate.data)
        data["bike_lane"] = "yes"
        data["length"] = candidate.length
        self._graph.add_edge(u, v, **data)
        self._metrics.add_edge(u, v, data)
        # Ensure both endpoints exist as nodes in the env graph (they should
        # already, from the walk graph; this is a no-op safety).
        for n in (u, v):
            self._bike_nodes.add(n)
        # Mark the slot used (fixed action-space index — §5.4).
        idx = self._slots.index(candidate)
        self._slots[idx] = None

    def _truncated_now(self) -> bool:
        """True iff the step cap is enabled and reached."""
        return self._cfg.max_episode_steps > 0 and self._steps >= self._cfg.max_episode_steps

    def _episode_info(self) -> dict[str, Any]:
        """SB3-style episode summary for the ``info`` dict (§5.12)."""
        return {"episode": {"r": float(self._episode_reward), "l": int(self._steps)}}
```

Important notes for the executor (read these carefully — they are
load-bearing):

1. **`self._slots.index(candidate)`** works because `Candidate` equality is
   `(u, v, length, road_priority)` (plan 002's `field(compare=False)` on the
   flag and `data`). The candidate you pass to `_apply_edge` is the exact
   object stored in the slot, so `index` finds it by identity first anyway.
   This is O(n) per step — acceptable for now; the §6.4 heap optimisation is
   **out of scope** for this plan (see Maintenance notes). Do not "optimise"
   by re-sorting slots — the action-space indices must stay stable (§5.4).

2. **`_bike_nodes_before`** is set inside `_apply_edge` **before** the node
   set is updated. The reward computation in `step` reads it after calling
   `_apply_edge`. This ordering is deliberate: `connects_to_bike_path` and
   `would_create_isolated` are defined relative to the network *before* this
   edge was added (§3.5/§5.8). Do not reorder.

3. **Episode info on termination only** (§5.12): `info["episode"]` is added
   when `terminated` is true. Plan 005's callback reads `info["episode"]["r"]`
   and `["l"]` — this is the standard SB3 convention (used by
   `BaseCallback`/`EvaluatePolicy`), so the last episode is recorded exactly
   when it ends, not at the next `reset`. Do not also emit it on
   truncation-only unless `terminated` is also true (a truncated-but-not-
   terminated episode is a training-time artefact; plan 005 decides whether
   to log it — keep this env's contract as "emit on `terminated`").

4. **`terminated` after an invalid action**: if every action is masked, the
   env terminates (budget exhausted / no candidates). This can happen on the
   very first `step` if the budget is too small for any candidate — that is
   correct behaviour, and the test plan covers it.

5. **Reward `float` cast**: return `float(reward)` so the value is a Python
   float (satisfies `SupportsFloat` and serialises cleanly in SubprocVecEnv).

6. Initialise `self._bike_nodes_before: set[int | str] = set()` in `__init__`
   (add it next to the other episode-state fields) so the attribute exists
   before the first `step`. Do not rely on `_apply_edge` having run.

Add `self._bike_nodes_before: set[int | str] = set()` to the `__init__`
episode-state block (after `self._bike_nodes = set()`).

**Verify**:
- `ruff check bike_rl/env.py` → exit 0
- `ruff format --check bike_rl/env.py` → exit 0
- `mypy --strict bike_rl/env.py` → exit 0
- `python -c "
import networkx as nx, datetime, pathlib
from bike_rl.config import Config
from bike_rl.run_context import RunContext
from bike_rl.env import BikePathEnv
bg = nx.MultiDiGraph()
for n,(x,y) in {1:(0.,0.),2:(0.001,0.),3:(0.,0.001),4:(0.001,0.001)}.items(): bg.add_node(n,x=x,y=y)
bg.add_edge(1,2,length=120.0,highway='residential',bike_lane='yes'); bg.add_edge(3,4,length=130.0,highway='residential',bike_lane='yes')
wg = nx.MultiDiGraph()
for n,(x,y) in {1:(0.,0.),2:(0.001,0.),3:(0.,0.001),4:(0.001,0.001),5:(0.002,0.002)}.items(): wg.add_node(n,x=x,y=y)
wg.add_edge(2,5,length=200.0,highway='primary'); wg.add_edge(3,5,length=150.0,highway='secondary'); wg.add_edge(4,5,length=600.0,highway='tertiary')
rc = RunContext(run_id='t', output_dir=pathlib.Path('/tmp/t'), timestamp=datetime.datetime.now(datetime.timezone.utc))
env = BikePathEnv(bg, wg, Config(), rc)
env.reset(options={'budget': 100_000.0})
mask = env.action_masks()
a = int(mask.argmax())          # first legal action
obs, r, term, trunc, info = env.step(a)
print('reward', round(r,3), 'term', term, 'trunc', trunc, 'info', info, 'obs', obs)
"` → prints a finite float reward, `term False` (more candidates remain affordable), `trunc False`, an `info` dict (empty or with episode info), and a 4-float obs. The reward should be **positive** (adding a primary road that connects to the bike network: continuity_bonus +50, road_priority 5*5=25, plus state-gain and budget terms, minus any fragmentation change) — assert mentally that it is `> 0`.

### Step 4: Write `tests/test_env.py`

Create `tests/test_env.py`. Model the structure on `tests/test_candidates.py`
(top docstring, `from __future__ import annotations`, small focused
functions with docstrings, fixtures from `conftest.py` plus local fixtures).

Local helper + fixtures to define at the top of the file:

```python
"""Tests for ``bike_rl.env.BikePathEnv``.

Covers: action masking (§5.4), budget exhaustion termination, invalid-action
no-op, reset restoration, observation shape/dtype/range, episode info on
termination (§5.12), step-cap truncation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
import pytest

from bike_rl.config import Config
from bike_rl.env import BikePathEnv
from bike_rl.run_context import RunContext

if TYPE_CHECKING:
    _NXGraph = nx.MultiDiGraph[Any, Any, Any]
else:
    _NXGraph = nx.MultiDiGraph


@pytest.fixture
def run_context(tmp_path: Path) -> RunContext:
    """A throwaway RunContext pointing at a tmp dir."""
    return RunContext(
        run_id="test",
        output_dir=tmp_path / "out",
        timestamp=datetime.now(timezone.utc),
    )


def _make_env(
    bike_graph: _NXGraph, walk_graph: _NXGraph, cfg: Config, run_context: RunContext,
    budget: float = 100_000.0,
) -> BikePathEnv:
    """Build and reset a BikePathEnv with the given budget."""
    env = BikePathEnv(bike_graph, walk_graph, cfg, run_context)
    env.reset(options={"budget": budget})
    return env
```

(Keep this import order — it satisfies ruff `I`. `networkx` and `Any` are
used only via the `_NXGraph` type alias and the `_make_env` annotation.)

Test cases (write all; each is a short function with a docstring):

1. `test_observation_space_and_shape(tiny_bike_graph, tiny_walk_graph, run_context)`
   — `env.observation_space.shape == (4,)` and
   `env.observation_space.dtype == np.float32`;
   `env.action_space.n == 3` (the three candidates `(2,5)`, `(3,5)`, `(4,5)`
   from `tiny_walk_graph`). After `reset`, `obs` is an `np.ndarray` of shape
   `(4,)`, dtype `float32`, and every element is in `[0, 1]`.
2. `test_action_masks_correct_initially(tiny_bike_graph, tiny_walk_graph, run_context)`
   — after reset with a large budget, `env.action_masks()` is all `True`
   for the 3 valid slots and has length 3; `mask.dtype == bool`.
3. `test_action_masks_masks_unaffordable(tiny_bike_graph, tiny_walk_graph, run_context)`
   — reset with `budget = 1000.0`. The `(4,5)` edge has length 600 → cost
   `6000`, unaffordable; the `(2,5)` cost `2000` and `(3,5)` cost `1500` are
   also unaffordable. So **all** actions are masked → `mask.any()` is `False`.
   Then the first `step(0)` must `terminate` immediately (no legal action)
   with reward `0.0` and `info["episode"]` present (§5.12 — even a
   zero-length episode emits episode info on termination). Assert
   `terminated is True`, `reward == 0.0`, and `"episode" in info`.
4. `test_invalid_action_is_noop_and_not_done(tiny_bike_graph, tiny_walk_graph, run_context)`
   — reset with a large budget. Find an index whose mask is `False` by
   stepping once (consumes one slot), then `step` with the consumed index
   again: the slot is now `None` → masked. Assert `reward == 0.0`,
   `terminated is False` (§5.4: invalid actions do not end the episode),
   and `info["invalid_action"] is True`. Also assert the obs is unchanged
   from the previous step (no state mutation).
5. `test_step_uses_slot_and_budget_decreases(tiny_bike_graph, tiny_walk_graph, run_context)`
   — reset with `budget = 100_000`. `mask = env.action_masks()`; pick
   `a = int(mask.argmax())`. Before stepping, read
   `cost = env._slots[a].length * cfg.edge_cost_factor` (accessing the
   private slot is fine in a test). `obs, r, term, trunc, info = env.step(a)`.
   Assert `env._budget == 100_000 - cost` (budget deducted), and
   `env._slots[a] is None` (slot consumed). Assert `r` is a finite float.
6. `test_budget_exhaustion_terminates(tiny_bike_graph, tiny_walk_graph, run_context)`
   — reset with a budget just enough for one cheapest candidate. The
   cheapest candidate is `(3,5)` length 150 → cost 1500. Set
   `budget = 1500.0`. `mask` should have exactly one `True` (the `(3,5)`
   slot). `step` that action: after it, budget is 0, so the next mask is
   all `False`. Assert the first step returns `terminated is True` (no
   further affordable candidate) and `info["episode"]["l"] == 1`.
7. `test_reset_restores_state(tiny_bike_graph, tiny_walk_graph, run_context)`
   — reset with a large budget; step twice (consuming two slots, deducting
   budget). Call `reset(options={"budget": 100_000.0})` again. Assert
   `env._slots` has 3 non-None entries again, `env._budget == 100_000.0`,
   `env._steps == 0`, `env._episode_reward == 0.0`, and the obs equals the
   first-reset obs (determinism — same `cfg.seed`).
8. `test_reset_deterministic_with_same_seed(tiny_bike_graph, tiny_walk_graph, run_context)`
   — `env.reset(seed=42, options={"budget": 100_000.0})`; record `obs1`.
   `env.reset(seed=42, options={"budget": 100_000.0})` again; record `obs2`.
   Assert `np.array_equal(obs1, obs2)`. (The env is deterministic: metrics
   sampling is seeded by `cfg.seed` inside `MetricsState`, and `reset`
   rebuilds the same graph.)
9. `test_episode_info_emitted_on_termination(tiny_bike_graph, tiny_walk_graph, run_context)`
   — drive the env to termination with a small budget (one affordable
   candidate). On the terminating step, assert `info["episode"]["r"]` is a
   float equal to the sum of rewards returned across the episode, and
   `info["episode"]["l"]` equals the number of steps taken. (Pin §5.12:
   the last episode is recorded when it ends, not lost.)
10. `test_step_cap_truncates(tiny_bike_graph, tiny_walk_graph, run_context)`
    — build `cfg = Config(max_episode_steps=1)`; reset with a large budget;
    `step` any legal action. Assert `truncated is True` and
    `terminated is False` (there are still affordable candidates, so it is
    the cap that stopped the episode, not budget). This pins that
    `max_episode_steps > 0` enables truncation independent of budget.
11. `test_action_masks_length_matches_action_space(tiny_bike_graph, tiny_walk_graph, run_context)`
    — `len(env.action_masks()) == env.action_space.n`.
12. `test_no_bike_lane_candidates_terminates_immediately(run_context)`
    — a bike graph and walk graph with **no** valid candidates (e.g. all
    walk edges already in the bike graph, or all too short). Build a tiny
    pair inline: `bike` with edge `(1,2)` bike_lane; `walk` with only edge
    `(1,2)` (already in bike → skipped). `env.action_space.n == 1` (padded),
    `action_masks()` is all `False`, and the first `step(0)` terminates
    with reward 0 and `info["episode"]["l"] == 0`. (Covers the
    zero-candidate edge case and the `max(1, …)` padding.)
13. `test_env_does_not_mutate_input_graphs(tiny_bike_graph, tiny_walk_graph, run_context)`
    — snapshot `tiny_bike_graph.number_of_edges()` and the set of edge
    `bike_lane` tags before constructing the env. Construct, reset, step
    twice. Assert the original `tiny_bike_graph` still has the same edge
    count and same `bike_lane` tags (the env copies in `__init__` and
    `reset`; §5.8 no source mutation). Likewise `tiny_walk_graph` is
    unchanged.

Structural pattern: `tests/test_candidates.py` (short functions, docstrings,
`from __future__ import annotations`).

**Verify**:
- `ruff check tests/test_env.py` → exit 0
- `ruff format --check tests/test_env.py` → exit 0
- `pytest -q tests/test_env.py` → all pass (13 tests)

### Step 5: Write `tests/test_reward.py`

Create `tests/test_reward.py` — focused on the reward formula (§3.5 + §5.5
fix). Use the same `_make_env`/`run_context` helpers (copy them into this
file; do not import from `test_env.py` — test files should be independent).

Test cases (each a short function with a docstring):

1. `test_budget_efficiency_term_is_nonzero(tiny_bike_graph, tiny_walk_graph, run_context)`
   — **the §5.5 regression test**. Build two envs from the same graphs:
   one with `Config(w_budget_efficiency=0.0)` (term off), one with
   `Config()` (term on, default `0.1`). Reset both with a large budget.
   Step the **same** first legal action in both (find the action index that
   is legal in both via `action_masks().argmax()` — they start identical).
   Let `r_off` and `r_on` be the two rewards. Assert `r_on != r_off` and
   specifically `r_on > r_off` (the budget-efficiency term is a **bonus**
   that is non-zero when the addition produces a non-negative state gain,
   which adding a primary road that connects to the network does). Document
   in the docstring that this pins RECREATE_SPEC §5.5 (the original term was
   a no-op ≈0).
2. `test_budget_efficiency_term_clamped(tiny_bike_graph, tiny_walk_graph, run_context)`
   — the cap binds when the raw term would be huge. Concretely: build
   `cfg = Config(budget_efficiency_cap=1.0, w_budget_efficiency=1.0)`;
   reset with `budget = 1e9` (so `frac = cost/initial_budget` is tiny and
   `gain_scaled/frac` is enormous). Step the first legal action (the
   `(2,5)` primary edge, cost 2000 ≪ 1e9 so it is affordable); compute
   `r_on`. Build a second env with `Config(w_budget_efficiency=0.0)`
   (term off) and step the same action; compute `r_off`. The capped term
   is clamped to exactly `cfg.budget_efficiency_cap = 1.0`, so assert
   `abs((r_on - r_off) - cfg.budget_efficiency_cap) < 1e-6` **and**
   `math.isfinite(r_on)` (§5.5 must not blow up). (Verified: with the real
   metrics, `state_gain ≈ +0.00426` for this step, so `gain_scaled > 0`
   and the raw term is `1.0 * 0.426 / 2e-6 ≈ 213_000`, far above the 1.0
   cap.)
3. `test_continuity_bonus_only_when_connects(tiny_bike_graph, tiny_walk_graph, run_context)`
   — every candidate in `tiny_walk_graph` has an endpoint in the bike graph
   (nodes 2, 3, 4 are all in `tiny_bike_graph`), so every first step gets
   the `continuity_bonus`. To test the "no bonus" branch, build an inline
   graph pair where a candidate does **not** touch the bike network:
   `bike` = single node `1` (no bike-lane edges, so `_bike_nodes` is empty
   after reset) → wait, with no bike-lane edges `coverage` is 0 and the
   env still works; but `extract_candidates` needs the walk edge's
   endpoints not in bike. Simpler: `bike` has edge `(1,2)` bike_lane (so
   `_bike_nodes = {1,2}`); `walk` has an extra edge `(50,51)` residential
   length 200 (neither endpoint in bike). Reset; find the slot for
   `(50,51)` (its mask is True if affordable). Step it. The reward must
   **not** include `continuity_bonus` (connects is False) and **must**
   include `isolation_penalty` (`-100`, since both endpoints are new →
   `would_create_isolated`). Assert the reward is `< 0` (the isolation
   penalty dominates for an isolated edge with ~0 state gain) — this also
   pins the isolation term. Then step a connecting candidate (`(2,5)`-like
   in `tiny_walk_graph`) in a separate env and assert its reward is `> 0`
   (continuity bonus applies). This single test covers both the
   no-continuity and the isolation-penalty branches.
4. `test_isolation_penalty_applied_once(tiny_bike_graph, tiny_walk_graph, run_context)`
   — (Covered by case 3's isolated-edge assertion, but write it
   separately for clarity.) Build the inline isolated-candidate env from
   case 3. Step the isolated `(50,51)` action once. Compute the reward.
   Then build a second identical env and step the **same** action with
   `Config(isolation_penalty=0.0)`. Assert the difference
   `r_default - r_nopenalty == cfg.isolation_penalty` (i.e. `-100.0`, to
   within `1e-6`). This pins that the isolation penalty is applied exactly
   once for an isolated addition (§3.5).
5. `test_reward_is_finite_float(tiny_bike_graph, tiny_walk_graph, run_context)`
   — reset with a large budget; step every legal action until termination
   (loop: `while not terminated: a = mask.argmax(); step`). Assert every
   reward returned is a finite Python float (`math.isfinite(r)` and
   `isinstance(r, float)`). No `NaN`/`inf` from the budget-efficiency
   division (guarded by `cost > 0` and `initial_budget > 0`).
6. `test_reward_road_priority_component(tiny_bike_graph, tiny_walk_graph, run_context)`
   — the first candidate by `argmax` of the mask is the highest-priority
   affordable one. In `tiny_walk_graph` the candidates are `(2,5)` primary
   (priority 5), `(3,5)` secondary (4), `(4,5)` tertiary (3); with a large
   budget the mask is all True so `argmax` returns index 0 which is the
   primary (sorted by priority descending in `extract_candidates`). Step
   it. Build a second env with `Config(road_priority_scale=0.0)` and step
   the same action. Assert `r_default - r_noscale == 5 *
   cfg.road_priority_scale` (i.e. `25.0`) to within `1e-6` — pinning the
   road-priority term `road_priority * road_priority_scale` (§3.5). (The
   state-gain, continuity, fragmentation, and budget terms are identical
   across the two envs since the action and graphs are the same, so they
   cancel in the difference.)

Structural pattern: `tests/test_candidates.py`.

**Verify**:
- `ruff check tests/test_reward.py` → exit 0
- `ruff format --check tests/test_reward.py` → exit 0
- `pytest -q tests/test_reward.py` → all pass (6 tests)

### Step 6: Full gate + coverage

Run the full scoped gate and confirm coverage on the new module.

**Verify**:
- `ruff check bike_rl tests` → exit 0
- `ruff format --check bike_rl tests` → exit 0
- `mypy --strict bike_rl` → exit 0
- `pytest -q` → all pass (43 prior + 13 env + 6 reward = 62)
- `pytest --cov=bike_rl --cov-report=term-missing tests/test_env.py tests/test_reward.py`
  → exit 0; **`bike_rl/env.py` shows ≥80% line coverage**. The uncovered
  lines should only be rare branches (e.g. the `options` budget-override
  path that the CLI uses, or the `initial_budget <= 0` fallback). If
  coverage is below 80%, add a test that exercises the
  `options={"budget": ...}` override path and the fallback (reset without
  options and with `options={}`). `config.py`'s new three fields will also
  show as covered by `test_smoke.test_config_defaults` only if you extend
  it — **do not modify `test_smoke.py`**; instead the new env/reward tests
  reading `cfg.w_budget_efficiency` etc. cover the fields' read path, and
  the field definitions themselves are covered by instantiation.

## Test plan

New test files:

- `tests/test_env.py` (13 tests, listed in Step 4): observation
  space/shape/dtype/range, initial mask correctness, unaffordable-action
  masking + immediate termination, invalid-action no-op (§5.4), slot/budget
  bookkeeping, budget-exhaustion termination, reset restoration +
  determinism, episode info on termination (§5.12), step-cap truncation,
  zero-candidate padding, no input-graph mutation.
- `tests/test_reward.py` (6 tests, listed in Step 5): §5.5 budget-efficiency
  term non-zero (regression), cap binding, continuity bonus only when
  connects + isolation penalty for isolated additions, isolation penalty
  applied once, rewards finite across a full episode, road-priority
  component.

Structural pattern: `tests/test_candidates.py`.

Verification: `pytest -q tests/test_env.py tests/test_reward.py` → 19 pass;
`pytest --cov=bike_rl --cov-report=term-missing` shows `bike_rl/env.py`
≥80%.

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `ruff check bike_rl tests` exits 0
- [ ] `ruff format --check bike_rl tests` exits 0
- [ ] `mypy --strict bike_rl` exits 0
- [ ] `pytest -q` exits 0; 13 new tests in `tests/test_env.py` and 6 new
      tests in `tests/test_reward.py` exist and pass; the 43 prior tests
      still pass
- [ ] `pytest --cov=bike_rl --cov-report=term-missing tests/test_env.py tests/test_reward.py`
      shows ≥80% line coverage on `bike_rl/env.py`
- [ ] No `raise NotImplementedError("Plan 004")` remains in `bike_rl/env.py`
      (`grep -rn "Plan 004" bike_rl/env.py` returns nothing)
- [ ] `bike_rl/env.py` does **not** import `sb3_contrib` or
      `stable_baselines3` (`grep -n "sb3_contrib\|stable_baselines3"
      bike_rl/env.py` returns nothing — MaskablePPO is plan 005's concern)
- [ ] `bike_rl/env.py` has no `print(` and no bare `except:`
      (`grep -nE "print\(|except:" bike_rl/env.py` returns nothing)
- [ ] `bike_rl/config.py` has exactly three new fields
      (`w_budget_efficiency`, `budget_efficiency_cap`, `max_episode_steps`)
      and no other changes
      (`git diff 06a124d..HEAD -- bike_rl/config.py` shows only those
      additions + docstring lines)
- [ ] `bike_rl/metrics.py`, `bike_rl/candidates.py`, `bike_rl/graph_utils.py`,
      `bike_rl/run_context.py`, `bike_rl/objective.py` are **unchanged**
      (`git diff --stat 06a124d..HEAD -- bike_rl/metrics.py
      bike_rl/candidates.py bike_rl/graph_utils.py bike_rl/run_context.py
      bike_rl/objective.py` shows no changes)
- [ ] No files outside the in-scope list are modified (`git status --short`
      lists only `bike_rl/env.py`, `bike_rl/config.py`,
      `tests/test_env.py`, `tests/test_reward.py`)
- [ ] `plans/README.md` status row for 004 updated (TODO → DONE) — unless a
      reviewer told you they maintain the index

## STOP conditions

Stop and report back (do not improvise) if:

- The code at the locations in "Current state" doesn't match the excerpts
  (e.g. `env.py` already has real bodies, `Config` is missing the fields
  plan 003 said it has — `coverage_radius_m`, `coverage_mode`, `seed` — or
  `MetricsState` lacks `connectivity`/`path_efficiency`/`fragmentation`/
  `coverage`/`add_edge`). The codebase has drifted since this plan was
  written; re-baseline before continuing.
- The gymnasium API differs from "Gymnasium / MaskablePPO API facts" —
  specifically: `gymnasium.__version__` is not `1.3.x`, `gym.spaces.Box`
  rejects `dtype=np.float32`, or `gym.Env.step`'s return arity is not
  `(obs, reward, terminated, truncated, info)`. Report the installed
  version and the actual signature; do not invent a different API path.
- `MetricsState.add_edge` does not keep `self._metrics.fragmentation()` /
  `.coverage()` / `.connectivity()` / `.path_efficiency()` consistent with
  the pure functions after an add (i.e. plan 003's incremental invariant is
  broken). Do not patch `metrics.py` from this plan — report it so plan 003
  can be re-baselined.
- The reward's budget-efficiency term (§5.5) cannot be made non-zero for a
  positive-gain addition with the formula in Step 3 — i.e. the
  `test_budget_efficiency_term_is_nonzero` test cannot pass. Report the
  computed `r_on`, `r_off`, and the term value; do not ship a no-op term.
- `extract_candidates` on `tiny_walk_graph` does not return the three
  candidates `(2,5)`, `(3,5)`, `(4,5)` (priority 5/4/3) — i.e. plan 002's
  candidate contract has drifted. Report what it returns.
- A step's verification fails twice after a reasonable fix attempt.
- The fix appears to require touching an out-of-scope file (e.g. you find
  `metrics.py` or `objective.py` or `training.py` must be edited to make
  tests pass — they must not be).

## Maintenance notes

For the human/agent who owns this code after the change lands:

- **§6.4 heap-based candidate queue is deferred.** `step` does an O(n)
  `self._slots.index(candidate)` and `action_masks` is O(n) per call. For
  large candidate counts (real cities: thousands), this is O(n) per step
  and O(n²) per episode — the same asymptotic class as the original's
  re-sort, but without the re-sort constant. If plan 005 (training) shows
  env throughput as the bottleneck, implement §6.4: keep a
  `dirty`-set of candidates whose `connects_to_bike_path` flag changed
  (only candidates adjacent to newly-added `bike_nodes`), and a
  bucket/heap keyed on `(connects, road_priority)`. The **fixed action-space
  index contract (§5.4) must remain** — do not reindex slots; the heap is
  an *advisory* structure for choosing actions inside the agent, not a
  reordering of the env's action space. MaskablePPO relies on index →
  candidate being stable across the episode.
- **Picklability for SubprocVecEnv (plan 005).** `BikePathEnv` holds a
  `MetricsState` which holds a rustworkx `PyGraph` (verified picklable) and
  a `UnionFind` (plain dicts — picklable). The env also holds
  `self._bike_graph_original` (NetworkX — picklable) and `self._walk_graph`.
  So the env should pickle cleanly for `SubprocVecEnv`. If plan 005 hits a
  pickle error, the likely culprit is a lambda stored on `self` — there is
  none in this design; if one is added later, switch to a module-level
  function. The `run_context` carries a `Path` and a `datetime` — both
  picklable.
- **Budget is passed via `reset(options={"budget": …})`, not `Config`.**
  This is deliberate (§3.9: budget is a per-run CLI arg). Plan 005's
  `make_env` / `SubprocVecEnv` wrapper must pass the budget through
  `options` on the first `reset`. If you forget, the env falls back to
  `100_000.0` and logs nothing — add a `logger.warning` in the fallback
  branch if you want it to be loud (currently silent; the tests cover the
  fallback path).
- **Episode info contract.** `info["episode"] = {"r": …, "l": …}` is
  emitted **only on `terminated=True`**. Plan 005's `TrainingProgressCallback`
  must read `info["episode"]` (the SB3 convention) rather than appending at
  the next `reset` (that was the §5.12 bug). If plan 005 also wants
  truncated-episode stats, it can also check `truncated` — but the env's
  contract is: episode summary available in `info` when `terminated`.
- **Reviewer scrutiny.** Check that (a) the action space is truly fixed
  across `reset` (no reassignment of `self.action_space`); (b) invalid
  actions never set `terminated=True` *because* they're invalid (only
  all-masked terminates); (c) `_bike_nodes_before` is captured before the
  edge is applied (the continuity/isolation logic depends on it); (d) the
  budget-efficiency term is guarded by `cost > 0` and `initial_budget > 0`
  so no `ZeroDivisionError`; (e) `step` returns `np.float32` obs and a
  Python `float` reward (SubprocVecEnv serialises these).
- **Follow-up deferred:** the `--export_geojson` / plotting / SLURM wiring
  (plans 006–007) and the `objective.py` shared objective (optimiser plan).
  The env's reward is intentionally a *derived* reward per §3.5, separate
  from `objective.py`'s canonical scalar; do not merge them — that
  conflation is explicitly out of scope and would break the optimiser plans.
