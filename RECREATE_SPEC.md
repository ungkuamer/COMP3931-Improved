# Bike-Path RL — Recreation & Improvement Specification

This document is a complete specification for recreating an **improved version** of the `nx-rx-simple-ur.py` project in a **new repository**. It describes what the original project does, how it works, all known bugs and weaknesses, and a concrete target design. An AI agent should be able to build the improved repo from this document alone.

---

## 1. Project Purpose

A **reinforcement-learning (RL) tool that plans where to add new bike lanes to a city's street network under a fixed construction budget.**

Given a city (by name or bounding box), it:

1. Loads the city's street network from OpenStreetMap via OSMnx.
2. Builds a custom Gymnasium environment where each *action* = "add bike infrastructure to one candidate road segment," subject to a budget.
3. Trains a PPO agent (Stable-Baselines3) with multiple parallel environments to maximise a reward combining connectivity, path efficiency, population coverage, fragmentation reduction, and budget efficiency.
4. Evaluates the trained policy over several episodes, picks the best solution, and renders maps/metrics plots.

Original reference implementation: a single 1,582-line file `COMP3931/nx-rx-simple-ur.py` (~1,582 LOC), plus SLURM job scripts. It uses **rustworkx** for faster graph operations than NetworkX.

---

## 2. Stack & Dependencies

Runtime (original `requirements.txt` — **note the errors**):

```
osmnx
networkx
numpy
pandas
matplotlib
geopandas
shapely
gym            # ❌ should be gymnasium
tqdm
stable-baselines3
shimmy          # ❌ unused, drop
rustworkx
scipy
```

**Corrected & pinned dependency list for the new repo** (`requirements.txt`):

```
osmnx>=1.9
networkx>=3.2
gymnasium>=0.29
stable-baselines3>=2.2
sb3-contrib>=2.2          # for MaskablePPO (action masking)
rustworkx>=0.14
numpy>=1.26
pandas>=2.1
scipy>=1.11
geopandas>=0.14
shapely>=2.0
matplotlib>=3.8
tqdm>=4.66
```

`requirements-dev.txt`:

```
pytest>=8.0
pytest-cov>=4.1
ruff>=0.4
mypy>=1.8
```

Python **3.10+** (the original ran on 3.12).

---

## 3. How the Original Works (reference behaviour to preserve)

### 3.1 Data acquisition

- `ox.graph_from_place(city_name, network_type='bike')` → the existing bike network (`original_graph`).
- `ox.graph_from_place(city_name, network_type='walk')` → the walkable network (`walk_graph`), the source of *candidate* edges to upgrade.
- Or via bbox: `ox.graph_from_bbox(north, south, east, west, ...)` — order is **(N, S, E, W)**.

### 3.2 Candidate edges

Edges from the walk graph that are **not already** in the bike graph, with:

- `highway` type in `{primary, secondary, tertiary, residential, unclassified}` (priorities 5/4/3/2/1).
- Length filter: `100 < length < 1000` metres.
- Each candidate carries `road_priority` and a `connects_to_bike_path` flag (whether an endpoint already touches the bike network).
- Sorted by `(road_priority, connects_to_bike_path)` descending.

### 3.3 Graph conversion

`nx_to_rx(nx_graph)` converts NetworkX → rustworkx via `rx.networkx_converter`, builds a `node_map` (original node ID → rustworkx index) using the `"__networkx_node__"` attribute, and defaults missing `length` to 1.0.

### 3.4 Environment (`BikePathEnv`, Gymnasium)

- **Action space**: `Discrete(len(candidate_edges))` — index into the candidate list.
- **Observation space**: `Box(low=0, high=1, shape=(4,), float32)` = `[connectivity, path_efficiency, population_served, normalized_budget]`.
- **`step(action)`**:
  - Look up `candidate_edges[action] = (u, v, data)`.
  - `cost = data['length'] * edge_cost_factor` (factor default 10).
  - If `cost > budget`: pop the action, return reward −1, maybe done.
  - If action ≥ candidate count: **terminate** with −10 (this is a bug — see §5.4).
  - Add edge to `self.graph` with `bike_lane='yes'`; also add to `self.rx_graph`.
  - Update `bike_nodes`, `added_paths`.
  - State update: if the edge connects to the existing network → **incremental** update of path efficiency only (bug — §5.6); otherwise full `_get_state()`.
  - Compute reward (see §3.5).
  - Update `connects_to_bike_path` flags on remaining candidates, re-sort, remove the used edge.
  - `done` when budget < cheapest remaining edge cost or no candidates left.
- **`reset()`**: restore `graph = original_graph.copy()`, rebuild rx graph, rebuild candidates, reset budget/steps, return state.

### 3.5 Reward (original weights — to revise)

```
state_improvement = (new-old)*w for (connectivity, efficiency, population) with w = [0.9, 0.2, 0.3]
reward = sum(state_improvement) * 100
+ road_priority * 5
+ 50  if connects_to_bike_path
+ (old_frag - new_frag) * 200
- 100 if would_create_isolated
+ budget_efficiency term (currently a no-op — §5.5)
```

### 3.6 Metrics

- **Connectivity** = normalised clustering/transitivity of the largest connected component of the (simplified, undirected) graph; uses rustworkx `transitivity` or sampled `local_clustering_coefficient` for large graphs. Cached by edge count.
- **Path efficiency** = `1 / (1 + avg_path_length/1000)`, where avg path length is from sampled-source Dijkstra (`rx.dijkstra_shortest_path_lengths`). Aggressive sampling by graph size.
- **Fragmentation** (NetworkX, recomputed every step) = `0.3*min(1,(n_comp-1)/10) + 0.4*(1 - largest_cc_fraction) + 0.3*isolation_factor` on the bike-lane-only subgraph.
- **Population served** = `0.7*coverage_ratio + 0.3*largest_component_ratio` (currently near-constant — §5.7).
- **Update frequency** by graph size: >5000 edges → every 5 steps; >1000 → every 3; else every 1 (this throttling exists in init but isn't actually applied in `step`).

### 3.7 Training

- `SubprocVecEnv` with `n_envs` workers (default `cpu_count - 1`).
- Each worker re-instantiates `BikePathEnv` (re-downloads OSM — bug §6.1).
- PPO hyperparameters: `MlpPolicy`, `lr=3e-4`, `n_steps=2048`, `batch_size=64`, `n_epochs=10`, `gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.2`, `device="cpu"`.
- Timesteps are rounded up to `n_steps * n_envs` per update batch.
- Callbacks: `TrainingProgressCallback` (tqdm + reward tracking) and `CheckpointCallback` (every `total_timesteps//10`).

### 3.8 Evaluation

- `evaluate_and_visualize` runs `num_evaluations` deterministic rollouts on fresh envs.
- `EvaluationTracker` stores per-episode reward/paths/connectivity/efficiency/population/budget, tracks best.
- Renders best solution map (OSMnx base + red added paths + stats annotation) and plots metrics.
- Outputs saved to a per-run directory `./bike_path_figures/<city>_<run_id>/` (run_id = timestamp).

### 3.9 CLI (`main`)

`argparse` with mutually exclusive `--city` / `--bbox N S E W`; `--budget` (default 100000), `--timesteps` (10240), `--eval_episodes` (5), `--skip_training`, `--model_path`, `--device`, `--n_envs`, `--no_plots`. Writes `run_summary.txt`.

### 3.10 SLURM

- `run_bike_path_slurm.sh` / `submit_jobs.sh` submit array jobs over timestep values `[10240, 20480, 51200, 102400]`, per-task venv + pip install, default city "Otley, UK", 8 CPUs, 16G, 2h, email notifications. Outputs tarballed.

---

## 4. Target Repo Structure (improved)

Modular package instead of one file:

```
bike-path-rl/
├── README.md
├── pyproject.toml              # or setup.py + requirements.txt
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
├── .github/workflows/ci.yml    # pytest + ruff + mypy
├── bike_rl/
│   ├── __init__.py
│   ├── config.py               # dataclass: all hyperparameters & magic numbers
│   ├── run_context.py          # RunContext dataclass replacing global RUN_ID/RUN_FIGURES_DIR
│   ├── graph_utils.py          # nx_to_rx, graph validation, OSM loading + caching
│   ├── candidates.py           # candidate edge extraction (no source-graph mutation)
│   ├── metrics.py              # connectivity, fragmentation, population-served (pure fns)
│   ├── env.py                  # BikePathEnv with action masking
│   ├── training.py             # make_env, train_model, callbacks
│   ├── evaluation.py           # EvaluationTracker, evaluate_and_visualize
│   ├── plotting.py             # render + reward/metric plots (Agg-safe)
│   └── cli.py                  # main() / argparse
├── scripts/
│   ├── run_bike_path_slurm.sh
│   └── submit_jobs.sh
└── tests/
    ├── conftest.py             # small synthetic graph fixtures, OSM mocks
    ├── test_graph_utils.py
    ├── test_candidates.py
    ├── test_metrics.py
    ├── test_env.py
    ├── test_reward.py
    ├── test_training.py        # mock PPO.learn
    └── test_cli.py
```

---

## 5. Correctness Bugs to Fix (priority order)

### 5.1 Duplicated graph-loading block (CRITICAL — script fails on bbox)

Original runs an `if city_name / elif bbox / else` loader, **then unconditionally** re-runs `ox.graph_from_place(city_name, ...)` again with the local `city_name` (which is `None` when bbox is used). **Delete the second block entirely.** Load once in the if/elif/else.

### 5.2 Missing import (CRITICAL — won't run)

`from learning_analyzer import analyze_learning_curve, find_latest_rewards_data` — the module doesn't exist and the names are unused. **Remove the import.**

### 5.3 `requirements.txt` mismatch

`gym` → `gymnasium`; drop `shimmy`; pin versions (§2).

### 5.4 Action-space / spurious termination (CORE RL BUG)

Action space is `Discrete(initial_candidate_count)` but candidates shrink every step. PPO outputs over the **original** size, so any action ≥ current count triggers `return ..., -10, True, ...` — the agent learns to terminate early.
**Fix**: use **`MaskablePPO` from `sb3-contrib`** with `env.action_masks()` returning a boolean mask over the *fixed* action space of valid (budget-affordable, not-yet-used) candidates. Never terminate on an invalid/masked action; instead pad the action space to the initial size and mask invalid indices. Invalid actions should yield reward 0 and **not** end the episode.

### 5.5 Budget-efficiency reward is a no-op

```
cost = length * edge_cost_factor      # edge_cost_factor = 10
budget_efficiency = cost / (length+1) # ≈ 10
reward += max(0, (1 - budget_efficiency/10) * 10)   # ≈ 0
```

**Fix**: define a meaningful cost-effectiveness term, e.g. `reward += connectivity_gain / cost` (reward per unit budget), or normalise `cost` against `initial_budget`: `reward += w * (state_gain - cost/initial_budget)`. Make weights live in `Config`.

### 5.6 Incremental state update leaves connectivity & population stale

When `connects_to_bike_path`, only `state[1]` (efficiency) and `state[3]` (budget) update — connectivity `[0]` and population `[2]` stay stale, so the reward's improvement terms for those are always 0 for the most important (connected) additions.
**Fix**: maintain incremental connectivity & population updates, **or** always run full `_get_state()` but make it cheap via incremental data structures (§6.2–6.3). Prefer correctness over the fragile incremental shortcut.

### 5.7 `_calculate_population_served` is near-constant

`bike_nodes = len(self.graph.nodes())` counts the **whole original** bike graph plus a few added edges, so coverage barely moves. The signal the agent needs is missing.
**Fix**: base population served on **bike-lane-reachable** nodes (e.g. nodes within `X` metres of any bike lane, or nodes in the same connected component as a bike lane), not on total graph node count. Document the chosen proxy in `config.py`.

### 5.8 Candidate extraction mutates the source walk graph

Original writes `data['road_priority']` / `data['connects_to_bike_path']` onto `walk_graph`'s own edge dicts; these persist across episodes after `reset()`.
**Fix**: in `candidates.py`, build a **fresh** `Candidate` dataclass per edge with its own copied attributes; never mutate the source graph. Re-derive `connects_to_bike_path` from current `bike_nodes` at step time.

### 5.9 `rx_to_nx` is dead + broken

Defined but never called; the undirected branch uses a non-existent `rx_graph.edge_list(u_idx)` API. **Delete it.** If back-conversion is needed later, implement and unit-test it.

### 5.10 `plt.show()` on headless clusters

`render` always calls `plt.show()`, which breaks/logs noise on SLURM. **Fix**: set `matplotlib.use("Agg")` by default in `plotting.py`; gate `plt.show()` behind a `show: bool = False` param and a `--show` CLI flag.

### 5.11 Bare `except:` clauses

Three bare `except:` swallow `KeyboardInterrupt`/`SystemExit` and hide errors. Use `except Exception as e:` and log via `logging`.

### 5.12 Episode bookkeeping drops the last episode

`episode_rewards` only appended at the **start** of the next `reset()`, so the final episode is lost. Append in a proper `_on_episode_end` hook (or in `step` when `done`).

### 5.13 `RUN_FIGURES_DIR` global doesn't exist in SubprocVecEnv workers

Workers never call `setup_run_directory`, so the global is `None` inside workers and output paths fall back to defaults inconsistently. **Fix**: pass a `RunContext` explicitly to every component; eliminate globals entirely.

### 5.14 Connectivity cache keyed only on edge count

Safe today but fragile. **Fix**: use a monotonic `self.graph_version` counter incremented on every edge add; cache keyed on that.

---

## 6. Performance Improvements

### 6.1 Cache the OSM download (big win with parallel envs)

Each `SubprocVecEnv` worker re-downloads the same graphs from OSM (n_envs × duplicate network I/O, rate-limited).
**Fix**: in the parent process, download & simplify the graphs once; serialise (pickle) and pass to workers, **or** enable `ox.settings.use_cache=True` + `ox.settings.use_cache` and cache to `./osm_cache/` on disk. Workers then load from cache.

### 6.2 Maintain a persistent `rx_simple_graph` for metrics

`_calculate_connectivity` rebuilds an `nx.Graph`, simplifies, and re-converts to rustworkx **every call**.
**Fix**: keep one `rx_simple_graph` updated incrementally on edge add; recompute connected components on it directly.

### 6.3 Incremental fragmentation via union-find

`_calculate_fragmentation` rebuilds an `nx.Graph` and runs `connected_components` **every step** (O(V+E) per step).
**Fix**: maintain a union-find over bike-lane endpoints; component counts/largest-component size update in O(α(N)) per edge add.

### 6.4 Heap-based candidate queue

`candidate_edges` is fully re-sorted every step (O(n log n) per step → O(n² log n) per episode), and the `connects_to_bike_path` flags are recomputed for **all** candidates every step.
**Fix**: use a bucket/heap keyed on `(connects_to_bike, road_priority)`; only update flags for candidates adjacent to newly-added `bike_nodes` (track a dirty set).

### 6.5 `nx_to_rx` weight-default loop is O(V·E)

The loop that defaults `length` is unnecessary because `weight_fn` already does `edge_data.get('length', 1.0)`. **Remove the loop.**

### 6.6 `_update_path_efficiency_incremental` runs up to 25 Dijkstras/step

Replace with a single multi-source Dijkstra or reduce neighbour sampling; make the sample count a `Config` field.

### 6.7 Make sampling thresholds configurable

The original hard-codes graph-size thresholds (`>10000/>5000/>1000/>100`) and sample counts (`log`, `sqrt`). Move them to `Config` with sane defaults and document the bias/variance tradeoff.

---

## 7. Tests Required (currently 0% coverage)

Use small **synthetic** NetworkX graphs as fixtures (no network access). Mock `ox.graph_from_place` / `ox.graph_from_bbox` in `conftest.py`.

| File                  | Covers                                                                                                                                                                                                    |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_graph_utils.py` | `nx_to_rx` round-trip preserves nodes/edges/weights; node map correctness; validation fills missing `length`; OSM loader caching + bbox/city branches                                                     |
| `test_candidates.py`  | length filter `100<L<1000`; road-priority ordering; **no mutation of source graph**; `connects_to_bike_path` propagation after an add                                                                     |
| `test_metrics.py`     | fragmentation/connectivity on fixtures with known answers; caching invalidates on graph change; population-served **actually moves** when a bike lane is added                                            |
| `test_env.py`         | budget exhaustion terminates; invalid/over-budget action handling; reward sign for connected vs isolated additions; `reset` restores state, budget, candidates, and `bike_nodes`; action mask correctness |
| `test_reward.py`      | budget-efficiency term is non-zero (pins §5.5 fix); continuity bonus only when `connects_to_bike_path`; isolation penalty applied once                                                                    |
| `test_training.py`    | `make_env` seeds differ; `train_model` calls `PPO.learn` with rounded timesteps (mock the model); reward callback records episodes including the last one                                                 |
| `test_cli.py`         | `main()` argparse: city/bbox mutual exclusivity; `--skip_training` loads model; `--no_plots` skips plotting (all mocked)                                                                                  |

Add `pytest-cov` and target **≥80% line coverage** on `bike_rl/` (excluding `cli.py`'s `__main__`). Add a CI workflow running `ruff check`, `ruff format --check`, `mypy`, `pytest`.

---

## 8. Modularity / Maintainability Requirements

- **No global mutable state.** Replace `RUN_ID` / `RUN_FIGURES_DIR` with a `RunContext` dataclass (run id, output dir, timestamp) created once in `cli.main()` and **explicitly passed** to training/evaluation/plotting. SubprocVecEnv workers receive serialised config, not globals.
- **Centralise all magic numbers** in `config.py` as a frozen `Config` dataclass:
  - reward weights: `w_connectivity=0.9, w_efficiency=0.2, w_population=0.3, reward_scale=100, road_priority_scale=5, continuity_bonus=50, fragmentation_weight=200, isolation_penalty=-100`
  - `edge_cost_factor=10`
  - candidate length bounds `min_len=100, max_len=1000`
  - road priorities dict
  - graph-size sampling thresholds + sample-count formulas
  - update-frequency thresholds
  - PPO hyperparameters
- **Logging over print.** Use the stdlib `logging` module with a console handler + optional file handler (for SLURM `.out`). Remove all `print` calls except the final CLI banner.
- **Type hints** on all public functions; `mypy --strict` clean on `bike_rl/`.
- **Docstrings** (Google or NumPy style) on every public function/class.
- **Remove dead/commented code** (the commented GeoJSON export block, `rx_to_nx`, unused `shimmy`).
- **GeoJSON export**: restore the commented-out result export behind a `--export_geojson` flag, writing to the run directory.
- **Headless-safe plotting**: `matplotlib.use("Agg")` by default; `show` opt-in.
- `.gitignore` already covers `__pycache__/` — keep it; also ignore `bike_path_figures/`, `osm_cache/`, `*.zip`, `checkpoints/`, `venv/`.

---

## 9. CLI (improved) — keep feature parity, fix the bugs

```
python -m bike_rl.cli --city "Otley, UK" --budget 300000 \
    --timesteps 51200 --eval_episodes 10 --n_envs 7 --device cpu \
    [--bbox N S E W] [--skip_training --model_path PATH] \
    [--no_plots] [--show] [--export_geojson] [--out_dir ./bike_path_figures]
```

Behaviours to preserve:

- Round timesteps up to `n_steps * n_envs` per update.
- Create a per-run output dir `<out_dir>/<safe_city>_<timestamp>/`.
- Save `final_model.zip`, `training_rewards.png`, `evaluation_metrics.png`, `evaluation_rewards.png`, best-solution map, and `run_summary.txt` in the run dir.
- Print total runtime as `Hh Mm Ss`.

Behaviours to add:

- `--show` to enable `plt.show()` (default off).
- `--export_geojson` to write `suggested_bike_paths.geojson`.
- `--seed` for full reproducibility (seeds Python, NumPy, torch, SB3, envs).
- `--config` to load a YAML/TOML `Config` override.

---

## 10. SLURM Scripts (improved)

Keep `run_bike_path_slurm.sh` and `submit_jobs.sh` array-job pattern (timestep sweep `[10240, 20480, 51200, 102400]`), but:

- Install from `requirements.txt` instead of a hand-maintained `pip install` line.
- Use the package entry point `python -m bike_rl.cli ...` (copy the installed package or `pip install -e .`).
- Set `MPLBACKEND=Agg` in the SLURM env.
- Pass `--out_dir $RUN_DIR` so each array task writes to its own dir (already the case) — keep the tarball step.

---

## 11. Implementation Order (recommended for the AI agent)

1. Scaffold the package layout (§4), `Config` + `RunContext` (§8), `requirements.txt` (§2), `pyproject.toml`, `.gitignore`, CI skeleton.
2. `graph_utils.py` (§3.3, §6.1, §6.5) + `candidates.py` (§5.8) with `test_graph_utils.py` / `test_candidates.py`.
3. `metrics.py` (§3.6, §6.2, §6.3, §6.7) with `test_metrics.py` — fix the population-served signal (§5.7).
4. `env.py` with **`MaskablePPO` action masking** (§5.4), fixed reward (§5.5), fixed incremental updates (§5.6), logging (§5.11), episode bookkeeping (§5.12), cache versioning (§5.14) + `test_env.py` / `test_reward.py`.
5. `training.py` (§3.7, §6.1 cache) + `test_training.py`.
6. `evaluation.py` (§3.8) + `plotting.py` (§5.10, headless-safe) + `test_cli.py`.
7. `cli.py` (§9) and SLURM scripts (§10).
8. Run `ruff`, `mypy`, `pytest --cov`; ensure ≥80% coverage and green CI.
9. Smoke test on a tiny bbox (e.g. a few blocks) to confirm end-to-end training + evaluation + figure output works headlessly.

---

## 12. Definition of Done

- `python -m bike_rl.cli --city "Otley, UK" --budget 300000 --timesteps 10240 --eval_episodes 3 --n_envs 2` runs end-to-end headless, writes all artefacts to a run dir, and exits cleanly.
- `pytest` green with ≥80% coverage on `bike_rl/`.
- `ruff check` and `mypy` clean.
- No `print` outside `cli.py`; no global mutable state; no bare `except`; no `plt.show()` by default.
- Action masking in place; the population-served and budget-efficiency reward terms are non-degenerate; `reset()` fully restores state across episodes.
- SLURM scripts submit and complete on a cluster with `MPLBACKEND=Agg`.
