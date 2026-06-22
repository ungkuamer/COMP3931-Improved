# Plan 001: Scaffold the `bike_rl` package, Config, RunContext, deps, and CI

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**: `git diff --stat 0bdf56a..HEAD -- .`
> This repo starts as a spec-only repo (only `*.md`, `AGENTS.md`,
> `skills-lock.json`, `.git/`). If any of the in-scope paths below already
> exist with different content, compare against the "Current state" excerpts
> before proceeding; on a mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: LOW
- **Depends on**: none
- **Category**: tech-debt (foundational scaffold; enables all other plans)
- **Planned at**: commit `0bdf56a`, 2026-06-22
- **Issue**: _(not published)_

## Why this matters

Every subsequent plan (graph utils, candidates, metrics, env, training,
evaluation, CLI, and the optimiser baselines) imports from a shared
`bike_rl/` package and relies on a central `Config` for all magic numbers, a
`RunContext` to replace the original's broken global mutable state
(`RUN_ID`/`RUN_FIGURES_DIR`), pinned dependencies, and a CI workflow. Without
this scaffold there is no place to put any of that code, no import path, and no
verification gates. This plan creates the skeleton — package layout, type
stubs with full annotations and docstrings, the two frozen dataclasses, deps,
pyproject, gitignore, and CI — so that plans 002+ can fill in real logic module
by module without ever fighting the build.

It deliberately creates **stub modules that `raise NotImplementedError`** with
complete type signatures and docstrings. The goal is: the package imports
cleanly, `mypy --strict` passes on every signature, `ruff` is green, and a
smoke test runs in CI — before any real algorithm code exists.

## Current state

The repo is brand new. Working tree contents (verified at SHA `0bdf56a`):

```
AGENTS.md
OPTIMIZER_SPEC.md
RECREATE_SPEC.md
RESEARCH_DIRECTION.md
skills-lock.json
.git/
.agents/        (tooling, not source)
```

There is **no Python code, no `pyproject.toml`, no `requirements*.txt`, no
`bike_rl/`, no `tests/`, no `.github/`** yet. `AGENTS.md` says:

```
# AGENTS.md
Use cavemen skill.

### Git/Version Control
Use git commit conventions when writing commit messages.
```

The original reference implementation lives **outside this repo** at
`/home/ungku/programming/COMP3931/nx-rx-simple-ur.py` (1,582 lines, single
file). Do **not** copy it. The specs in this repo (`RECREATE_SPEC.md`,
`OPTIMIZER_SPEC.md`, `RESEARCH_DIRECTION.md`) are the source of truth for the
recreation. Read them when a step references a section (e.g. "§4"); they are
already in the repo root.

### Repo conventions to follow

- **Python 3.10+** (original ran on 3.12; pin `requires-python = ">=3.10"`).
- **Type hints on all public functions; `mypy --strict` clean on `bike_rl/`.**
- **Docstrings** (Google style) on every public function/class.
- **No `print`** outside `cli.py`. Use stdlib `logging`.
- **No global mutable state.** `Config` and `RunContext` are frozen dataclasses,
  passed explicitly.
- **Conventional Commits** for git messages (per `AGENTS.md`), e.g.
  `feat(scaffold): add bike_rl package skeleton and CI`.
- Module/file naming: lowercase snake_case for modules; PascalCase for classes;
  UPPER_SNAKE for constants.

### Spec vocabulary to honor (quote-checked)

`RECREATE_SPEC.md` §4 defines the target tree. `RESEARCH_DIRECTION.md` §4.1
mandates that the RL package and the optimiser baselines share one codebase:

> "Build **both** the RL pipeline (per `RECREATE_SPEC.md`) and the direct
> optimisers (per `OPTIMIZER_SPEC.md`) in a single shared codebase, using a
> **common objective module** (`objective.py`), **common candidate extraction**
> (`candidates.py`), and **common metrics** (`metrics.py`)."

So the scaffold must include `bike_rl/objective.py` and `bike_rl/optim/` from
the start (per `OPTIMIZER_SPEC.md` §2), even though their logic is filled in by
later plans. `OPTIMIZER_SPEC.md` §3.1 defines a **separate** frozen dataclass
`ObjectiveWeights` (not part of `Config`) — create it in `objective.py`.

## Commands you will need

| Purpose        | Command                          | Expected on success |
|---------------|----------------------------------|---------------------|
| Make venv     | `python3 -m venv .venv`          | `.venv/` created    |
| Activate      | `source .venv/bin/activate`      | shell prompt changes |
| Install (dev) | `pip install -e ".[dev]"`        | exit 0, installs `bike_rl` editable + dev deps |
| Ruff lint     | `ruff check .`                   | exit 0, "All checks passed" |
| Ruff format   | `ruff format --check .`          | exit 0 |
| Mypy          | `mypy --strict bike_rl`          | exit 0, no errors |
| Tests         | `pytest -q`                      | exit 0, ≥1 test passed |
| Import smoke  | `python -c "import bike_rl; from bike_rl.config import Config, RunContext; print(Config())"` | prints a Config repr, exit 0 |

Use a **clean Python 3.10+ environment**. Do not install into the system
interpreter. All commands run from repo root unless noted.

## Scope

**In scope** (the only files you should create or modify):
- `pyproject.toml`
- `requirements.txt`
- `requirements-dev.txt`
- `.gitignore`
- `.github/workflows/ci.yml`
- `bike_rl/__init__.py`
- `bike_rl/config.py`
- `bike_rl/run_context.py`
- `bike_rl/graph_utils.py` (stub)
- `bike_rl/candidates.py` (stub)
- `bike_rl/metrics.py` (stub)
- `bike_rl/objective.py` (stub + `ObjectiveWeights` dataclass)
- `bike_rl/env.py` (stub)
- `bike_rl/training.py` (stub)
- `bike_rl/evaluation.py` (stub)
- `bike_rl/plotting.py` (stub)
- `bike_rl/cli.py` (stub — full `main()` comes in plan 007)
- `bike_rl/optim/__init__.py`
- `bike_rl/optim/greedy.py` (stub)
- `bike_rl/optim/local_search.py` (stub)
- `bike_rl/optim/ilp.py` (stub)
- `bike_rl/optim/budget.py` (stub)
- `bike_rl/optim/evaluate.py` (stub)
- `tests/__init__.py`
- `tests/conftest.py`
- `tests/test_smoke.py`
- `README.md` (create a minimal project README)

**Out of scope** (do NOT touch, even though they look related):
- `scripts/run_bike_path_slurm.sh`, `scripts/submit_jobs.sh` — plan 007.
- Any real algorithm implementation inside `graph_utils.py`/`candidates.py`/
  `metrics.py`/`env.py`/`training.py`/`evaluation.py`/`plotting.py`/
  `cli.py`/`objective.py`/`optim/*.py`. Stubs only. Real logic is plans 002–006
  and optimiser plans.
- The existing `*.md` spec files and `AGENTS.md` — do not edit.
- `.agents/`, `skills-lock.json` — tooling, leave alone.

## Git workflow

- Branch: `advisor/001-scaffold-package`
- Commit per step (or per logical cluster). Message style: Conventional Commits,
  e.g. `feat(scaffold): add pyproject and pinned requirements`, then
  `feat(scaffold): add Config and RunContext dataclasses`, etc.
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Create `.gitignore`

Write `.gitignore` at repo root. Must ignore at minimum:

```
# Python
__pycache__/
*.py[cod]
*.egg-info/
.eggs/
build/
dist/

# Virtual envs
.venv/
venv/
env/

# Tooling caches
.mypy_cache/
.ruff_cache/
.pytest_cache/
.coverage
htmlcov/

# Project artefacts (per RECREATE_SPEC §8)
bike_path_figures/
osm_cache/
checkpoints/
*.zip

# OS / editors
.DS_Store
.idea/
.vscode/
```

**Verify**: `test -f .gitignore && grep -q "__pycache__" .gitignore && grep -q "bike_path_figures/" .gitignore && echo OK` → prints `OK`.

### Step 2: Create `requirements.txt` and `requirements-dev.txt`

`requirements.txt` — the corrected, pinned runtime deps from `RECREATE_SPEC.md`
§2 **plus** the extras the optimiser and CLI need (per `OPTIMIZER_SPEC.md` §7
for ILP and `RECREATE_SPEC.md` §9 `--config` for YAML/TOML):

```
osmnx>=1.9
networkx>=3.2
gymnasium>=0.29
stable-baselines3>=2.2
sb3-contrib>=2.2
rustworkx>=0.14
numpy>=1.26
pandas>=2.1
scipy>=1.11
geopandas>=0.14
shapely>=2.0
matplotlib>=3.8
tqdm>=4.66
pyyaml>=6.0
ortools>=9.10
```

Notes for the executor:
- `gym` → `gymnasium` (fixes §5.3). `shimmy` dropped (unused). `sb3-contrib`
  added for `MaskablePPO` (§5.4). `ortools` added for the ILP oracle
  (`OPTIMIZER_SPEC.md` §7). `pyyaml` added for the `--config` flag (§9).
- Do **not** add `gym`, `shimmy`, or `pulp` (PuLP is a documented fallback only;
  defer to the ILP plan).

`requirements-dev.txt`:

```
pytest>=8.0
pytest-cov>=4.1
ruff>=0.4
mypy>=1.8
```

**Verify**: `test -f requirements.txt && test -f requirements-dev.txt && ! grep -qE "^gym\b|^shimmy" requirements.txt && grep -q "ortools" requirements.txt && grep -q "sb3-contrib" requirements.txt && echo OK` → prints `OK`.

### Step 3: Create `pyproject.toml`

Use a standard setuptools layout (no `src/` dir — the spec's tree puts
`bike_rl/` at repo root). Include ruff, mypy, and pytest config inline so there
is one source of tooling config.

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "bike-path-rl"
version = "0.1.0"
description = "RL and direct-optimiser baselines for budgeted bike-network expansion"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
authors = [{ name = "COMP3931-Improved" }]
dependencies = [
    "osmnx>=1.9",
    "networkx>=3.2",
    "gymnasium>=0.29",
    "stable-baselines3>=2.2",
    "sb3-contrib>=2.2",
    "rustworkx>=0.14",
    "numpy>=1.26",
    "pandas>=2.1",
    "scipy>=1.11",
    "geopandas>=0.14",
    "shapely>=2.0",
    "matplotlib>=3.8",
    "tqdm>=4.66",
    "pyyaml>=6.0",
    "ortools>=9.10",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=4.1",
    "ruff>=0.4",
    "mypy>=1.8",
]

[project.scripts]
bike-rl = "bike_rl.cli:main"

[tool.setuptools.packages.find]
include = ["bike_rl*"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "D"]
ignore = ["D100", "D104", "D105"]  # module/package/missing-docstring-in-magic-method

[tool.ruff.lint.pydocstyle]
convention = "google"

[tool.ruff.format]
quote-style = "double"

[tool.mypy]
python_version = "3.10"
strict = true
warn_unused_ignores = true
warn_redundant_casts = true

[[tool.mypy.overrides]]
module = ["rustworkx.*", "osmnx.*", "stable_baselines3.*", "sb3_contrib.*", "ortools.*", "geopandas.*", "shapely.*"]
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"
```

Notes:
- `dependencies` mirror `requirements.txt` so `pip install -e .` is
  self-sufficient. Keep them in sync; the plan's Done criteria checks both.
- `mypy` strict mode is on for `bike_rl`. Third-party libs without stubs
  (rustworkx, osmnx, sb3, ortools, geopandas, shapely) are listed under
  `ignore_missing_imports` so missing stubs don't block CI. **Do not** blanket
  `ignore_missing_imports = true` globally — only per-module.
- The CLI entry point `bike-rl = "bike_rl.cli:main"` requires `cli.py` to expose
  a `main` function (the stub in Step 9 provides it).

**Verify**: `python -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))" && echo OK` → prints `OK` (Python 3.11+ has `tomllib`; on 3.10 use `pip install tomli && python -c "import tomli; tomli.load(open('py_tool...')"` — simpler: just run `python -m pip install -e ".[dev]"` in Step 4 which validates the file).

### Step 4: Install the package editable and confirm it imports

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

Then create the **empty `bike_rl/__init__.py`** first (next step) so the import
works — actually do Step 5's `__init__.py` before this verify, but you may run
the install after the package files exist. **Order**: create `bike_rl/__init__.py`
and the stubs (Steps 5–9) first, then run the install + verification here.

Reordered for safety: do Steps 5–9 (write all `bike_rl/` files), then come back
and run:

```bash
pip install -e ".[dev]"
python -c "import bike_rl; from bike_rl.config import Config; from bike_rl.run_context import RunContext; print('import OK')"
```

**Verify**: the `python -c` line prints `import OK`, exit 0. `pip install` exits 0.

### Step 5: Create `bike_rl/__init__.py`

```python
"""bike_rl: RL and direct-optimiser baselines for budgeted bike-network expansion.

Shared modules (candidates, metrics, objective, graph_utils) are consumed by
both the Gymnasium RL environment and the direct optimisers (greedy / local
search / ILP) so the two approaches solve exactly the same problem.
"""

__version__ = "0.1.0"
```

**Verify**: `python -c "import bike_rl; print(bike_rl.__version__)"` → prints `0.1.0`.

### Step 6: Create `bike_rl/config.py` — the frozen `Config` dataclass

This centralises **every magic number** from `RECREATE_SPEC.md` §8 plus the
optimiser thresholds from `OPTIMIZER_SPEC.md` §§5–7. Use `@dataclass(frozen=True)`
with `field(default_factory=...)` for mutable defaults (dicts). Group fields with
comments. Every field needs a docstring via `#: ...` inline comments or a class
docstring that lists them (Google style class docstring).

Required fields (verify against this list — do not omit):

**Reward weights** (from original lines cited in recon: `0.9, 0.2, 0.3`, scale
`100`, road_priority scale `5`, continuity `50`, fragmentation `200`,
isolation `-100`):
- `w_connectivity: float = 0.9`
- `w_efficiency: float = 0.2`
- `w_population: float = 0.3`
- `reward_scale: float = 100.0`
- `road_priority_scale: float = 5.0`
- `continuity_bonus: float = 50.0`
- `fragmentation_weight: float = 200.0`
- `isolation_penalty: float = -100.0`

**Cost / budget**:
- `edge_cost_factor: float = 10.0`  (cost = length * factor)

**Candidate filtering** (original line 317: `length > 100 and length < 1000`):
- `min_candidate_length: float = 100.0`
- `max_candidate_length: float = 1000.0`
- `road_priorities: dict[str, int]` via `field(default_factory=lambda: {"primary": 5, "secondary": 4, "tertiary": 3, "residential": 2, "unclassified": 1})`

**Population-served proxy** (§5.7 — "nodes within X metres of any bike lane"):
- `coverage_radius_m: float = 300.0`  # documented proxy; tune later
- `coverage_mode: str = "radius"`  # or "component"; documents the chosen proxy

**Metrics sampling** (original lines 435–493): graph-size thresholds and
sample-count formulas. Encode as nested dataclasses or plain fields:
- `sampling_thresholds: dict[str, int]` via default_factory with keys
  `"huge": 10000, "large": 5000, "medium": 1000, "small": 100`.
- `clustering_sample_sizes: dict[str, int]` default
  `{"huge": 40, "large": 60}` (original lines 437/454).
- `path_sample_formula: str = "log"`  # document: >10000/>5000 use log-based;
  >1000 uses log*1.5; >100 uses sqrt. Keep formula names as documented constants.

**Update frequency** (original lines 236–241):
- `update_freq_huge: int = 5`   # graph_size > 5000
- `update_freq_large: int = 3`  # graph_size > 1000
- `update_freq_default: int = 1`

**PPO hyperparameters** (original lines 1174–1179, §3.7):
- `ppo_policy: str = "MlpPolicy"`
- `ppo_learning_rate: float = 3e-4`
- `ppo_n_steps: int = 2048`
- `ppo_batch_size: int = 64`
- `ppo_n_epochs: int = 10`
- `ppo_gamma: float = 0.99`
- `ppo_gae_lambda: float = 0.95`
- `ppo_clip_range: float = 0.2`
- `device: str = "cpu"`

**Optimiser thresholds** (`OPTIMIZER_SPEC.md` §§6–7):
- `local_search_max_iter: int = 1000`
- `local_search_time_limit_s: float = 60.0`
- `ilp_time_limit_s: float = 300.0`
- `ilp_default_solver: str = "ortools"`
- `max_candidates_for_ilp: int = 300`  # above this, ILP skipped (documented)

**Reproducibility** (§9 `--seed`):
- `seed: int = 0`

Also add a class docstring (Google style) explaining that `Config` is the single
source of magic numbers, is frozen, and is passed explicitly to every
component (no globals).

Add one helper method:
```python
def reward_weights(self) -> tuple[float, float, float]:
    """Return (connectivity, efficiency, population) reward weights."""
    return (self.w_connectivity, self.w_efficiency, self.w_population)
```

**Verify**: `python -c "from bike_rl.config import Config; c=Config(); assert c.edge_cost_factor==10.0 and c.continuity_bonus==50.0 and c.road_priorities['primary']==5 and c.max_candidates_for_ilp==300 and c.coverage_radius_m==300.0; print('config OK')"` → prints `config OK`. Then `mypy --strict bike_rl/config.py` → exit 0.

### Step 7: Create `bike_rl/run_context.py` — the frozen `RunContext` dataclass

Replaces the original's global `RUN_ID` / `RUN_FIGURES_DIR` (§5.13, §8). Per
`RECREATE_SPEC.md` §8: "created once in `cli.main()` and explicitly passed to
training/evaluation/plotting. SubprocVecEnv workers receive serialised config,
not globals."

```python
"""RunContext: per-run output directory and identity, replacing global state."""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class RunContext:
    """Per-run context: identity and output directory.

    Attributes:
        run_id: Stable run identifier (e.g. ``<safe_city>_<timestamp>``).
        output_dir: Directory for this run's artefacts (model, figures, summary).
        timestamp: When the run started (UTC).
    """

    run_id: str
    output_dir: Path
    timestamp: datetime

    @classmethod
    def create(cls, base_dir: Path, label: str, timestamp: datetime | None = None) -> "RunContext":
        """Create a RunContext with a ``<label>_<timestamp>`` id under ``base_dir``.

        The directory is NOT created here; callers create it lazily so dry-runs
        and tests don't litter the filesystem. Use ``ensure_output_dir()``.
        """
        ts = timestamp or datetime.utcnow()
        run_id = f"{label}_{ts.strftime('%Y%m%d_%H%M%S')}"
        return cls(run_id=run_id, output_dir=base_dir / run_id, timestamp=ts)

    def ensure_output_dir(self) -> Path:
        """Create ``output_dir`` if missing and return it."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir
```

**Verify**: `python -c "import tempfile,datetime; from pathlib import Path; from bike_rl.run_context import RunContext; d=Path(tempfile.mkdtemp()); rc=RunContext.create(d,'otley'); rc.ensure_output_dir(); assert rc.output_dir.exists() and rc.run_id.startswith('otley_'); print('runctx OK')"` → prints `runctx OK`. `mypy --strict bike_rl/run_context.py` → exit 0.

### Step 8: Create `bike_rl/objective.py` — `ObjectiveWeights` + stub

`OPTIMIZER_SPEC.md` §3.1 defines `ObjectiveWeights` as a **separate** frozen
dataclass (not in `Config`). Create it here, plus stub signatures for the
objective functions (real logic is a later plan).

```python
"""Canonical scalar objective — the single source of truth for 'what we optimise'.

Both the RL reward (derived) and the direct optimisers (direct call) use this
module so RL and optimisers solve exactly the same problem. See
OPTIMIZER_SPEC.md §3 and RESEARCH_DIRECTION.md §4.1–4.2.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bike_rl.config import Config


@dataclass(frozen=True)
class ObjectiveWeights:
    """Weights for the canonical objective (OPTIMIZER_SPEC §3.1).

    Attributes:
        connectivity: Weight on graph connectivity in [0,1].
        coverage: Weight on bike-lane-reachable population (coverage) in [0,1].
        fragmentation: Weight on the fragmentation penalty in [0,1]; subtracted.
    """

    connectivity: float = 0.4
    coverage: float = 0.4
    fragmentation: float = 0.2


def objective(graph: "object", added_edges: "list", weights: ObjectiveWeights, cfg: "Config") -> float:
    """Return the canonical objective value of ``graph`` with ``added_edges``.

    Higher is better. Pure and deterministic.

    Args:
        graph: Base bike graph (NetworkX MultiDiGraph).
        added_edges: Chosen candidate edges applied on top of ``graph``.
        weights: Objective component weights.
        cfg: Config (carries metric parameters).

    Returns:
        The scalar objective value.

    Raises:
        NotImplementedError: Implementation lands in a later plan.
    """
    raise NotImplementedError


def objective_delta(graph: "object", added_edges: "list", new_edge: "tuple", weights: ObjectiveWeights, cfg: "Config") -> float:
    """Return the marginal objective change from adding ``new_edge``.

    Must match ``objective(..., S + [new_edge]) - objective(..., S)``. Backed by
    incremental data structures in ``metrics.py``.

    Raises:
        NotImplementedError: Implementation lands in a later plan.
    """
    raise NotImplementedError


def apply_added_edges(graph: "object", added_edges: "list") -> "object":
    """Return a copy of ``graph`` with ``added_edges`` applied (bike_lane='yes').

    Raises:
        NotImplementedError: Implementation lands in a later plan.
    """
    raise NotImplementedError
```

Type the graph/edge parameters loosely as `object` with quotes for now (real
typed aliases arrive with the metrics/graph_utils plans). The point is: mypy
strict passes and the API is documented.

**Verify**: `python -c "from bike_rl.objective import ObjectiveWeights; w=ObjectiveWeights(); assert abs(w.connectivity-0.4)<1e-9 and abs(w.coverage-0.4)<1e-9 and abs(w.fragmentation-0.2)<1e-9; print('objweights OK')"` → prints `objweights OK`. `mypy --strict bike_rl/objective.py` → exit 0.

### Step 9: Create the remaining `bike_rl/*.py` stubs

Each stub file: a module docstring, the public class/function **signatures with
full type annotations and Google docstrings**, and a `raise NotImplementedError`
body. No real logic. These are the contracts later plans implement.

For each file below, follow this pattern (example for `graph_utils.py`):

```python
"""<one-line module purpose, referencing the spec section>.

See RECREATE_SPEC.md §<n> for behaviour.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx
    import rustworkx as rx


def nx_to_rx(nx_graph: "nx.MultiDiGraph") -> "tuple[rx.PyDiGraph, dict[int | str, int]]":
    """Convert a NetworkX MultiDiGraph to a rustworkx PyDiGraph.

    Builds a node map (original node id -> rustworkx index) using the
    ``__networkx_node__`` attribute. Missing ``length`` defaults to 1.0 via the
    rustworkx weight_fn (do NOT run an O(V*E) defaulting loop — RECREATE_SPEC §6.5).

    Args:
        nx_graph: Source NetworkX graph.

    Returns:
        (rustworkx graph, node_map).

    Raises:
        NotImplementedError: Implementation lands in plan 002.
    """
    raise NotImplementedError
```

Create these files with the listed public signatures (use the spec sections to
write accurate docstrings; signatures below are the minimum contract):

- **`bike_rl/graph_utils.py`** (§3.3, §6.1, §6.5):
  - `def nx_to_rx(nx_graph) -> tuple[PyDiGraph, dict]`
  - `def load_city_graph(city_name: str, cfg: Config) -> nx.MultiDiGraph`
  - `def load_bbox_graph(north: float, south: float, east: float, west: float, cfg: Config) -> nx.MultiDiGraph`  # note (N,S,E,W) order per §3.1
  - `def validate_graph(graph) -> None`  # fills missing length, raises on invalid
  - `def cache_graph(graph, path: Path) -> None` and `def load_cached_graph(path: Path) -> nx.MultiDiGraph`  # §6.1

- **`bike_rl/candidates.py`** (§3.2, §5.8):
  - `@dataclass(frozen=True) class Candidate:` with fields `u: int|str`,
    `v: int|str`, `length: float`, `road_priority: int`,
    `connects_to_bike_path: bool`, `data: dict` (copied — never mutate source).
  - `def extract_candidates(bike_graph, walk_graph, cfg: Config) -> list[Candidate]`
  - `def recompute_connects(candidates: list[Candidate], bike_nodes: set) -> list[Candidate]`

- **`bike_rl/metrics.py`** (§3.6, §6.2, §6.3, §6.7):
  - `def connectivity(graph, cfg: Config) -> float`
  - `def path_efficiency(graph, cfg: Config) -> float`
  - `def fragmentation(graph, cfg: Config) -> float`
  - `def coverage(graph, cfg: Config) -> float`  # the FIXED population-served, §5.7

- **`bike_rl/env.py`** (§3.4, §5.4):
  - `class BikePathEnv(gymnasium.Env):` with `__init__(self, bike_graph, walk_graph, cfg: Config, run_context: RunContext)`, `step(self, action)`, `reset(self, *, seed=None, options=None)`, `action_masks(self) -> np.ndarray`. Import `gymnasium` and `numpy` at top (not under TYPE_CHECKING — they're runtime). `gymnasium` may lack strict stubs; that's covered by the mypy override.

- **`bike_rl/training.py`** (§3.7):
  - `def make_env(bike_graph, walk_graph, cfg: Config, run_context: RunContext, rank: int, seed: int) -> gymnasium.Env`
  - `def train_model(envs, cfg: Config, run_context: RunContext, total_timesteps: int) -> object`  # returns a PPO/MaskablePPO model
  - `class TrainingProgressCallback:` (stub `__init__`, `_on_step`)

- **`bike_rl/evaluation.py`** (§3.8):
  - `@dataclass class EvaluationTracker:` with reward/paths/connectivity/efficiency/population/budget list fields and a `best` tracking method.
  - `def evaluate_and_visualize(model, bike_graph, walk_graph, cfg: Config, run_context: RunContext, num_evaluations: int) -> EvaluationTracker`

- **`bike_rl/plotting.py`** (§5.10, headless-safe):
  - At module top: `import matplotlib; matplotlib.use("Agg")` BEFORE `import matplotlib.pyplot as plt` (§5.10).
  - `def render_solution_map(graph, added_paths, run_context: RunContext, show: bool = False) -> Path`
  - `def plot_metrics(tracker: EvaluationTracker, run_context: RunContext, show: bool = False) -> None`
  - `def plot_rewards(rewards: list[float], run_context: RunContext, show: bool = False) -> None`

- **`bike_rl/cli.py`** (§9): provide a minimal `main()` that parses args with
  `argparse` (city/bbox mutex, budget, timesteps, eval_episodes, n_envs,
  device, skip_training, model_path, no_plots, show, export_geojson, seed,
  config, out_dir) and then `raise NotImplementedError("Full CLI in plan 007")`
  after parsing. This makes the `bike-rl` entry point importable and the
  `--help` work, while leaving end-to-end execution to plan 007.
  - `def main() -> int:` ... `return 0`

**For every stub**: full type annotations, Google docstring referencing the
spec section, `raise NotImplementedError("...plan NNN")` body. Use
`from __future__ import annotations` so forward refs and `int | str` work on
3.10. Import heavy third-party libs (`networkx`, `rustworkx`, `gymnasium`,
`numpy`, `matplotlib`, `stable_baselines3`) at module top **only if needed at
import time** (e.g. `env.py` subclasses `gymnasium.Env`, `plotting.py` must set
the Agg backend, `training.py` needs the base class for type hints). For
signature-only references, prefer `TYPE_CHECKING` imports to keep stub import
cheap. `mypy --strict` must pass on all of them.

**Verify** (after all stubs written):
```bash
python -c "import bike_rl.graph_utils, bike_rl.candidates, bike_rl.metrics, bike_rl.env, bike_rl.training, bike_rl.evaluation, bike_rl.plotting, bike_rl.cli, bike_rl.objective; print('all stubs import OK')"
mypy --strict bike_rl
```
→ first prints `all stubs import OK`; second exits 0 with no errors.

### Step 10: Create `bike_rl/optim/` stubs

`bike_rl/optim/__init__.py`:
```python
"""Direct optimiser baselines: greedy, local search, ILP.

Shares candidates/metrics/objective with the RL env so both solve the same
problem. See OPTIMIZER_SPEC.md.
"""
```

Create these stubs (same pattern as Step 9 — annotations + docstrings +
`raise NotImplementedError`):

- **`bike_rl/optim/budget.py`**:
  - `def cost(edge, cfg: Config) -> float`  # length * edge_cost_factor
  - `def remaining_budget(spent: float, budget: float) -> float`

- **`bike_rl/optim/greedy.py`** (OPTIMIZER_SPEC §5):
  - `@dataclass class Solution:` with `edges: list`, `objective: float`, `spent: float`, `runtime_s: float`, `solver: str`, `extra: dict`.
  - `class GreedySolver:` with `__init__(self, cfg: Config, weights: ObjectiveWeights)` and `solve(self, graph, candidates: list, budget: float) -> Solution`.

- **`bike_rl/optim/local_search.py`** (§6):
  - `class LocalSearchSolver:` with `__init__(self, cfg: Config, weights: ObjectiveWeights, max_iter: int | None = None, time_limit_s: float | None = None)` and `solve(...) -> Solution`.

- **`bike_rl/optim/ilp.py`** (§7):
  - `class ILPSolver:` with `__init__(self, cfg: Config, weights: ObjectiveWeights, solver: str | None = None, time_limit_s: float | None = None)` and `solve(...) -> Solution`.

- **`bike_rl/optim/evaluate.py`** (§8):
  - `def evaluate_solver(solver, instance) -> Solution`
  - `def evaluate_rl_policy(policy, instance, budget: float) -> Solution`

Put `Solution` in `greedy.py` and re-export from `evaluate.py` and
`__init__.py` (`from bike_rl.optim.greedy import Solution`) so callers can do
`from bike_rl.optim import GreedySolver, LocalSearchSolver, ILPSolver, Solution`.

**Verify**:
```bash
python -c "from bike_rl.optim import GreedySolver, LocalSearchSolver, ILPSolver, Solution; print('optim stubs OK')"
mypy --strict bike_rl/optim
```
→ prints `optim stubs OK`; mypy exits 0.

### Step 11: Create `tests/` skeleton

`tests/__init__.py`: empty file (or a one-line docstring).

`tests/conftest.py` — minimal now; later plans expand it. Provide one tiny
synthetic graph fixture and a placeholder for OSM mocks:

```python
"""Shared pytest fixtures.

Later plans add OSM mocks and richer synthetic graphs. For now, provide a
minimal fixture so the smoke test and CI have something to run.
"""

from __future__ import annotations
import networkx as nx
import pytest


@pytest.fixture
def tiny_bike_graph() -> nx.MultiDiGraph:
    """A 4-node synthetic bike graph for smoke tests (no network access)."""
    g = nx.MultiDiGraph()
    g.add_node(1, x=0.0, y=0.0)
    g.add_node(2, x=0.001, y=0.0)
    g.add_node(3, x=0.0, y=0.001)
    g.add_node(4, x=0.001, y=0.001)
    g.add_edge(1, 2, length=120.0, highway="residential", bike_lane="yes")
    g.add_edge(3, 4, length=130.0, highway="residential", bike_lane="yes")
    return g
```

`tests/test_smoke.py`:
```python
"""Smoke tests: the package imports and core dataclasses instantiate."""

from bike_rl.config import Config
from bike_rl.run_context import RunContext
from bike_rl.objective import ObjectiveWeights


def test_config_defaults():
    cfg = Config()
    assert cfg.edge_cost_factor == 10.0
    assert cfg.road_priorities["primary"] == 5
    assert cfg.max_candidates_for_ilp == 300


def test_objective_weights_defaults():
    w = ObjectiveWeights()
    assert abs(w.connectivity - 0.4) < 1e-9
    assert abs(w.coverage - 0.4) < 1e-9
    assert abs(w.fragmentation - 0.2) < 1e-9


def test_run_context_create(tmp_path):
    rc = RunContext.create(tmp_path, "test")
    assert rc.run_id.startswith("test_")
    rc.ensure_output_dir()
    assert rc.output_dir.exists()
```

**Verify**: `pytest -q` → 3 passed, exit 0. `pytest --cov=bike_rl --cov-report=term-missing -q` runs without error (coverage will be low; that's expected — the ≥80% target is a later-plan done criterion, not this plan's).

### Step 12: Create `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [main, master]
  pull_request:

jobs:
  lint-type-test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
      - name: Ruff lint
        run: ruff check .
      - name: Ruff format check
        run: ruff format --check .
      - name: Mypy
        run: mypy --strict bike_rl
      - name: Pytest
        run: pytest -q
```

Notes:
- `geopandas`/`shapely`/`osmnx` system deps: on `ubuntu-latest` these install
  fine via pip wheels for recent versions. If CI fails on `geopandas` native
  build, add `apt-get` libgeos/geos steps — but try pip wheels first; do not
  pre-emptively add system packages.
- Matrix 3.10 + 3.12 covers the floor and the original's version.

**Verify**: `test -f .github/workflows/ci.yml && python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))" && echo OK` (requires `pyyaml`, installed via dev deps) → prints `OK`. (Full CI validation happens on push; locally we only confirm the YAML parses and the same commands run by hand.)

### Step 13: Create `README.md` (minimal)

A short project README so `pyproject.toml`'s `readme = "README.md"` resolves
and the repo has a landing page. Keep it under ~60 lines:

- Title + one-line description (from `RESEARCH_DIRECTION.md` §1).
- Status: work-in-progress scaffold.
- Links to the three spec docs.
- Install: `python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`.
- Commands: `ruff check .`, `mypy --strict bike_rl`, `pytest -q`.
- Repo layout (the tree from RECREATE_SPEC §4 + optim additions).
- License: MIT.

**Verify**: `test -f README.md && grep -q "bike_rl" README.md && echo OK` → prints `OK`.

### Step 14: Run the full local verification suite

```bash
source .venv/bin/activate
ruff check .
ruff format --check .
mypy --strict bike_rl
pytest -q
python -c "import bike_rl; from bike_rl.config import Config; from bike_rl.run_context import RunContext; from bike_rl.objective import ObjectiveWeights; from bike_rl.optim import GreedySolver, LocalSearchSolver, ILPSolver, Solution; print('full import OK')"
```

**Verify**: all five commands exit 0; the last prints `full import OK`.

### Step 15: Commit

Stage everything and commit in logical chunks, e.g.:

```
git add .gitignore pyproject.toml requirements.txt requirements-dev.txt README.md
git commit -m "chore(scaffold): add pyproject, pinned deps, gitignore, readme"

git add bike_rl/
git commit -m "feat(scaffold): add bike_rl package skeleton, Config, RunContext, stubs"

git add tests/
git commit -m "test(scaffold): add smoke tests and conftest fixture"

git add .github/
git commit -m "ci(scaffold): add ruff+fmt+mypy+pytest workflow"
```

**Verify**: `git log --oneline -5` shows the four commits; `git status` clean.

## Test plan

- New tests (this plan): `tests/test_smoke.py` — 3 tests covering `Config`
  defaults, `ObjectiveWeights` defaults, `RunContext` creation/dir. Structural
  pattern for later test files: small, no network, plain asserts.
- No existing tests to model on (repo starts at 0% coverage). The conftest
  fixture `tiny_bike_graph` is the seed fixture later plans extend.
- Verification: `pytest -q` → 3 passed. Coverage is intentionally not gated
  here (stubs raise NotImplementedError); the ≥80% target belongs to plans
  002–008.

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `pip install -e ".[dev]"` exits 0
- [ ] `ruff check .` exits 0
- [ ] `ruff format --check .` exits 0
- [ ] `mypy --strict bike_rl` exits 0 with no errors
- [ ] `pytest -q` exits 0 with ≥3 tests passing
- [ ] `python -c "import bike_rl; from bike_rl.config import Config; from bike_rl.run_context import RunContext; from bike_rl.objective import ObjectiveWeights; from bike_rl.optim import GreedySolver, LocalSearchSolver, ILPSolver, Solution"` exits 0
- [ ] `.github/workflows/ci.yml` exists and parses as YAML
- [ ] `requirements.txt` contains `gymnasium`, `sb3-contrib`, `ortools` and does NOT contain `gym` or `shimmy` (`grep -qE "^gym\b" requirements.txt` returns no match)
- [ ] `bike_rl/config.py` `Config` is `@dataclass(frozen=True)` and contains every field listed in Step 6 (spot-check: `edge_cost_factor==10.0`, `continuity_bonus==50.0`, `road_priorities['primary']==5`, `coverage_radius_m==300.0`, `max_candidates_for_ilp==300`, `ppo_n_steps==2048`)
- [ ] `bike_rl/run_context.py` `RunContext` is frozen and has `create` + `ensure_output_dir`
- [ ] No real algorithm implementation exists in any stub (every stub body is `raise NotImplementedError`): `grep -rn "NotImplementedError" bike_rl | wc -l` ≥ 15
- [ ] `git status` clean; commits made with Conventional Commits messages
- [ ] No files outside the in-scope list are modified (`git status` shows only the listed paths)
- [ ] `plans/README.md` status row for 001 updated

## STOP conditions

Stop and report back (do not improvise) if:

- Any in-scope path already exists in the repo with conflicting content (the
  repo was expected to be spec-only at SHA `0bdf56a`).
- `pip install -e ".[dev]"` fails on a dependency that genuinely has no
  Python-3.10-compatible wheel (report which one; do not silently drop or
  downgrade pinned deps without confirmation).
- `geopandas`/`osmnx`/`ortools` fail to install in CI and the fix requires
  system packages not available in the GitHub Actions `ubuntu-latest` image —
  report and propose the apt step rather than adding it blind.
- `mypy --strict` reports errors that cannot be resolved by adding type
  annotations to stub signatures (e.g. a third-party library fundamentally
  incompatible with strict mode) — report the specific error rather than
  loosening the global strict setting.
- A stub's spec-referenced signature turns out to be ambiguous (e.g. the spec
  doesn't pin a return type) — make a minimal documented choice and note it in
  the commit message, but if it changes the public API shape meaningfully,
  STOP and report instead.

## Maintenance notes

For whoever owns this code after the change lands:

- **`Config` is the single source of magic numbers.** Any later plan that's
  tempted to hardcode a threshold should add a `Config` field instead. Review
  PRs for new literals sneaking into `bike_rl/`.
- **`ObjectiveWeights` lives in `objective.py`, not `Config`**, by design
  (OPTIMIZER_SPEC §3.1) so the objective family can vary independently of the
  RL hyperparameters. Don't merge them.
- **The mypy per-module `ignore_missing_imports` override list** will need
  pruning as stubs become available (e.g. if `rustworkx` ships types later).
  Review that list when bumping deps.
- **Stubs `raise NotImplementedError`.** Plans 002 (graph_utils/candidates),
  003 (metrics), 004 (env/reward), 005 (training), 006 (evaluation/plotting),
  007 (cli/SLURM), and the optimiser plans replace them one module at a time.
  Keep stubs stub-only until their plan — don't half-implement.
- **CI matrix (3.10, 3.12)**: if a future dep drops 3.10 support, bump
  `requires-python` and the matrix together; don't let them drift.
- **Reviewer focus**: confirm every stub has a docstring referencing the spec
  section (so the executor of the next plan can find the behaviour spec), and
  that `Config` fields match the original's magic numbers cited in Step 6.
