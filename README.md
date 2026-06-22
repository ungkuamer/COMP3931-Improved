# Bike-Path RL

RL and direct-optimiser baselines for budgeted bike-network expansion.

**Status**: Scaffold — type-checked stub modules and CI in place; real
algorithm code to be added in subsequent plans.

## Spec documents

- [`RECREATE_SPEC.md`](RECREATE_SPEC.md) — RL pipeline specification
- [`OPTIMIZER_SPEC.md`](OPTIMIZER_SPEC.md) — Direct optimiser baselines
- [`RESEARCH_DIRECTION.md`](RESEARCH_DIRECTION.md) — Overall research direction

## Install

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Tooling

```bash
ruff check .            # Lint
ruff format --check .   # Format check
mypy --strict bike_rl   # Type check
pytest -q               # Run tests
```

## Repo layout

```
bike_rl/                # Main package
bike_rl/optim/          # Direct optimiser baselines (greedy, LS, ILP)
tests/                  # Smoke and unit tests
.github/workflows/      # CI
```

## License

MIT
