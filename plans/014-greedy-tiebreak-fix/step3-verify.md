# Step 3: Confirm the full QA gate and that the reproduction is gone

Run the whole gate end-to-end, then confirm the pre-fix reproduction no longer
raises. All from the repo root with the venv active.

### Run the gate

```bash
pytest -q
ruff check .
ruff format --check .
mypy --strict bike_rl
```

All four must exit 0 (or report "All checks passed!" / "Success"); `pytest -q`
must show whole-package coverage ≥80%.

### Run the reproduction script

```bash
python -c "
import networkx as nx
from bike_rl.candidates import Candidate
from bike_rl.config import Config
from bike_rl.objective import ObjectiveWeights
from bike_rl.optim.greedy import GreedySolver
def _edge(u, v, length, rp=1):
    return Candidate(u=u, v=v, length=float(length), road_priority=rp, connects_to_bike_path=False, data={'length': float(length), 'highway': 'residential'})
g = nx.MultiDiGraph()
for n, (x, y) in {1: (0, 0), 2: (0.01, 0), 10: (0.02, 0), 11: (0.03, 0)}.items():
    g.add_node(n, x=x, y=y)
g.add_edge(1, 2, length=120.0, highway='residential', bike_lane='yes')
sol = GreedySolver(Config(), ObjectiveWeights()).solve(g, [_edge(1, 10, 100.0, 5), _edge(2, 11, 100.0, 5)], 1000.0)
print('OK', len(sol.edges), sol.spent)
"
```

Expected output: `OK 1 1000.0` — no `TypeError`, no traceback.

### Confirm the done-criteria greps

```bash
grep -n "_, _, _, best = max(scored)" bike_rl/optim/greedy.py   # must return NO matches
grep -n "_, _, _, _, best" bike_rl/optim/greedy.py              # must return the new unpack
grep -n "scored: list\[tuple\[float, int, float, Candidate\]\]" bike_rl/optim/greedy.py  # must return NO matches
git status --short   # only bike_rl/optim/greedy.py, tests/test_greedy.py, (+plans/README.md after you edit it)
```

### Finish

Update `plans/README.md`: change plan 014's status from `TODO` to `DONE`.
Commit (if your workflow includes commits — see `README.md`) § Git workflow).