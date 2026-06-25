# Plan 014: Fix `GreedySolver` `TypeError` on exact scoring ties

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.

> **Drift check (run first)**: `git diff --stat bc58ad2..HEAD -- bike_rl/optim/greedy.py tests/test_greedy.py`
> If either in-scope file changed since this plan was written, compare the
> "Current state" excerpts below against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: S
- **Risk**: LOW
- **Depends on**: plans/011-greedy-and-tests.md (DONE) — adds a test to the existing `tests/test_greedy.py`; touches only `GreedySolver.solve`
- **Category**: bug
- **Planned at**: commit `bc58ad2`, 2026-06-25
- **Issue**: (not published)

## Why this matters

`GreedySolver.solve` crashes with `TypeError: '>' not supported between
instances of 'Candidate' and 'Candidate'` whenever two affordable candidates
produce an **exactly equal** `(delta/cost, road_priority, -cost)` score tuple.
On real OSM-derived instances this is not a corner case — parallel streets of
the same highway class and similar length routinely tie, so greedy (the
performance floor RL must beat, OPTIMIZER_SPEC §5) and the `LocalSearchSolver`
that seeds from it can crash mid-run with no recovery. This is acknowledged as
a "known upstream caveat" in plans 012 and 013 but never fixed.

The fix is a one-line, additive tie-break that makes greedy's selection fully
deterministic without changing the documented `(road_priority, -cost)` priority
or the `Candidate` identity contract. It unblocks safe greedy / local-search use
on tie-heavy real graphs and removes a latent crash that plan 013 explicitly
worked around (its fixture "deliberately creates such ties" and the plan forbids
an ILP-vs-greedy comparison test *because* of this bug — the follow-up note
there says "A one-line greedy tie-break fix is recommended as a separate small
follow-up." This is that follow-up).

## Current state

### Files and their roles
- `bike_rl/optim/greedy.py` — defines `GreedySolver` and the shared `Solution`
  dataclass. The bug is in `GreedySolver.solve`, in the per-round scoring loop
  (the `max(scored)` call). This plan edits exactly those lines.
- `bike_rl/candidates.py` — defines `Candidate`, a `@dataclass(frozen=True)`
  with `compare` semantics: two candidates are equal iff
  `(u, v, length, road_priority)` match; `connects_to_bike_path` and `data` are
  `field(compare=False)`. **There is no `order=True`** — `Candidate` has no
  `__lt__`/`__gt__`, so `max()` falls through to comparing the bare `Candidate`
  objects and raises `TypeError`. **This plan does NOT touch `candidates.py`**
  (see Out of scope).
- `tests/test_greedy.py` — the existing greedy test module. This plan adds one
  new regression test class to it, mirroring the existing `TestGreedySolver`
  style (local `_edge(...)` helper is already defined at module top — reuse it).

### Excerpt — the buggy loop (`bike_rl/optim/greedy.py`, inside `GreedySolver.solve`)
```python
        while remaining:
            scored: list[tuple[float, int, float, Candidate]] = []
            for e in remaining:
                c = cost(e, self.cfg)
                if spent + c > budget:
                    continue
                delta = objective_delta(graph, S, e, self.weights, self.cfg)
                if delta <= 0.0:
                    continue
                scored.append((delta / c, e.road_priority, -c, e))
            if not scored:
                break
            _, _, _, best = max(scored)
            S.append(best)
            spent += cost(best, self.cfg)
            remaining.remove(best)
```

The score tuple is `(delta / c, road_priority, -c, e)`. `max(scored)` compares
tuples element-wise; if two candidates share the same first three numeric keys
(exact float equality on `delta/c` and `-c`, equal `road_priority`), Python
compares the 4th element — a `Candidate` — which is not orderable → `TypeError`.

### Excerpt — `Candidate` (no ordering) (`bike_rl/candidates.py`)
```python
@dataclass(frozen=True)
class Candidate:
    u: int | str
    v: int | str
    length: float
    road_priority: int
    connects_to_bike_path: bool = field(compare=False)
    data: dict[str, Any] = field(default_factory=dict, compare=False)
```

### Excerpt — existing test helper to reuse (`tests/test_greedy.py`, module top)
```python
def _edge(u: int | str, v: int | str, length: float, road_priority: int = 1) -> Candidate:
    """Build a Candidate satisfying the Edge protocol."""
    return Candidate(
        u=u,
        v=v,
        length=float(length),
        road_priority=road_priority,
        connects_to_bike_path=False,
        data={"length": float(length)},
    )
```
And the `default_cfg` / `default_weights` fixtures are already defined in
`tests/test_greedy.py` (see lines under `# ── Local fixtures ──`). Reuse them;
do not redefine.

### Conventions to match
- Style: `from __future__ import annotations`; ruff config enforces
  `E,F,I,UP,B,SIM,D` at line-length 100; `mypy --strict` is enforced (plan 008).
- Docstrings: Google style. The `GreedySolver` class docstring already
  documents the tie-break as "tie-break by `(road_priority, -cost)` for
  determinism" — keep that wording accurate; this plan extends the tie-break
  with a final stable key but does **not** change the documented priority
  order, so the existing docstring sentence stays valid (the new key only
  breaks ties that are *already fully tied* on the documented keys).
- `Candidate` identity is `(u, v, length, road_priority)` (plan 002). Any
  tie-break MUST be derivable from the candidate's position in the input
  order (a stable, comparable key), not from `Candidate` comparison — this
  is exactly what this plan does.

### Verified reproduction (the bug is live at `bc58ad2`)
Running the following against the current code raises `TypeError: '>' not
supported between instances of 'Candidate' and 'Candidate'` (confirmed during
planning) — it constructs two candidates with identical `(delta/cost,
road_priority, -cost)`:
```python
import networkx as nx
from bike_rl.candidates import Candidate
from bike_rl.config import Config
from bike_rl.objective import ObjectiveWeights
from bike_rl.optim.greedy import GreedySolver

def _edge(u, v, length, rp=1):
    return Candidate(u=u, v=v, length=float(length), road_priority=rp,
                     connects_to_bike_path=False,
                     data={"length": float(length), "highway": "residential"})

g = nx.MultiDiGraph()
for n, (x, y) in {1: (0, 0), 2: (0.01, 0), 10: (0.02, 0), 11: (0.03, 0)}.items():
    g.add_node(n, x=x, y=y)
g.add_edge(1, 2, length=120.0, highway="residential", bike_lane="yes")
cands = [_edge(1, 10, 100.0, 5), _edge(2, 11, 100.0, 5)]
GreedySolver(Config(), ObjectiveWeights()).solve(g, cands, 1000.0)  # raises TypeError
```
This exact construction is the basis for the regression test in Step 2.

## Commands you will need

Run all commands from the repo root with the venv active.

| Purpose | Command | Expected on success |
|---|---|---|
| Activate venv | `source .venv/bin/activate` | shell prompt changes |
| Run only the greedy tests | `pytest -q tests/test_greedy.py` | all pass (existing + 1 new class) |
| Full suite | `pytest -q` | all pass, coverage ≥80% |
| Lint | `ruff check .` | exit 0 |
| Format check | `ruff format --check .` | exit 0 (run `ruff format .` if it reports diffs) |
| Type check | `mypy --strict bike_rl` | exit 0 |

## Scope

**In scope** (the only files you should modify):
- `bike_rl/optim/greedy.py` — extend the score tuple and the unpack in
  `GreedySolver.solve` only (the `while remaining:` loop). No other change.
- `tests/test_greedy.py` — add one new test class `TestGreedyTieBreak` with the
  regression tests below.

**Out of scope** (do NOT touch, even though they look related):
- `bike_rl/candidates.py` — do **not** add `order=True` to `Candidate`. It is
  a frozen dataclass with a documented identity contract
  (`(u, v, length, road_priority)`; `connects_to_bike_path`/`data` excluded),
  used by the env, `extract_candidates`, set membership, and every optimiser.
  Adding ordering would broaden its surface area, and `u`/`v` are `int | str`
  — a `__lt__` generated by `order=True` would raise `TypeError` on mixed
  int/str node ids anyway, so it wouldn't even fix the bug universally. The
  fix belongs in `greedy.py`, not the dataclass.
- `bike_rl/optim/local_search.py` — seeds from greedy and inherits the fix;
  no change needed there. Do not add tie-handling to local search.
- `bike_rl/optim/ilp.py`, `evaluate.py` — stubs; leave alone.
- The `GreedySolver` class docstring, the `Solution` dataclass, `objective`,
  `objective_delta`, `budget.cost` — reuse unchanged.
- Any change to greedy's *priority* semantics (the order of the existing
  `(delta/cost, road_priority, -cost)` keys). Only a **final** stable key is
  appended; do not reorder or drop existing keys.

## Git workflow

- Branch: `advisor/014-greedy-tiebreak-fix` (matches the `advisor/NNN-<slug>`
  convention used by prior plans).
- Two commits, one per logical unit, conventional-commit style (see
  `git log --oneline -10` for the repo's existing style):
  1. `fix(optim): make GreedySolver tie-break deterministic (OPTIMIZER_SPEC §5)`
  2. `test(optim): add greedy exact-tie regression test`
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Add the deterministic final tie-break key in `GreedySolver.solve`

In `bike_rl/optim/greedy.py`, inside `GreedySolver.solve`, change the scoring
loop so the score tuple carries a **final, always-comparable** key before the
`Candidate`. Use the candidate's index in the current `remaining` list
(negated, so that on a full tie the *earlier* candidate in input order wins —
stable and intuitive). This is the "one-line tie-break" the maintenance notes
in plans 012/013 call for.

Replace exactly this block (the `while remaining:` body up to and including
the `max(scored)` unpack):

```python
        while remaining:
            scored: list[tuple[float, int, float, Candidate]] = []
            for e in remaining:
                c = cost(e, self.cfg)
                if spent + c > budget:
                    continue
                delta = objective_delta(graph, S, e, self.weights, self.cfg)
                if delta <= 0.0:
                    continue
                scored.append((delta / c, e.road_priority, -c, e))
            if not scored:
                break
            _, _, _, best = max(scored)
            S.append(best)
            spent += cost(best, self.cfg)
            remaining.remove(best)
```

with:

```python
        while remaining:
            # Score = (gain-per-cost, road_priority, -cost, -index, edge).
            # The first three keys are the documented OPTIMIZER_SPEC §5
            # tie-break. The final ``-index`` key is a stable, always-comparable
            # tie-breaker: ``Candidate`` is a frozen dataclass without
            # ``order=True`` (identity = (u, v, length, road_priority), plan
            # 002), so without it ``max`` would compare ``Candidate`` objects
            # and raise ``TypeError`` on exact score ties. ``-index`` makes
            # the earliest candidate in the (current) remaining order win a
            # full tie; it never reorders non-tied candidates.
            scored: list[tuple[float, int, float, int, Candidate]] = []
            for i, e in enumerate(remaining):
                c = cost(e, self.cfg)
                if spent + c > budget:
                    continue
                delta = objective_delta(graph, S, e, self.weights, self.cfg)
                if delta <= 0.0:
                    continue
                scored.append((delta / c, e.road_priority, -c, -i, e))
            if not scored:
                break
            _, _, _, _, best = max(scored)
            S.append(best)
            spent += cost(best, self.cfg)
            remaining.remove(best)
```

Notes the executor must honor:
- The tuple type annotation grows from a 4-tuple to a 5-tuple
  (`tuple[float, int, float, int, Candidate]`) — keep it exact; ruff/mypy
  strict check it.
- The unpack grows from `_, _, _, best` to `_, _, _, _, best`.
- `enumerate(remaining)` is taken over the **current** `remaining` list each
  round (it shrinks via `remove` between rounds). That is correct and
  deterministic: the input `candidates` order is fixed, `list.remove`
  preserves order, so "earliest in remaining" is a reproducible notion.
- Do NOT change anything else in the file (imports, docstring, `Solution`,
  the `return Solution(...)` block).

**Verify**:
- `ruff check bike_rl/optim/greedy.py` → exit 0
- `ruff format --check bike_rl/optim/greedy.py` → exit 0 (run `ruff format
  bike_rl/optim/greedy.py` if it reports a diff)
- `mypy --strict bike_rl` → exit 0
- `pytest -q tests/test_greedy.py` → all existing tests still pass (the
  change is additive and preserves existing selection order on non-tied
  instances; `test_deterministic_across_runs` must still be green)

### Step 2: Add the `TestGreedyTieBreak` regression test to `tests/test_greedy.py`

Append a new test class at the **end** of `tests/test_greedy.py` (after the
existing `TestGreedySolver` class). Reuse the module-level `_edge(...)` helper
and the existing `default_cfg` / `default_weights` fixtures (they are
module-scoped `pytest.fixture`s already defined in this file — do not redefine
them). Do NOT import anything new beyond what the file already imports.

Add exactly:

```python
class TestGreedyTieBreak:
    """Regression tests for the exact-score tie-break crash (plan 014).

    Before the fix, two candidates with identical (delta/cost,
    road_priority, -cost) caused ``max(scored)`` to compare ``Candidate``
    objects (frozen dataclass, no ``order=True``) and raise ``TypeError``.
    """

    def _tie_graph(self) -> nx.MultiDiGraph:
        """4-node graph whose two far candidates produce identical scores.

        Base bike-lane edge (1,2). Two candidates (1,10) and (2,11) each cost
        1000 (length 100 * edge_cost_factor 10), road_priority 5, and each
        adds one previously-unreached far node (10 / 11) — symmetric, so
        ``objective_delta`` is exactly equal for both, producing an exact
        (delta/cost, road_priority, -cost) tie that used to crash greedy.
        """
        g = nx.MultiDiGraph()
        for n, (x, y) in {
            1: (0.0, 0.0),
            2: (0.01, 0.0),
            10: (0.02, 0.0),
            11: (0.03, 0.0),
        }.items():
            g.add_node(n, x=x, y=y)
        g.add_edge(1, 2, length=120.0, highway="residential", bike_lane="yes")
        return g

    def _tie_candidates(self) -> list[Candidate]:
        """Two candidates that tie exactly on all documented greedy keys."""
        return [
            _edge(1, 10, 100.0, road_priority=5),
            _edge(2, 11, 100.0, road_priority=5),
        ]

    def test_does_not_crash_on_exact_score_tie(
        self,
        default_cfg: Config,
        default_weights: ObjectiveWeights,
    ) -> None:
        """Greedy on a full-score tie returns a Solution instead of TypeError.

        This is the core regression: before plan 014 the call raised
        ``TypeError: '>' not supported between instances of 'Candidate' and
        'Candidate'``.
        """
        graph = self._tie_graph()
        candidates = self._tie_candidates()
        # Budget 1000 affords exactly one candidate (cost 1000 each).
        sol = GreedySolver(default_cfg, default_weights).solve(graph, candidates, 1000.0)
        # It picked one edge, did not overspend, and improved over the empty
        # selection (the performance floor).
        assert len(sol.edges) == 1
        assert sol.spent == pytest.approx(1000.0)
        assert sol.objective > objective(graph, [], default_weights, default_cfg) + 1e-9

    def test_tie_break_is_deterministic_across_runs(
        self,
        default_cfg: Config,
        default_weights: ObjectiveWeights,
    ) -> None:
        """Two solves pick the identical edge and spent on a full tie.

        Determinism comes from the stable ``-index`` final tie-break key
        (earliest candidate in input order wins a full tie).
        """
        graph = self._tie_graph()
        candidates = self._tie_candidates()
        solver = GreedySolver(default_cfg, default_weights)
        sol1 = solver.solve(graph, candidates, 1000.0)
        sol2 = solver.solve(graph, candidates, 1000.0)
        assert sol1.edges == sol2.edges
        assert sol1.spent == pytest.approx(sol2.spent)
        assert sol1.objective == pytest.approx(sol2.objective)

    def test_tie_break_picks_earliest_candidate(
        self,
        default_cfg: Config,
        default_weights: ObjectiveWeights,
    ) -> None:
        """On a full tie, the earliest candidate in input order is selected.

        Pins the tie-break direction (``-index`` → earliest wins) so a later
        change cannot silently flip it. The chosen edge's identity is
        ``Candidate(1, 10, 100.0, road_priority=5)``.
        """
        graph = self._tie_graph()
        candidates = self._tie_candidates()
        sol = GreedySolver(default_cfg, default_weights).solve(graph, candidates, 1000.0)
        assert len(sol.edges) == 1
        chosen = sol.edges[0]
        assert chosen.u == 1
        assert chosen.v == 10
        assert chosen.length == pytest.approx(100.0)
        assert chosen.road_priority == 5

    def test_non_mutating_tie_does_not_affect_other_rounds(
        self,
        default_cfg: Config,
        default_weights: ObjectiveWeights,
    ) -> None:
        """Greedy still selects greedily when ties coexist with a clear winner.

        A third, strictly-better candidate (higher gain-per-cost) must be
        selected first regardless of the tie between the other two — the
        ``-index`` tie-break only affects fully-tied candidates, never
        overrides the primary ``delta / cost`` key.
        """
        g = nx.MultiDiGraph()
        for n, (x, y) in {
            1: (0.0, 0.0),
            2: (0.01, 0.0),
            10: (0.02, 0.0),
            11: (0.03, 0.0),
            20: (0.009, 0.009),
        }.items():
            g.add_node(n, x=x, y=y)
        g.add_edge(1, 2, length=120.0, highway="residential", bike_lane="yes")
        candidates = [
            _edge(1, 10, 100.0, road_priority=5),  # ties with (2,11)
            _edge(2, 11, 100.0, road_priority=5),   # ties with (1,10)
            _edge(1, 20, 50.0, road_priority=5),    # cheaper, adds nearby node 20
        ]
        sol = GreedySolver(default_cfg, default_weights).solve(g, candidates, 1000.0)
        # Node 20 is within ~140 m of bike node 1, so (1,20) has high coverage
        # gain per cost — it must be selected. Spent <= budget.
        assert sol.spent <= 1000.0 + 1e-9
        chosen_ids = {(e.u, e.v) for e in sol.edges}
        assert (1, 20) in chosen_ids
```

Notes:
- The `nx` and `pytest` imports are already present at the top of
  `tests/test_greedy.py`; `Candidate`, `Config`, `ObjectiveWeights`,
  `objective`, `GreedySolver` are already imported there too. Do not add
  duplicate imports (ruff `I` rule will fail).
- `test_tie_break_picks_earliest_candidate` pins the tie-break *direction*.
  If your Step 1 implementation instead picks the **latest** candidate on a
  full tie (e.g. you used `+i` instead of `-i`), this test will fail — fix
  Step 1 to use `-i` so earliest wins, do not weaken this test. The direction
  is a deliberate choice (stable wrt appending new candidates to the end of
  the input list).

**Verify**:
- `ruff check tests/test_greedy.py` → exit 0
- `ruff format --check tests/test_greedy.py` → exit 0 (run `ruff format
  tests/test_greedy.py` if it reports a diff)
- `mypy --strict bike_rl` → exit 0 (tests are not mypy-checked, but keep the
  annotation in `_tie_graph` / `_tie_candidates` for editor consistency)
- `pytest -q tests/test_greedy.py` → all pass; the new `TestGreedyTieBreak`
  class contributes 4 tests

### Step 3: Confirm the full gate and the reproduction is gone

Run the whole QA gate end-to-end and confirm the pre-fix reproduction no longer
raises.

```bash
pytest -q
ruff check .
ruff format --check .
mypy --strict bike_rl
```

Then run the exact reproduction from the "Verified reproduction" excerpt
above and confirm it now returns a `Solution` instead of raising:

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

**Verify**:
- `pytest -q` → all pass, whole-package coverage ≥80%
- `ruff check .` → exit 0
- `ruff format --check .` → exit 0
- `mypy --strict bike_rl` → exit 0
- The `python -c` reproduction prints `OK 1 1000.0` (one edge, spent 1000),
  no `TypeError`

## Test plan

- New tests (in `tests/test_greedy.py`, class `TestGreedyTieBreak`, 4 tests):
  1. `test_does_not_crash_on_exact_score_tie` — the core regression: the
     full-score-tie instance returns a `Solution` instead of `TypeError`.
  2. `test_tie_break_is_deterministic_across_runs` — two solves pick the
     identical edge/spent/objective on a tie.
  3. `test_tie_break_picks_earliest_candidate` — pins the tie-break *direction*
     (earliest in input order wins; `Candidate(1, 10, ...)` is selected).
  4. `test_non_mutating_tie_does_not_affect_other_rounds` — a strictly-better
     candidate is still picked first when ties coexist; the `-index` key never
     overrides the primary `delta / cost` key.
- Structural pattern: model after the existing `TestGreedySolver` class in the
  same file (scoped fixtures `default_cfg` / `default_weights`, `pytest.approx`
  for floats, `1e-9` tolerances, one `assert` per concern with a clear message
  where helpful). Do not duplicate the `_edge` helper or the fixtures.
- Verification: `pytest -q tests/test_greedy.py` → all pass (existing tests +
  4 new); `pytest -q` → full suite green, coverage ≥80%.

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `pytest -q tests/test_greedy.py` exits 0; the 4 new
      `TestGreedyTieBreak` tests pass.
- [ ] `pytest -q` exits 0; whole-package coverage ≥80% (the CI `addopts` gate).
- [ ] `ruff check .` exits 0.
- [ ] `ruff format --check .` exits 0.
- [ ] `mypy --strict bike_rl` exits 0.
- [ ] `grep -n "_, _, _, best = max(scored)" bike_rl/optim/greedy.py` returns
      no matches (the 4-tuple unpack is gone); `grep -n "_, _, _, _, best"
      bike_rl/optim/greedy.py` returns the new unpack.
- [ ] `grep -n "scored: list\[tuple\[float, int, float, Candidate\]\]" bike_rl/optim/greedy.py`
      returns no matches (the 4-tuple annotation is gone).
- [ ] No files outside `bike_rl/optim/greedy.py` and `tests/test_greedy.py` are
      modified (`git status --short` shows only those two, plus
      `plans/README.md` for the status row).
- [ ] `plans/README.md` status row for plan 014 updated (TODO → DONE).
- [ ] The reproduction `python -c "..."` from Step 3 prints `OK 1 1000.0`
      with no traceback.

## STOP conditions

Stop and report back (do not improvise) if:

- The code at the locations in "Current state" doesn't match the excerpts (the
  codebase has drifted since `bc58ad2`) — in particular if `GreedySolver.solve`
  no longer uses `max(scored)` with a 4-tuple, or if `Candidate` has gained
  `order=True` (which would mean the bug may already be fixed differently;
  re-evaluate rather than layering a second tie-break on top).
- `Candidate` is no longer a frozen dataclass with the documented
  `(u, v, length, road_priority)` identity (plan 002 contract changed) — the
  `-index` tie-break is still safe but the "why" comment in Step 1 would need
  rewording; report so the plan can be realigned.
- The two candidates in `_tie_candidates` do **not** produce exactly equal
  `objective_delta` on your environment (the regression relies on an exact
  tie; if float determinism differs across platforms such that the deltas are
  not exactly equal, the test would pass trivially without exercising the
  tie — report so the fixture can be hardened, e.g. by making the geometry
  more obviously symmetric, rather than silently shipping a non-triggering
  test).
- Step 2's `test_tie_break_picks_earliest_candidate` fails because greedy picks
  the *latest* candidate on a full tie — do NOT "fix" it by flipping the test
  expectation; fix Step 1 to use `-i` so earliest wins (the direction is
  deliberate and documented).
- You find the fix requires touching `candidates.py` or `local_search.py` —
  it must not; report instead.

## Maintenance notes

For the human/agent owning this code after it lands:

- **Plan 013 interaction**: plan 013's `max_coverage_instance` fixture
  "deliberately creates greedy `max(scored)` ties" and its maintenance notes
  forbid an ILP-vs-greedy comparison test *because* of this bug. After 014
  lands, that caveat is resolved — a *separate* small plan may now add an
  ILP-vs-greedy sanity test on a tie-free instance (use plan 013's brute-force
  fixture only if it is confirmed tie-free for greedy, which it currently is
  NOT by design). Do not rush to add it inside 014; keep 014 focused on the
  bug.
- **`LocalSearchSolver`** (`bike_rl/optim/local_search.py`) seeds from
  `GreedySolver.solve` and inherits this fix for free; no change was needed
  there. If a later plan makes local search construct its own `max(scored)`-style
  selection over candidates, it must apply the same final-stable-key tie-break
  (or import a shared helper if one is later extracted) — `Candidate` will
  remain unorderable.
- **Do not** later "clean up" the `-i` key thinking it is redundant: it is the
  only thing keeping greedy from raising `TypeError` on real tie-heavy OSM
  graphs. The comment in Step 1 documents this; keep the comment.
- **Reviewer focus**: (a) the tuple annotation and the unpack both grew by one
  element (5-tuple, 5-underscore unpack) — a mismatch would be a `ValueError`
  on unpack, caught by `mypy` and the tests; (b) the tie-break *direction*
  (`-i`, earliest wins) is pinned by `test_tie_break_picks_earliest_candidate`;
  (c) coverage of `greedy.py` should remain ≥99% (it was 100% at `bc58ad2`;
  the new branch is exercised by the new tests).
- **Future perf**: if greedy is ever moved to a `MetricsState`-backed
  incremental `objective_delta` (plans 010/012 defer this), the tie-break key
  is unaffected — it operates on the score, not the delta computation. No
  interaction.