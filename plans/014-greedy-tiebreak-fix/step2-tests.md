# Step 2: Add the `TestGreedyTieBreak` regression test class

File: `tests/test_greedy.py`. Append the class below at the **end** of the
file, after the existing `TestGreedySolver` class.

### Reuse — do NOT redefine these (already in `tests/test_greedy.py`)

- The module-level helper `_edge(u, v, length, road_priority=1) -> Candidate`.
- The `default_cfg` and `default_weights` fixtures (defined under the
  `# ── Local fixtures ──` section).
- All needed imports (`nx`, `pytest`, `Candidate`, `Config`,
  `ObjectiveWeights`, `objective`, `GreedySolver`) are already at module top.
  Do NOT add duplicate imports — ruff `I` rule will fail.

### Append exactly this class

```python
class TestGreedyTieBreak:
    """Regression tests for the exact-score tie-break crash (plan 014).

    Before the fix, two candidates with identical (delta/cost,
    road_priority, -cost) caused ``max(scored)`` to compare ``Candidate``
    objects (frozen dataclass, no ``order=True``) and raise ``TypeError``.
    """

    def _tie_graph(self) -> nx.MultiDiGraph:
        """4-node graph whose two far candidates produce identical scores.

        Base bike-lane edge (1,2). Candidates (1,10) and (2,11) each cost
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

        Before plan 014 the call raised
        ``TypeError: '>' not supported between instances of 'Candidate' and
        'Candidate'``.
        """
        graph = self._tie_graph()
        candidates = self._tie_candidates()
        # Budget 1000 affords exactly one candidate (cost 1000 each).
        sol = GreedySolver(default_cfg, default_weights).solve(graph, candidates, 1000.0)
        assert len(sol.edges) == 1
        assert sol.spent == pytest.approx(1000.0)
        assert sol.objective > objective(graph, [], default_weights, default_cfg) + 1e-9

    def test_tie_break_is_deterministic_across_runs(
        self,
        default_cfg: Config,
        default_weights: ObjectiveWeights,
    ) -> None:
        """Two solves pick the identical edge and spent on a full tie."""
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
        """A strictly-better candidate is still picked first when ties coexist.

        The ``-index`` tie-break only affects fully-tied candidates; it never
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
        assert sol.spent <= 1000.0 + 1e-9
        chosen_ids = {(e.u, e.v) for e in sol.edges}
        assert (1, 20) in chosen_ids
```

### Note on the tie-break direction

`test_tie_break_picks_earliest_candidate` pins the direction (earliest wins,
i.e. step 1 uses `-i`). If it fails because greedy picks the *latest*
candidate, fix step 1 to use `-i` — do **not** weaken the test.

### Verify

- `ruff check tests/test_greedy.py` → exit 0
- `ruff format --check tests/test_greedy.py` → exit 0 (run `ruff format tests/test_greedy.py` if it reports a diff)
- `pytest -q tests/test_greedy.py` → all pass; the new `TestGreedyTieBreak` class contributes 4 tests

Proceed to `step3-verify.md` only after all three pass.