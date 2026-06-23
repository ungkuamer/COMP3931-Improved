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

    def predict(
        self, observation: Any, deterministic: bool = False, action_masks: Any = None
    ) -> tuple[np.ndarray, None]:
        mask = np.asarray(action_masks) if action_masks is not None else None
        idx = int(np.argmax(mask)) if mask is not None and bool(mask.any()) else 0
        return np.array(idx, dtype=np.int64), None


def test_evaluate_records_per_episode_metrics(
    tiny_bike_graph: _NXGraph,
    tiny_walk_graph: _NXGraph,
    run_context: Any,
) -> None:
    """evaluate_and_visualize records one entry per episode across all lists."""
    tracker = evaluate_and_visualize(
        _FakeModel(),
        tiny_bike_graph,
        tiny_walk_graph,
        Config(),
        run_context,
        num_evaluations=3,
        budget=100_000.0,
        no_plots=True,
    )
    assert tracker.n_episodes == 3
    assert len(tracker.rewards) == 3
    assert len(tracker.connectivity) == len(tracker.efficiency) == 3
    assert len(tracker.population) == len(tracker.budget_remaining) == 3


def test_evaluate_selects_best_by_reward(
    tiny_bike_graph: _NXGraph,
    tiny_walk_graph: _NXGraph,
    run_context: Any,
) -> None:
    """best_index points at the argmax of rewards; best_reward matches."""
    tracker = evaluate_and_visualize(
        _FakeModel(),
        tiny_bike_graph,
        tiny_walk_graph,
        Config(),
        run_context,
        num_evaluations=4,
        budget=100_000.0,
        no_plots=True,
    )
    assert tracker.best_index == int(np.argmax(tracker.rewards))
    assert tracker.best_reward == max(tracker.rewards)


def test_evaluate_captures_added_edges_for_best(
    tiny_bike_graph: _NXGraph,
    tiny_walk_graph: _NXGraph,
    run_context: Any,
) -> None:
    """best_added_edges is non-empty when the policy adds at least one edge."""
    tracker = evaluate_and_visualize(
        _FakeModel(),
        tiny_bike_graph,
        tiny_walk_graph,
        Config(),
        run_context,
        num_evaluations=1,
        budget=100_000.0,
        no_plots=True,
    )
    # The tiny walk graph has affordable candidates; the fake model picks the
    # first legal one until budget exhaustion, so >=1 edge should be added.
    assert len(tracker.best_added_edges) >= 1
    for u, v, d in tracker.best_added_edges:
        assert d.get("bike_lane") == "yes"
        assert not tiny_bike_graph.has_edge(u, v)


def test_evaluate_no_plots_writes_no_figures(
    tiny_bike_graph: _NXGraph,
    tiny_walk_graph: _NXGraph,
    run_context: Any,
) -> None:
    """no_plots=True writes no image/geojson files into the output dir."""
    tracker = evaluate_and_visualize(
        _FakeModel(),
        tiny_bike_graph,
        tiny_walk_graph,
        Config(),
        run_context,
        num_evaluations=2,
        budget=100_000.0,
        no_plots=True,
        export_geojson=True,
    )
    # ensure_output_dir may create the dir, but no figures inside.
    if run_context.output_dir.exists():
        files = [p.name for p in run_context.output_dir.iterdir()]
        assert not any(f.endswith((".png", ".geojson")) for f in files), files
    _ = tracker


def test_evaluate_writes_plots_when_enabled(
    tiny_bike_graph: _NXGraph,
    tiny_walk_graph: _NXGraph,
    run_context: Any,
) -> None:
    """With plots enabled, the map + metric + reward PNGs are written."""
    evaluate_and_visualize(
        _FakeModel(),
        tiny_bike_graph,
        tiny_walk_graph,
        Config(),
        run_context,
        num_evaluations=2,
        budget=100_000.0,
        no_plots=False,
        export_geojson=False,
    )
    names = {p.name for p in run_context.output_dir.iterdir()}
    assert "best_solution_map.png" in names
    assert "evaluation_metrics.png" in names
    assert "evaluation_rewards.png" in names


def test_evaluate_export_geojson_writes_file(
    tiny_bike_graph: _NXGraph,
    tiny_walk_graph: _NXGraph,
    run_context: Any,
) -> None:
    """export_geojson=True writes suggested_bike_paths.geojson."""
    evaluate_and_visualize(
        _FakeModel(),
        tiny_bike_graph,
        tiny_walk_graph,
        Config(),
        run_context,
        num_evaluations=1,
        budget=100_000.0,
        no_plots=False,
        export_geojson=True,
    )
    assert (run_context.output_dir / "suggested_bike_paths.geojson").exists()


def test_evaluate_passes_budget_via_reset_options(
    tiny_bike_graph: _NXGraph,
    tiny_walk_graph: _NXGraph,
    run_context: Any,
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
        _FakeModel(),
        tiny_bike_graph,
        tiny_walk_graph,
        Config(),
        run_context,
        num_evaluations=2,
        budget=25_000.0,
        no_plots=True,
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
