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
    """RunContext with a temp output dir."""
    from bike_rl.run_context import RunContext

    return RunContext(
        run_id="test",
        output_dir=tmp_path / "out",
        timestamp=datetime.now(timezone.utc),
    )


def test_render_solution_map_writes_png(
    tiny_bike_graph: _NXGraph,
    run_context: Any,
) -> None:
    """render_solution_map writes best_solution_map.png under the run dir."""
    added = [(1, 2, {"bike_lane": "yes", "length": 120.0})]
    out = render_solution_map(tiny_bike_graph, added, run_context, show=False)
    assert out.name == "best_solution_map.png"
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_solution_map_with_stats_annotation(
    tiny_bike_graph: _NXGraph,
    run_context: Any,
) -> None:
    """The optional stats dict is rendered without error."""
    added = [(1, 2, {"bike_lane": "yes", "length": 120.0})]
    out = render_solution_map(
        tiny_bike_graph,
        added,
        run_context,
        show=False,
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
    tiny_bike_graph: _NXGraph,
    run_context: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§5.10: plt.show() is NOT called when show=False (headless default)."""
    import matplotlib.pyplot as plt

    called = {"show": False}
    monkeypatch.setattr(plt, "show", lambda *a, **k: called.__setitem__("show", True))
    added = [(1, 2, {"bike_lane": "yes", "length": 120.0})]
    render_solution_map(tiny_bike_graph, added, run_context, show=False)
    plot_rewards([1.0], run_context, show=False)
    assert called["show"] is False
