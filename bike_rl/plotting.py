"""Headless-safe plotting for bike-path solutions. See RECREATE_SPEC.md §5.10."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bike_rl.evaluation import EvaluationTracker
    from bike_rl.run_context import RunContext


def render_solution_map(
    graph: object, added_paths: list[object], run_context: RunContext, show: bool = False
) -> Path:
    """Render a GeoJSON-style map of added paths."""
    raise NotImplementedError("Plan 006")


def plot_metrics(tracker: EvaluationTracker, run_context: RunContext, show: bool = False) -> None:
    """Plot evaluation metrics over episodes."""
    raise NotImplementedError("Plan 006")


def plot_rewards(rewards: list[float], run_context: RunContext, show: bool = False) -> None:
    """Plot reward history."""
    raise NotImplementedError("Plan 006")
