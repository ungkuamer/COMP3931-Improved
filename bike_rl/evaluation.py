"""Evaluation and visualization of trained policies. See RECREATE_SPEC.md §3.8."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bike_rl.config import Config
    from bike_rl.run_context import RunContext


@dataclass
class EvaluationTracker:
    """Tracks evaluation metrics across episodes."""

    rewards: list[float] = field(default_factory=list)
    paths: list[list[object]] = field(default_factory=list)
    connectivity: list[float] = field(default_factory=list)
    efficiency: list[float] = field(default_factory=list)
    population: list[float] = field(default_factory=list)
    budget: list[float] = field(default_factory=list)
    best_reward: float = float("-inf")


def evaluate_and_visualize(
    model: object,
    bike_graph: object,
    walk_graph: object,
    cfg: Config,
    run_context: RunContext,
    num_evaluations: int = 10,
) -> EvaluationTracker:
    """Evaluate a trained policy and produce visualizations."""
    raise NotImplementedError("Plan 006")
