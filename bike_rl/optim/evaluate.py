"""Evaluation harness for solvers and RL policies. See OPTIMIZER_SPEC.md §8."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bike_rl.optim.greedy import Solution


def evaluate_solver(solver: object, instance: object) -> Solution:
    """Run a solver on an instance and return a Solution."""
    raise NotImplementedError("Optimiser plan")


def evaluate_rl_policy(policy: object, instance: object, budget: float) -> Solution:
    """Run a trained RL policy on an instance and return a Solution."""
    raise NotImplementedError("Optimiser plan")
