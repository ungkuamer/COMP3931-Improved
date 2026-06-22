"""Gymnasium RL environment for budgeted bike-network expansion.

See RECREATE_SPEC.md §3.4, §5.4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, SupportsFloat

import gymnasium as gym
import numpy as np

if TYPE_CHECKING:
    from bike_rl.config import Config
    from bike_rl.run_context import RunContext


class BikePathEnv(gym.Env[object, object]):
    """Bike-path expansion environment with MaskablePPO action masking."""

    def __init__(
        self,
        bike_graph: object,
        walk_graph: object,
        cfg: Config,
        run_context: RunContext,
    ) -> None:
        super().__init__()
        raise NotImplementedError("Plan 004")

    def step(self, action: object) -> tuple[object, SupportsFloat, bool, bool, dict[str, Any]]:
        raise NotImplementedError("Plan 004")

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[object, dict[str, Any]]:
        raise NotImplementedError("Plan 004")

    def action_masks(self) -> np.ndarray:
        raise NotImplementedError("Plan 004")
