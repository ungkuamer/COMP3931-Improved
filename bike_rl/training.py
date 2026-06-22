"""PPO/MaskablePPO training loop. See RECREATE_SPEC.md §3.7."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import gymnasium as gym

    from bike_rl.config import Config
    from bike_rl.run_context import RunContext


def make_env(
    bike_graph: object,
    walk_graph: object,
    cfg: Config,
    run_context: RunContext,
    rank: int,
    seed: int,
) -> gym.Env[object, object]:
    """Create a single BikePathEnv wrapped with Monitor."""
    raise NotImplementedError("Plan 005")


def train_model(envs: object, cfg: Config, run_context: RunContext, total_timesteps: int) -> object:
    """Train a PPO/MaskablePPO model."""
    raise NotImplementedError("Plan 005")


class TrainingProgressCallback:
    """Callback for logging training progress."""

    def __init__(self) -> None:
        raise NotImplementedError("Plan 005")

    def _on_step(self) -> bool:
        raise NotImplementedError("Plan 005")
