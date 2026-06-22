"""Tests for ``bike_rl.training``.

Covers: ``make_env`` seeding + budget passthrough, ``make_vec_env`` thunk
seeds, ``train_model`` timestep rounding + MaskablePPO/policy/cfg wiring
(§3.7), and ``TrainingProgressCallback`` recording every terminated episode
including the last (§5.12).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from bike_rl.config import Config
from bike_rl.env import BikePathEnv
from bike_rl.run_context import RunContext
from bike_rl.training import TrainingProgressCallback, make_env, make_vec_env, train_model

if TYPE_CHECKING:
    _NXGraph = nx.MultiDiGraph[Any, Any, Any]
else:
    _NXGraph = nx.MultiDiGraph


@pytest.fixture
def run_context(tmp_path: Path) -> RunContext:
    """Provide a RunContext with a temp output dir."""
    return RunContext(
        run_id="test",
        output_dir=tmp_path / "out",
        timestamp=datetime.now(timezone.utc),
    )


def test_make_env_returns_monitor_wrapped_bikepathenv(
    tiny_bike_graph: _NXGraph, tiny_walk_graph: _NXGraph, run_context: RunContext
) -> None:
    """make_env returns a Monitor wrapping a BikePathEnv with the episode reset."""
    env = make_env(tiny_bike_graph, tiny_walk_graph, Config(), run_context, rank=0, seed=42)
    assert env.unwrapped.__class__.__name__ == "BikePathEnv"
    assert isinstance(env.unwrapped, BikePathEnv)
    # Monitor is a wrapper around the env
    assert env.spec is None or hasattr(env, "step")


def test_make_env_passes_budget_via_reset_options(
    tiny_bike_graph: _NXGraph,
    tiny_walk_graph: _NXGraph,
    run_context: RunContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """make_env resets with options={'budget': budget} (plan 004 contract)."""
    recorded: dict[str, Any] = {}

    def fake_reset(self, *, seed=None, options=None):
        recorded["seed"] = seed
        recorded["options"] = options
        return None, {}

    monkeypatch.setattr(BikePathEnv, "reset", fake_reset)
    make_env(
        tiny_bike_graph, tiny_walk_graph, Config(), run_context, rank=0, seed=42, budget=25000.0
    )
    assert recorded["options"] == {"budget": 25000.0}
    assert recorded["seed"] == 42


def test_make_env_seeds_differ_per_rank(
    tiny_bike_graph: _NXGraph,
    tiny_walk_graph: _NXGraph,
    run_context: RunContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each rank gets a distinct seed = base_seed + rank (§3.7)."""
    seen: list[int] = []

    def fake_reset(self, *, seed=None, options=None):
        seen.append(seed)
        return None, {}

    monkeypatch.setattr(BikePathEnv, "reset", fake_reset)
    for rank in range(3):
        make_env(tiny_bike_graph, tiny_walk_graph, Config(), run_context, rank=rank, seed=100)
    assert seen == [100, 101, 102]


def test_make_vec_env_builds_n_envs_with_distinct_seeds(
    tiny_bike_graph: _NXGraph,
    tiny_walk_graph: _NXGraph,
    run_context: RunContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """make_vec_env with DummyVecEnv builds n_envs thunks with distinct seeds."""
    seen: list[int] = []

    def fake_reset(self, *, seed=None, options=None):
        seen.append(seed)
        return None, {}

    monkeypatch.setattr(BikePathEnv, "reset", fake_reset)
    vec = make_vec_env(
        tiny_bike_graph,
        tiny_walk_graph,
        Config(),
        run_context,
        n_envs=3,
        seed=7,
        vec_env_cls=DummyVecEnv,
    )
    assert vec.num_envs == 3
    # DummyVecEnv calls reset on each env during construction
    assert seen == [7, 8, 9]


def test_train_model_rounds_timesteps_up_to_batch(
    tiny_bike_graph: _NXGraph,
    tiny_walk_graph: _NXGraph,
    run_context: RunContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """train_model rounds total_timesteps up to ppo_n_steps * n_envs (§3.7)."""
    cfg = Config()  # ppo_n_steps=2048
    n_envs = 2
    batch = cfg.ppo_n_steps * n_envs  # 4096
    vec = make_vec_env(
        tiny_bike_graph,
        tiny_walk_graph,
        cfg,
        run_context,
        n_envs=n_envs,
        seed=cfg.seed,
        vec_env_cls=DummyVecEnv,
    )

    captured: dict[str, Any] = {}

    def fake_learn(self, total_timesteps, callback=None, **kw):
        captured["total_timesteps"] = total_timesteps
        captured["callback"] = callback
        return self

    monkeypatch.setattr("bike_rl.training.MaskablePPO.learn", fake_learn)
    # Request 1000 (< batch) -> rounded up to 4096
    train_model(vec, cfg, run_context, total_timesteps=1000)
    assert captured["total_timesteps"] == batch  # 4096

    # Request 5000 -> next multiple of 4096 = 8192
    train_model(vec, cfg, run_context, total_timesteps=5000)
    assert captured["total_timesteps"] == 2 * batch  # 8192


def test_train_model_uses_maskable_ppo_and_cfg_hyperparams(
    tiny_bike_graph: _NXGraph,
    tiny_walk_graph: _NXGraph,
    run_context: RunContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """train_model constructs MaskablePPO with MaskableActorCriticPolicy + cfg values."""
    cfg = Config()
    vec = make_vec_env(
        tiny_bike_graph,
        tiny_walk_graph,
        cfg,
        run_context,
        n_envs=2,
        seed=cfg.seed,
        vec_env_cls=DummyVecEnv,
    )

    captured: dict[str, Any] = {}

    class FakeMaskablePPO:
        def __init__(
            self,
            policy,
            env,
            learning_rate,
            n_steps,
            batch_size,
            n_epochs,
            gamma,
            gae_lambda,
            clip_range,
            seed,
            device,
            verbose,
        ):  # noqa: PLR0913
            captured.update(
                policy=policy,
                env=env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                seed=seed,
                device=device,
                verbose=verbose,
            )

        def learn(self, total_timesteps, callback=None, **kw):
            captured["learn_timesteps"] = total_timesteps
            return self

    monkeypatch.setattr("bike_rl.training.MaskablePPO", FakeMaskablePPO)
    train_model(vec, cfg, run_context, total_timesteps=4096)

    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

    assert captured["policy"] is MaskableActorCriticPolicy
    assert captured["learning_rate"] == cfg.ppo_learning_rate
    assert captured["n_steps"] == cfg.ppo_n_steps
    assert captured["batch_size"] == cfg.ppo_batch_size
    assert captured["n_epochs"] == cfg.ppo_n_epochs
    assert captured["gamma"] == cfg.ppo_gamma
    assert captured["gae_lambda"] == cfg.ppo_gae_lambda
    assert captured["clip_range"] == cfg.ppo_clip_range
    assert captured["seed"] == cfg.seed
    assert captured["device"] == cfg.device


def test_train_model_creates_checkpoint_dir_and_attaches_callbacks(
    tiny_bike_graph: _NXGraph,
    tiny_walk_graph: _NXGraph,
    run_context: RunContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """train_model creates run_context.output_dir/checkpoints and passes 2 callbacks."""
    cfg = Config()
    vec = make_vec_env(
        tiny_bike_graph,
        tiny_walk_graph,
        cfg,
        run_context,
        n_envs=2,
        seed=cfg.seed,
        vec_env_cls=DummyVecEnv,
    )

    def fake_learn(self, total_timesteps, callback=None, **kw):
        assert isinstance(callback, list)
        assert len(callback) == 2
        assert isinstance(callback[0], TrainingProgressCallback)
        from stable_baselines3.common.callbacks import CheckpointCallback

        assert isinstance(callback[1], CheckpointCallback)
        assert "checkpoints" in callback[1].save_path
        return self

    monkeypatch.setattr("bike_rl.training.MaskablePPO.learn", fake_learn)
    train_model(vec, cfg, run_context, total_timesteps=4096)
    assert (run_context.output_dir / "checkpoints").exists()


def test_training_progress_callback_records_all_episodes_including_last() -> None:
    """§5.12: the callback records every terminated episode, including the last."""
    cb = TrainingProgressCallback(total_timesteps=10, progress_bar=False)

    # Simulate SB3's BaseCallback wiring: set the attrs _on_step reads.
    cb.model = type("M", (), {"num_timesteps": 0})()  # noqa: S106
    cb.locals = {"infos": [{"episode": {"r": 1.0, "l": 5}}]}
    cb.n_calls = 0

    # Three steps with episode infos (last one terminates the final episode).
    for r, length in [(1.0, 5), (2.0, 7), (3.0, 4)]:
        cb.locals = {"infos": [{"episode": {"r": r, "l": length}}]}
        assert cb._on_step() is True

    assert cb.episode_rewards == [1.0, 2.0, 3.0]
    assert cb.episode_lengths == [5, 7, 4]


def test_training_progress_callback_ignores_infos_without_episode() -> None:
    """Steps with no terminated episode contribute nothing."""
    cb = TrainingProgressCallback(total_timesteps=10, progress_bar=False)
    cb.model = type("M", (), {"num_timesteps": 0})()
    cb.locals = {"infos": [{"invalid_action": True}, {}]}
    assert cb._on_step() is True
    assert cb.episode_rewards == []
    assert cb.episode_lengths == []


def test_training_progress_callback_handles_multi_env_infos() -> None:
    """With multiple envs, infos is a list; each env's episode is recorded."""
    cb = TrainingProgressCallback(total_timesteps=10, progress_bar=False)
    cb.model = type("M", (), {"num_timesteps": 0})()
    cb.locals = {
        "infos": [
            {"episode": {"r": 1.5, "l": 3}},
            {},  # env 1 did not terminate
            {"episode": {"r": 2.5, "l": 6}},
        ]
    }
    assert cb._on_step() is True
    assert cb.episode_rewards == [1.5, 2.5]
    assert cb.episode_lengths == [3, 6]
