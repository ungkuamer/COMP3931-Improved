"""Tests for ``bike_rl.env.BikePathEnv``.

Covers: action masking (§5.4), budget exhaustion termination, invalid-action
no-op, reset restoration, observation shape/dtype/range, episode info on
termination (§5.12), step-cap truncation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
import pytest

from bike_rl.config import Config
from bike_rl.env import BikePathEnv
from bike_rl.run_context import RunContext

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


def _make_env(
    bike_graph: _NXGraph,
    walk_graph: _NXGraph,
    cfg: Config,
    run_context: RunContext,
    budget: float = 100_000.0,
) -> BikePathEnv:
    """Build and reset a BikePathEnv with the given budget."""
    env = BikePathEnv(bike_graph, walk_graph, cfg, run_context)
    env.reset(options={"budget": budget})
    return env


def test_observation_space_and_shape(
    tiny_bike_graph: _NXGraph, tiny_walk_graph: _NXGraph, run_context: RunContext
) -> None:
    """Observation space is Box(4,) float32; action space n = candidate count."""
    env = _make_env(tiny_bike_graph, tiny_walk_graph, Config(), run_context)

    assert env.observation_space.shape == (4,)
    assert env.observation_space.dtype == np.float32
    assert env.action_space.n == 4  # four candidates (incl. self-loop)

    obs, _ = env.reset(options={"budget": 100_000.0})
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4,)
    assert obs.dtype == np.float32
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)


def test_action_masks_correct_initially(
    tiny_bike_graph: _NXGraph, tiny_walk_graph: _NXGraph, run_context: RunContext
) -> None:
    """All slots are True initially with a large budget."""
    env = _make_env(tiny_bike_graph, tiny_walk_graph, Config(), run_context)
    mask = env.action_masks()
    assert mask.dtype == bool
    assert len(mask) == 4
    assert mask.all()


def test_action_masks_masks_unaffordable(
    tiny_bike_graph: _NXGraph, tiny_walk_graph: _NXGraph, run_context: RunContext
) -> None:
    """With budget 1000, all candidates cost > 1000 -> all masked -> immediate termination."""
    env = _make_env(tiny_bike_graph, tiny_walk_graph, Config(), run_context, budget=1000.0)
    mask = env.action_masks()
    assert not mask.any()

    obs, r, term, trunc, info = env.step(0)
    assert term is True
    assert r == 0.0
    assert "episode" in info
    assert info["episode"]["l"] == 0  # no steps taken


def test_invalid_action_is_noop_and_not_done(
    tiny_bike_graph: _NXGraph, tiny_walk_graph: _NXGraph, run_context: RunContext
) -> None:
    """Stepping with a previously consumed slot returns 0 reward and doesn't terminate."""
    env = _make_env(tiny_bike_graph, tiny_walk_graph, Config(), run_context)
    mask = env.action_masks()
    a = int(mask.argmax())
    # Consume the slot
    obs1, r1, term1, trunc1, _ = env.step(a)
    # Now step with the same (now consumed) index
    obs2, r2, term2, trunc2, info2 = env.step(a)
    assert r2 == 0.0
    assert term2 is False
    assert info2.get("invalid_action") is True
    # obs should be unchanged (same as previous state)
    assert np.array_equal(obs1, obs2)


def test_step_uses_slot_and_budget_decreases(
    tiny_bike_graph: _NXGraph, tiny_walk_graph: _NXGraph, run_context: RunContext
) -> None:
    """After a valid step, budget is deducted and slot is consumed."""
    cfg = Config()
    env = _make_env(tiny_bike_graph, tiny_walk_graph, cfg, run_context)
    mask = env.action_masks()
    a = int(mask.argmax())
    # Candidate (2,5): length 200, cost = 200 * 10 = 2000
    expected_cost = 2000.0
    before_budget = env._budget  # type: ignore[attr-defined]
    obs, r, term, trunc, info = env.step(a)
    assert env._budget == before_budget - expected_cost  # type: ignore[attr-defined]
    assert env._slots[a] is None  # type: ignore[attr-defined]
    assert np.isfinite(r)


def test_budget_exhaustion_terminates(
    tiny_bike_graph: _NXGraph, tiny_walk_graph: _NXGraph, run_context: RunContext
) -> None:
    """With budget just enough for cheapest candidate, step terminates immediately after."""
    # Cheapest candidate is (3,5): length 150, cost = 150 * 10 = 1500
    env = _make_env(tiny_bike_graph, tiny_walk_graph, Config(), run_context, budget=1500.0)
    mask = env.action_masks()
    # Only the cheapest (3,5) should be affordable
    assert mask.sum() == 1
    a = int(mask.argmax())
    obs, r, term, trunc, info = env.step(a)
    assert term is True
    assert info["episode"]["l"] == 1


def test_reset_restores_state(
    tiny_bike_graph: _NXGraph, tiny_walk_graph: _NXGraph, run_context: RunContext
) -> None:
    """After stepping and resetting, env returns to initial state."""
    env = _make_env(tiny_bike_graph, tiny_walk_graph, Config(), run_context)
    # Get initial snapshot
    initial_obs = env._last_obs.copy()  # type: ignore[attr-defined]

    # Step twice
    mask = env.action_masks()
    env.step(int(mask.argmax()))
    mask = env.action_masks()
    env.step(int(mask.argmax()))

    # Reset
    env.reset(options={"budget": 100_000.0})
    assert env._budget == 100_000.0  # type: ignore[attr-defined]
    assert env._steps == 0  # type: ignore[attr-defined]
    assert env._episode_reward == 0.0  # type: ignore[attr-defined]
    # All slots should be non-None again
    assert all(s is not None for s in env._slots)  # type: ignore[attr-defined]
    assert np.array_equal(env._last_obs, initial_obs)  # type: ignore[attr-defined]


def test_reset_deterministic_with_same_seed(
    tiny_bike_graph: _NXGraph, tiny_walk_graph: _NXGraph, run_context: RunContext
) -> None:
    """Two resets with the same seed produce identical observations."""
    env = BikePathEnv(tiny_bike_graph, tiny_walk_graph, Config(), run_context)
    obs1, _ = env.reset(seed=42, options={"budget": 100_000.0})
    obs2, _ = env.reset(seed=42, options={"budget": 100_000.0})
    assert np.array_equal(obs1, obs2)


def test_episode_info_emitted_on_termination(
    tiny_bike_graph: _NXGraph, tiny_walk_graph: _NXGraph, run_context: RunContext
) -> None:
    """When terminated, info contains episode reward and length."""
    env = _make_env(tiny_bike_graph, tiny_walk_graph, Config(), run_context, budget=1500.0)
    mask = env.action_masks()
    a = int(mask.argmax())
    obs, r, term, trunc, info = env.step(a)
    assert term is True
    assert "episode" in info
    assert isinstance(info["episode"]["r"], float)
    assert info["episode"]["l"] == 1
    # Episode reward should equal the reward from this step
    assert abs(info["episode"]["r"] - r) < 1e-6


def test_step_cap_truncates(
    tiny_bike_graph: _NXGraph, tiny_walk_graph: _NXGraph, run_context: RunContext
) -> None:
    """max_episode_steps=1 truncates after one step."""
    cfg = Config(max_episode_steps=1)
    env = _make_env(tiny_bike_graph, tiny_walk_graph, cfg, run_context)
    mask = env.action_masks()
    obs, r, term, trunc, info = env.step(int(mask.argmax()))
    assert trunc is True
    assert term is False  # budget remains, it's truncation not termination


def test_action_masks_length_matches_action_space(
    tiny_bike_graph: _NXGraph, tiny_walk_graph: _NXGraph, run_context: RunContext
) -> None:
    """len(action_masks()) == action_space.n always."""
    env = _make_env(tiny_bike_graph, tiny_walk_graph, Config(), run_context)
    assert len(env.action_masks()) == env.action_space.n

    # Also holds between resets
    env.reset(options={"budget": 100_000.0})
    assert len(env.action_masks()) == env.action_space.n


def test_no_bike_lane_candidates_terminates_immediately(run_context: RunContext) -> None:
    """When no candidates exist, action space is padded to 1 and step terminates."""
    bike: _NXGraph = nx.MultiDiGraph()
    bike.add_node(1, x=0.0, y=0.0)
    bike.add_node(2, x=0.001, y=0.0)
    bike.add_edge(1, 2, length=120.0, highway="residential", bike_lane="yes")

    walk: _NXGraph = nx.MultiDiGraph()
    walk.add_node(1, x=0.0, y=0.0)
    walk.add_node(2, x=0.001, y=0.0)
    walk.add_edge(1, 2, length=200.0, highway="primary")  # already in bike graph -> skipped

    env = BikePathEnv(bike, walk, Config(), run_context)
    assert env.action_space.n == 1  # padded to 1
    env.reset(options={"budget": 100_000.0})
    mask = env.action_masks()
    assert not mask.any()

    obs, r, term, trunc, info = env.step(0)
    assert term is True
    assert r == 0.0
    assert info["episode"]["l"] == 0


def test_env_does_not_mutate_input_graphs(
    tiny_bike_graph: _NXGraph, tiny_walk_graph: _NXGraph, run_context: RunContext
) -> None:
    """Input graphs are unchanged after env construction and steps."""
    bike_edges_before = tiny_bike_graph.number_of_edges()
    walk_edges_before = tiny_walk_graph.number_of_edges()
    bike_bike_lane_tags = {d.get("bike_lane") for _, _, d in tiny_bike_graph.edges(data=True)}

    env = _make_env(tiny_bike_graph, tiny_walk_graph, Config(), run_context)
    # Step a couple of times
    mask = env.action_masks()
    env.step(int(mask.argmax()))
    mask = env.action_masks()
    env.step(int(mask.argmax()))

    assert tiny_bike_graph.number_of_edges() == bike_edges_before
    assert tiny_walk_graph.number_of_edges() == walk_edges_before
    bike_bike_lane_tags_after = {d.get("bike_lane") for _, _, d in tiny_bike_graph.edges(data=True)}
    assert bike_bike_lane_tags == bike_bike_lane_tags_after
