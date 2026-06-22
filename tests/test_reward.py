"""Tests for the reward formula of ``bike_rl.env.BikePathEnv``.

Pins §3.5 structure and the §5.5 budget-efficiency fix, continuity bonus,
isolation penalty, road-priority component, and finite-reward guarantee.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx
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


def _triangle_graph_pair() -> tuple[_NXGraph, _NXGraph]:
    """Return (bike, walk) where adding edge (1,3) creates a triangle.

    The bike graph has nodes 1,2,3 with edges (1,2) and (2,3) forming
    a path. The walk graph adds a candidate edge (1,3) which, when added,
    creates a triangle and boosts connectivity from 0 to 1.
    """
    bike: _NXGraph = nx.MultiDiGraph()
    bike.add_node(1, x=0.0, y=0.0)
    bike.add_node(2, x=0.001, y=0.0)
    bike.add_node(3, x=0.0005, y=0.001)
    bike.add_edge(1, 2, length=120.0, highway="residential", bike_lane="yes")
    bike.add_edge(2, 3, length=130.0, highway="residential", bike_lane="yes")

    walk: _NXGraph = nx.MultiDiGraph()
    for n, (x, y) in {1: (0.0, 0.0), 2: (0.001, 0.0), 3: (0.0005, 0.001)}.items():
        walk.add_node(n, x=x, y=y)
    walk.add_edge(1, 2, length=120.0, highway="residential")
    walk.add_edge(2, 3, length=130.0, highway="residential")
    walk.add_edge(1, 3, length=140.0, highway="primary")

    return bike, walk


def test_budget_efficiency_term_is_nonzero(run_context: RunContext) -> None:
    """Section 5.5 regression: budget-efficiency term is non-zero when state gain > 0.

    Uses a triangle-forming edge (connectivity 0 -> 1) to ensure positive
    state gain where the term fires.
    """
    bike, walk = _triangle_graph_pair()

    cfg_off = Config(w_budget_efficiency=0.0)
    cfg_on = Config()

    env_off = _make_env(bike, walk, cfg_off, run_context)
    env_on = _make_env(bike, walk, cfg_on, run_context)

    a = int(env_off.action_masks().argmax())
    _, r_off, _, _, _ = env_off.step(a)
    _, r_on, _, _, _ = env_on.step(a)

    assert r_on != r_off, "Budget-efficiency term should be non-zero"
    assert r_on > r_off, "Budget-efficiency term should add a positive bonus"


def test_budget_efficiency_term_clamped(run_context: RunContext) -> None:
    """The budget-efficiency cap binds when raw term would be huge."""
    cap = 1.0
    bike, walk = _triangle_graph_pair()

    cfg_on = Config(w_budget_efficiency=1.0, budget_efficiency_cap=cap)
    cfg_off = Config(w_budget_efficiency=0.0)

    # Use an enormous budget so frac = cost / initial_budget is tiny,
    # making the raw term huge -- the cap should clamp it.
    env_on = _make_env(bike, walk, cfg_on, run_context, budget=1e9)
    env_off = _make_env(bike, walk, cfg_off, run_context, budget=1e9)

    a = int(env_off.action_masks().argmax())
    _, r_off, _, _, _ = env_off.step(a)
    _, r_on, _, _, _ = env_on.step(a)

    diff = r_on - r_off
    assert abs(diff - cap) < 1e-6, f"Budget term should be clamped to {cap}, got {diff}"
    assert math.isfinite(r_on), "Reward must be finite"


def test_continuity_bonus_only_when_connects(
    tiny_bike_graph: _NXGraph, tiny_walk_graph: _NXGraph, run_context: RunContext
) -> None:
    """Continuity bonus applies only when candidate touches bike network.

    Isolation penalty applies when both endpoints are new.
    """
    # Build a graph pair with an isolated candidate
    bike: _NXGraph = nx.MultiDiGraph()
    bike.add_node(1, x=0.0, y=0.0)
    bike.add_node(2, x=0.001, y=0.0)
    bike.add_edge(1, 2, length=120.0, highway="residential", bike_lane="yes")

    walk: _NXGraph = nx.MultiDiGraph()
    walk.add_node(1, x=0.0, y=0.0)
    walk.add_node(2, x=0.001, y=0.0)
    walk.add_node(50, x=0.01, y=0.01)
    walk.add_node(51, x=0.011, y=0.01)
    walk.add_edge(1, 2, length=120.0, highway="residential")
    walk.add_edge(50, 51, length=200.0, highway="residential")
    walk.add_edge(2, 50, length=300.0, highway="primary")  # connects to bike network

    cfg = Config()
    env = _make_env(bike, walk, cfg, run_context)

    # Find the isolated candidate (50,51): neither endpoint in bike network
    slots = env._slots  # type: ignore[attr-defined]
    isolated_idx = next(i for i, c in enumerate(slots) if c is not None and c.u == 50 and c.v == 51)
    connecting_idx = next(
        i for i, c in enumerate(slots) if c is not None and c.u == 2 and c.v == 50
    )

    # Step the isolated edge first
    env2 = _make_env(bike, walk, cfg, run_context)
    _, r_isolated, _, _, _ = env2.step(isolated_idx)
    assert r_isolated < 0, (
        "Isolated addition should get negative reward (isolation penalty dominates)"
    )

    # Step the connecting edge in a fresh env
    env3 = _make_env(bike, walk, cfg, run_context)
    _, r_connecting, _, _, _ = env3.step(connecting_idx)
    assert r_connecting > 0, (
        "Connecting addition should get positive reward (continuity bonus applies)"
    )


def test_isolation_penalty_applied_once(run_context: RunContext) -> None:
    """The isolation penalty of -100 is applied exactly once for an isolated addition."""
    bike: _NXGraph = nx.MultiDiGraph()
    bike.add_node(1, x=0.0, y=0.0)
    bike.add_node(2, x=0.001, y=0.0)
    bike.add_edge(1, 2, length=120.0, highway="residential", bike_lane="yes")

    walk: _NXGraph = nx.MultiDiGraph()
    walk.add_node(1, x=0.0, y=0.0)
    walk.add_node(2, x=0.001, y=0.0)
    walk.add_node(50, x=0.01, y=0.01)
    walk.add_node(51, x=0.011, y=0.01)
    walk.add_edge(1, 2, length=120.0, highway="residential")
    walk.add_edge(50, 51, length=200.0, highway="residential")

    cfg_default = Config()
    cfg_no_penalty = Config(isolation_penalty=0.0)

    env_default = _make_env(bike, walk, cfg_default, run_context)
    env_no_penalty = _make_env(bike, walk, cfg_no_penalty, run_context)

    # Find the isolated candidate
    slots = env_default._slots  # type: ignore[attr-defined]
    isolated_idx = next(i for i, c in enumerate(slots) if c is not None and c.u == 50 and c.v == 51)

    _, r_default, _, _, _ = env_default.step(isolated_idx)
    _, r_no_penalty, _, _, _ = env_no_penalty.step(isolated_idx)

    diff = r_default - r_no_penalty
    assert abs(diff - cfg_default.isolation_penalty) < 1e-6, (
        f"Isolation penalty should be {cfg_default.isolation_penalty}, got {diff}"
    )


def test_reward_is_finite_float(
    tiny_bike_graph: _NXGraph, tiny_walk_graph: _NXGraph, run_context: RunContext
) -> None:
    """All rewards returned across a full episode are finite Python floats."""
    env = _make_env(tiny_bike_graph, tiny_walk_graph, Config(), run_context)

    done = False
    while not done:
        mask = env.action_masks()
        if not mask.any():
            break
        a = int(mask.argmax())
        _, r, term, trunc, _ = env.step(a)
        assert isinstance(r, float), f"Reward should be float, got {type(r)}"
        assert math.isfinite(r), f"Reward should be finite, got {r}"
        done = term or trunc


def test_reward_road_priority_component(
    tiny_bike_graph: _NXGraph, tiny_walk_graph: _NXGraph, run_context: RunContext
) -> None:
    """The road-priority term is road_priority * road_priority_scale."""
    cfg_default = Config()
    cfg_noscale = Config(road_priority_scale=0.0)

    env_default = _make_env(tiny_bike_graph, tiny_walk_graph, cfg_default, run_context)
    env_noscale = _make_env(tiny_bike_graph, tiny_walk_graph, cfg_noscale, run_context)

    # First action in the mask is the primary (priority 5) candidate
    a = int(env_default.action_masks().argmax())

    _, r_default, _, _, _ = env_default.step(a)
    _, r_noscale, _, _, _ = env_noscale.step(a)

    expected_diff = 5 * cfg_default.road_priority_scale  # 5 * 5.0 = 25.0
    diff = r_default - r_noscale
    assert abs(diff - expected_diff) < 1e-6, (
        f"Road priority diff should be {expected_diff}, got {diff}"
    )
