"""Deterministic policy evaluation + best-solution tracking. See RECREATE_SPEC.md §3.8."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import networkx as nx

from bike_rl.env import BikePathEnv

if TYPE_CHECKING:
    from sb3_contrib import MaskablePPO

    from bike_rl.config import Config
    from bike_rl.run_context import RunContext

    _NXGraph = nx.MultiDiGraph[Any, Any, Any]
else:
    _NXGraph = nx.MultiDiGraph

logger = logging.getLogger(__name__)


@dataclass
class EvaluationTracker:
    """Per-episode evaluation metrics + the best solution's added edges (§3.8).

    Attributes:
        rewards: Total reward per episode (sum of step rewards).
        connectivity: Final connectivity per episode (from the final obs).
        efficiency: Final path efficiency per episode.
        population: Final coverage (population-served proxy) per episode.
        budget_remaining: Remaining budget at episode end, per episode.
        n_episodes: Number of evaluation episodes run.
        best_reward: The highest ``rewards[i]``; ``-inf`` if none ran.
        best_index: Index into the per-episode lists of the best episode,
            or ``-1`` if none ran.
        best_added_edges: Edges added in the best episode, as
            ``(u, v, data_dict)`` triples, for rendering the solution map.
    """

    rewards: list[float] = field(default_factory=list)
    connectivity: list[float] = field(default_factory=list)
    efficiency: list[float] = field(default_factory=list)
    population: list[float] = field(default_factory=list)
    budget_remaining: list[float] = field(default_factory=list)
    n_episodes: int = 0
    best_reward: float = float("-inf")
    best_index: int = -1
    best_added_edges: list[tuple[Any, Any, dict[str, Any]]] = field(default_factory=list)


def _run_episode(
    model: MaskablePPO,
    bike_graph: _NXGraph,
    walk_graph: _NXGraph,
    cfg: Config,
    run_context: RunContext,
    budget: float,
    seed: int,
) -> tuple[float, float, float, float, float, list[tuple[Any, Any, dict[str, Any]]]]:
    """Run one deterministic rollout and return metrics + added edges.

    Returns ``(total_reward, connectivity, efficiency, coverage,
    budget_remaining, added_edges)``. The four metric values are read from
    the final observation (plan 004's observation contract); ``added_edges``
    is the diff of the env's live graph against ``bike_graph`` (see "Added-edge
    collection" in the plan).
    """
    env = BikePathEnv(bike_graph, walk_graph, cfg, run_context)
    obs, _ = env.reset(seed=seed, options={"budget": float(budget)})
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        mask = env.action_masks()
        if not bool(mask.any()):
            break  # no legal action — episode is over
        action, _ = model.predict(obs, deterministic=True, action_masks=mask)
        obs, reward, terminated, truncated, _info = env.step(int(action))
        total_reward += float(reward)
        steps += 1
        if steps > 10_000:  # hard safety cap; env terminates by budget normally
            logger.warning("eval episode hit 10_000-step safety cap at seed=%d", seed)
            break

    conn = float(obs[0])
    eff = float(obs[1])
    pop = float(obs[2])
    norm_budget = float(obs[3])
    budget_remaining = norm_budget * float(budget)

    raw = env._graph  # noqa: SLF001 — read-only; see plan "Added-edge collection"
    added: list[tuple[Any, Any, dict[str, Any]]] = [
        (u, v, dict(d))
        for u, v, d in raw.edges(data=True)
        if d.get("bike_lane") == "yes" and not bike_graph.has_edge(u, v)
    ]
    return total_reward, conn, eff, pop, budget_remaining, added


def evaluate_and_visualize(
    model: object,
    bike_graph: _NXGraph,
    walk_graph: _NXGraph,
    cfg: Config,
    run_context: RunContext,
    num_evaluations: int = 10,
    budget: float = 100_000.0,
    seed: int | None = None,
    show: bool = False,
    export_geojson: bool = False,
    no_plots: bool = False,
) -> EvaluationTracker:
    """Evaluate a trained policy with deterministic rollouts (RECREATE_SPEC §3.8).

    Runs ``num_evaluations`` fresh episodes, records per-episode reward and
    final metrics (connectivity, efficiency, coverage, remaining budget),
    selects the best episode by **reward** (§3.8), and — unless
    ``no_plots`` — renders the best-solution map and metric/reward plots into
    ``run_context.output_dir`` via :mod:`bike_rl.plotting`.

    Args:
        model: A trained ``MaskablePPO`` (from :func:`bike_rl.training.train_model`
            or ``MaskablePPO.load``). Typed ``object`` to avoid a hard
            runtime import of ``sb3_contrib``; only ``model.predict`` is used.
        bike_graph: The original bike network (candidates are drawn from the
            walk graph; the env copies this and never mutates it).
        walk_graph: The walkable network.
        cfg: Config (uses ``cfg.seed`` when ``seed`` is None).
        run_context: Per-run output context for plot/GeoJSON paths.
        num_evaluations: Number of deterministic episodes.
        budget: Episode budget (passed via ``reset(options={"budget": …})``).
        seed: Base seed; episode ``i`` uses ``seed + i``. Defaults to ``cfg.seed``.
        show: If True, call ``plt.show()`` on each figure (headless default False — §5.10).
        export_geojson: If True, write ``suggested_bike_paths.geojson`` for the
            best solution (RECREATE_SPEC §8).
        no_plots: If True, skip all rendering (RECREATE_SPEC §9 ``--no_plots``).

    Returns:
        The populated :class:`EvaluationTracker`.
    """
    base_seed = cfg.seed if seed is None else seed
    tracker = EvaluationTracker()
    # public type is `object`; cast for the internal helper
    masked_model = cast("MaskablePPO", model)

    for i in range(num_evaluations):
        total_r, conn, eff, pop, brem, added = _run_episode(
            masked_model, bike_graph, walk_graph, cfg, run_context, budget, base_seed + i
        )
        tracker.rewards.append(total_r)
        tracker.connectivity.append(conn)
        tracker.efficiency.append(eff)
        tracker.population.append(pop)
        tracker.budget_remaining.append(brem)
        if total_r > tracker.best_reward:
            tracker.best_reward = total_r
            tracker.best_index = i
            tracker.best_added_edges = added

    tracker.n_episodes = num_evaluations
    logger.info(
        "evaluation: %d episodes, best_reward=%.4f (episode %d, %d added edges)",
        tracker.n_episodes,
        tracker.best_reward,
        tracker.best_index,
        len(tracker.best_added_edges),
    )

    if no_plots or tracker.best_index < 0:
        return tracker

    from bike_rl.plotting import (
        export_geojson as export_geojson_fn,
    )
    from bike_rl.plotting import (
        plot_metrics,
        plot_rewards,
        render_solution_map,
    )

    out_dir = run_context.ensure_output_dir()
    best_graph = bike_graph.copy()
    for u, v, d in tracker.best_added_edges:
        best_graph.add_edge(u, v, **d)
    render_solution_map(best_graph, tracker.best_added_edges, run_context, show=show)
    plot_metrics(tracker, run_context, show=show)
    plot_rewards(tracker.rewards, run_context, show=show, filename="evaluation_rewards.png")
    if export_geojson:
        export_geojson_fn(tracker.best_added_edges, run_context)
    _ = out_dir  # ensure dir created even if a plotting impl skips it
    return tracker
