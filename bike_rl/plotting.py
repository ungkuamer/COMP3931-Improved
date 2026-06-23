"""Headless-safe plotting for bike-path solutions. See RECREATE_SPEC.md §5.10, §3.8."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")  # headless-safe (§5.10); must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402

if TYPE_CHECKING:
    from bike_rl.evaluation import EvaluationTracker
    from bike_rl.run_context import RunContext

    _NXGraph = Any
else:
    _NXGraph = Any

logger = logging.getLogger(__name__)


def _node_xy(graph: Any, n: Any) -> tuple[float, float]:
    """Return ``(x, y)`` for node ``n`` from its attribute dict (x=lon, y=lat)."""
    data = graph.nodes[n]
    return float(data.get("x", 0.0)), float(data.get("y", 0.0))


def render_solution_map(
    graph: Any,
    added_paths: list[tuple[Any, Any, dict[str, Any]]],
    run_context: RunContext,
    show: bool = False,
    stats: dict[str, float] | None = None,
    filename: str = "best_solution_map.png",
) -> Path:
    """Render the bike network with ``added_paths`` highlighted in red (§3.8).

    Draws every edge of ``graph`` in grey and each edge in ``added_paths`` in
    red, using node ``x``/``y`` attributes. Optionally annotates ``stats``
    (e.g. connectivity/coverage) in the corner. Saves to
    ``run_context.output_dir / filename`` and returns the path.

    Args:
        graph: The full bike network (original + added edges).
        added_paths: ``(u, v, data)`` triples of the added edges to highlight.
        run_context: Per-run output context.
        show: If True, call ``plt.show()`` (default False — headless, §5.10).
        stats: Optional ``{label: value}`` dict rendered as an annotation.
        filename: Output filename within ``run_context.output_dir``.

    Returns:
        The path the figure was written to.
    """
    out_dir = run_context.ensure_output_dir()
    fig, ax = plt.subplots(figsize=(8, 8))
    # Base edges (grey).
    for u, v in graph.edges():
        x1, y1 = _node_xy(graph, u)
        x2, y2 = _node_xy(graph, v)
        ax.plot([x1, x2], [y1, y2], color="#cccccc", linewidth=1.0, zorder=1)
    # Added edges (red).
    for u, v, _d in added_paths:
        x1, y1 = _node_xy(graph, u)
        x2, y2 = _node_xy(graph, v)
        ax.plot([x1, x2], [y1, y2], color="red", linewidth=2.5, zorder=3)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Suggested bike lanes (red)")
    ax.set_aspect("equal", adjustable="datalim")
    if stats:
        text = "\n".join(f"{k}: {v:.4f}" for k, v in stats.items())
        ax.text(
            0.02,
            0.98,
            text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            family="monospace",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )
    out_path = out_dir / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_metrics(
    tracker: EvaluationTracker,
    run_context: RunContext,
    show: bool = False,
    filename: str = "evaluation_metrics.png",
) -> Path:
    """Plot connectivity/efficiency/coverage over evaluation episodes (§3.8).

    Saves to ``run_context.output_dir / filename`` and returns the path.
    """
    out_dir = run_context.ensure_output_dir()
    fig, ax = plt.subplots(figsize=(8, 5))
    episodes = list(range(1, tracker.n_episodes + 1))
    ax.plot(episodes, tracker.connectivity, marker="o", label="connectivity")
    ax.plot(episodes, tracker.efficiency, marker="s", label="efficiency")
    ax.plot(episodes, tracker.population, marker="^", label="coverage")
    ax.set_xlabel("evaluation episode")
    ax.set_ylabel("metric value [0, 1]")
    ax.set_title("Evaluation metrics per episode")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="best")
    out_path = out_dir / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_rewards(
    rewards: list[float],
    run_context: RunContext,
    show: bool = False,
    filename: str = "training_rewards.png",
) -> Path:
    """Plot a reward history (training or evaluation) and save it (§3.8/§9).

    ``filename`` selects between ``training_rewards.png`` (default, training
    callback rewards) and ``evaluation_rewards.png`` (evaluation rewards).
    """
    out_dir = run_context.ensure_output_dir()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(range(1, len(rewards) + 1)), rewards, marker="o", color="tab:blue")
    ax.set_xlabel("episode")
    ax.set_ylabel("total reward")
    ax.set_title("Reward history")
    out_path = out_dir / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def export_geojson(
    added_edges: list[tuple[Any, Any, dict[str, Any]]],
    run_context: RunContext,
    graph: Any | None = None,
    filename: str = "suggested_bike_paths.geojson",
) -> Path:
    """Write the added bike-lane edges to a GeoJSON file (RECREATE_SPEC §8).

    Each added edge becomes a LineString feature between its endpoint
    ``x``/``y`` coordinates. If ``graph`` is given, coordinates are read from
    its node attributes; otherwise the edge ``data`` dict's ``x``/``y`` are
    used if present (else 0/0). Uses geopandas + shapely.

    Args:
        added_edges: ``(u, v, data)`` triples of added edges.
        run_context: Per-run output context.
        graph: Optional graph to read node coordinates from.
        filename: Output filename within ``run_context.output_dir``.

    Returns:
        The path the GeoJSON was written to.
    """
    import geopandas as gpd
    from shapely.geometry import LineString

    out_dir = run_context.ensure_output_dir()
    geometries = []
    properties: list[dict[str, Any]] = []
    for u, v, d in added_edges:
        if graph is not None:
            x1, y1 = _node_xy(graph, u)
            x2, y2 = _node_xy(graph, v)
        else:
            x1, y1 = float(d.get("x", 0.0)), float(d.get("y", 0.0))
            x2, y2 = x1, y1
        geometries.append(LineString([(x1, y1), (x2, y2)]))
        props = {k: v for k, v in d.items() if isinstance(v, (str, int, float, bool))}
        props["u"] = str(u)
        props["v"] = str(v)
        properties.append(props)
    gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs="EPSG:4326")
    out_path = out_dir / filename
    gdf.to_file(out_path, driver="GeoJSON")
    return out_path
