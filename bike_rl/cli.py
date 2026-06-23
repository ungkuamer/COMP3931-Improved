"""Command-line entry point for bike-path RL and optimiser experiments.

See RECREATE_SPEC.md §9.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from dataclasses import fields, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from bike_rl.config import Config
from bike_rl.evaluation import evaluate_and_visualize
from bike_rl.graph_utils import configure_osm_cache, load_bbox_graph, load_city_graph
from bike_rl.plotting import plot_rewards
from bike_rl.run_context import RunContext
from bike_rl.training import TrainingProgressCallback, make_vec_env, train_model

if TYPE_CHECKING:
    from sb3_contrib import MaskablePPO

logger = logging.getLogger(__name__)


def _safe_label(s: str) -> str:
    """Convert a string to a safe filesystem label.

    Lowercases the string and replaces any run of non-``[a-z0-9]`` characters
    with a single ``_``. Leading and trailing underscores are stripped.

    Args:
        s: Input string (e.g. ``"Otley, UK"``).

    Returns:
        A safe label (e.g. ``"otley_uk"``).
    """
    out = "".join(ch if ch.isalnum() else "_" for ch in s.lower())
    # collapse repeated underscores
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def _should_plot_training_rewards(no_plots: bool, rewards: list[float]) -> bool:
    """Decide whether the training-rewards plot should be produced.

    Args:
        no_plots: ``True`` if the user passed ``--no-plots``.
        rewards: The episode rewards collected during training (may be empty
            if no episodes terminated).

    Returns:
        ``True`` only when ``no_plots`` is ``False`` and ``rewards`` is
        non-empty.
    """
    return not no_plots and bool(rewards)


def load_config(path: str | None, seed: int, device: str) -> Config:
    """Load a ``Config`` from an optional YAML/TOML file, then override seed/device.

    If ``path`` is ``None``, returns ``Config(seed=seed, device=device)``.
    Otherwise detects the file format by extension and overlays the values
    on top of ``Config()`` defaults. Unknown keys are silently ignored
    (with a warning log message).

    Args:
        path: Path to a ``.yaml``, ``.yml``, or ``.toml`` config file, or
            ``None`` to use all defaults.
        seed: Seed value (always wins over the file).
        device: Torch device (always wins over the file).

    Returns:
        A ``Config`` with values merged from defaults, file (if given), and
        the ``seed``/``device`` CLI overrides.

    Raises:
        SystemExit: If a TOML file is provided on Python < 3.11 (no
            ``tomllib``).
    """
    cfg = Config()
    if path is not None:
        p = Path(path)
        raw: dict[str, object] = {}

        if p.suffix in (".yaml", ".yml"):
            import yaml  # type: ignore[import-untyped]

            with p.open() as f:
                raw = yaml.safe_load(f) or {}
        elif p.suffix == ".toml":
            try:
                import tomllib  # Python >= 3.11
            except ImportError:
                logger.error("TOML config requires Python 3.11+; use YAML instead.")
                sys.exit(1)
            data = tomllib.loads(p.read_text())
            # Check for [tool.bike_rl] table; fall back to top-level
            tool_val = data.get("tool", {})
            raw = (
                tool_val["bike_rl"]
                if isinstance(tool_val, dict) and "bike_rl" in tool_val
                else data
            )
        else:
            logger.warning("unknown config extension '%s', ignoring", p.suffix)

        # Filter to valid Config fields
        valid_names = {f.name for f in fields(Config)}
        overrides: dict[str, object] = {}
        for k, v in raw.items():
            if k in valid_names:
                overrides[k] = v
            else:
                logger.warning("ignoring unknown config key: %s", k)

        cfg = replace(cfg, **overrides)  # type: ignore[arg-type]

    return replace(cfg, seed=seed, device=device)


def _seed_everything(seed: int) -> None:
    """Seed Python ``random``, ``numpy``, and ``torch`` for reproducibility.

    Does not seed SB3 globally — the ``MaskablePPO(seed=cfg.seed)``
    constructor handles that internally.

    Args:
        seed: The integer seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass  # torch is a dependency, but guard defensively


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser.

    The parser is factored into a separate function so tests can inspect
    its arguments without running ``main()``.

    Returns:
        The configured ``ArgumentParser``.
    """
    parser = argparse.ArgumentParser(
        description="Bike-path expansion RL / optimiser experiments",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--city", type=str, help="City name for OSMnx download")
    group.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("N", "S", "E", "W"),
        help="Bounding box (N S E W)",
    )
    parser.add_argument("--budget", type=float, default=100000.0)
    parser.add_argument("--timesteps", type=int, default=10240)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--export-geojson", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=str, default=None, help="Path to YAML/TOML config file")
    parser.add_argument("--out-dir", type=str, default="bike_path_figures")
    return parser


def _format_runtime(seconds: float) -> str:
    """Format a duration in seconds to ``Hh Mm Ss``.

    Args:
        seconds: Duration in seconds (may be fractional).

    Returns:
        A string like ``"1h 1m 1s"``.
    """
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h}h {m}m {s}s"


def main(argv: list[str] | None = None) -> int:
    """Entry point: parse args, run the RL pipeline, write artefacts.

    Accepts an optional ``argv`` list so tests can call
    ``main(["--city", ...])`` without monkeypatching ``sys.argv``.

    Args:
        argv: Command-line argument list (without the program name). If
            ``None``, ``sys.argv[1:]`` is used.

    Returns:
        Exit code (0 on success, 1 on unhandled exception).
    """
    args = _build_parser().parse_args(argv)

    # Load config (default or from file) and override seed/device
    cfg = load_config(args.config, seed=args.seed, device=args.device)
    _seed_everything(cfg.seed)

    # Resolve n_envs: CLI default is None → cpu_count - 1
    n_envs = args.n_envs if args.n_envs is not None else max(1, (os.cpu_count() or 2) - 1)

    # Determine the output label from city or bbox
    if args.city:
        label = _safe_label(args.city)
    else:
        n, s, e, w = args.bbox
        label = _safe_label(f"bbox_{n}_{s}_{e}_{w}")

    run_context = RunContext.create(Path(args.out_dir), label)
    run_context.ensure_output_dir()

    # Configure logging (WARNING level to stdout, quiet during normal runs)
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s:%(name)s:%(message)s",
        stream=sys.stdout,
    )

    # Enable OSM cache and load graphs once in the parent process (§6.1)
    configure_osm_cache(cfg)
    if args.city:
        bike_graph = load_city_graph(args.city, cfg, network_type="bike")
        walk_graph = load_city_graph(args.city, cfg, network_type="walk")
    else:
        n, s, e, w = args.bbox
        bike_graph = load_bbox_graph(n, s, e, w, cfg, network_type="bike")
        walk_graph = load_bbox_graph(n, s, e, w, cfg, network_type="walk")

    start = time.perf_counter()

    # Training (skipped if --skip-training)
    model: MaskablePPO
    progress_cb: TrainingProgressCallback | None = None
    if not args.skip_training:
        from stable_baselines3.common.vec_env import SubprocVecEnv

        envs = make_vec_env(
            bike_graph,
            walk_graph,
            cfg,
            run_context,
            n_envs=n_envs,
            seed=cfg.seed,
            budget=args.budget,
            vec_env_cls=SubprocVecEnv,
        )
        try:
            progress_cb = TrainingProgressCallback(args.timesteps, progress_bar=False)
            model = train_model(
                envs,
                cfg,
                run_context,
                args.timesteps,
                progress_callback=progress_cb,
            )
            model.save(str(run_context.output_dir / "final_model.zip"))
        finally:
            envs.close()

        if _should_plot_training_rewards(
            args.no_plots,
            progress_cb.episode_rewards if progress_cb else [],
        ):
            plot_rewards(
                progress_cb.episode_rewards,
                run_context,
                show=args.show,
                filename="training_rewards.png",
            )
    else:
        # --skip-training: require --model-path
        if not args.model_path:
            _build_parser().error("--model-path is required with --skip-training")
        from sb3_contrib import MaskablePPO as _MaskablePPO

        model = _MaskablePPO.load(args.model_path)

    # Evaluation (both training and skip-training paths)
    tracker = evaluate_and_visualize(
        model,
        bike_graph,
        walk_graph,
        cfg,
        run_context,
        num_evaluations=args.eval_episodes,
        budget=args.budget,
        seed=cfg.seed,
        show=args.show,
        export_geojson=args.export_geojson,
        no_plots=args.no_plots,
    )

    elapsed = time.perf_counter() - start

    # Write run_summary.txt
    summary_lines = [
        f"run_id: {run_context.run_id}",
        f"city/bbox: {args.city or f'bbox {args.bbox}'}",
        f"budget: {args.budget}",
        f"timesteps: {args.timesteps if not args.skip_training else 'skipped'}",
        f"n_envs: {n_envs}",
        f"eval_episodes: {args.eval_episodes}",
        f"seed: {cfg.seed}",
        f"device: {cfg.device}",
        f"evaluation episodes: {tracker.n_episodes}",
        f"best_reward: {tracker.best_reward}",
        f"best_index: {tracker.best_index}",
        f"best_added_edges: {len(tracker.best_added_edges)}",
        f"runtime: {_format_runtime(elapsed)}",
    ]
    (run_context.output_dir / "run_summary.txt").write_text("\n".join(summary_lines) + "\n")

    # Final banner (one allowed print — RECREATE_SPEC §8)
    print(
        f"Run {run_context.run_id} complete in "
        f"{_format_runtime(elapsed)} "
        f"— best reward {tracker.best_reward:.4f}, "
        f"artefacts in {run_context.output_dir}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
