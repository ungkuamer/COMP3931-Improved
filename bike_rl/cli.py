"""Command-line entry point for bike-path RL and optimiser experiments.

See RECREATE_SPEC.md §9.
"""

from __future__ import annotations

import argparse


def main() -> int:
    """Entry point: parse args, delegate to training or optimiser."""
    parser = argparse.ArgumentParser(description="Bike-path expansion RL / optimiser experiments")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--city", type=str, help="City name for OSMnx download")
    group.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("N", "S", "E", "W"),
        help="Bounding box (N S E W)",
    )
    parser.add_argument("--budget", type=float, default=10000.0)
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--export-geojson", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=str, help="Path to YAML/TOML config file")
    parser.add_argument("--out-dir", type=str, default="output")
    parser.parse_args()
    raise NotImplementedError("Full CLI in plan 007")
