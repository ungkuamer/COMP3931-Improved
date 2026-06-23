"""Tests for ``bike_rl.cli``. See RECREATE_SPEC.md §9.

Patches OSM-loading functions to return synthetic tiny graphs so all tests
run without network access. The CLI module imports ``load_city_graph`` and
``load_bbox_graph`` from ``bike_rl.graph_utils`` into its own namespace, so
patches must be applied to ``bike_rl.cli``, **not** to ``bike_rl.graph_utils``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
import pytest

from bike_rl.cli import (
    _build_parser,
    _format_runtime,
    _safe_label,
    _should_plot_training_rewards,
    load_config,
    main,
)

if TYPE_CHECKING:
    _NXGraph = nx.MultiDiGraph[Any, Any, Any]
else:
    _NXGraph = nx.MultiDiGraph


class _FakeModel:
    """Stand-in for MaskablePPO: always picks the first legal action.

    ``evaluate_and_visualize`` only calls ``model.predict(obs, deterministic=,
    action_masks=)``; this fake returns the first unmasked index so the rollout
    is deterministic and exercises the env without training a real PPO.
    Also provides a no-op ``save`` for the training-path stub.
    """

    def predict(
        self, observation: Any, deterministic: bool = False, action_masks: Any = None
    ) -> Any:
        mask = np.asarray(action_masks) if (action_masks is not None) else None
        idx = int(np.argmax(mask)) if mask is not None and bool(mask.any()) else 0
        return np.array(idx, dtype=np.int64), None

    def save(self, path: str) -> None:
        """No-op save; real models write a zip file."""
        return None


# ── Argparse tests ─────────────────────────────────────────────────


def test_parser_city_bbox_mutually_exclusive() -> None:
    """--city and --bbox cannot be used together (SystemExit 2)."""
    parser = _build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--city", "X", "--bbox", "1", "2", "3", "4"])
    assert exc.value.code == 2


def test_parser_requires_city_or_bbox() -> None:
    """At least one of --city or --bbox is required (SystemExit 2)."""
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_parser_accepts_verbose() -> None:
    """--verbose is accepted and defaults to False."""
    parser = _build_parser()
    args = parser.parse_args(["--city", "X", "--verbose"])
    assert args.verbose is True
    args_default = parser.parse_args(["--city", "X"])
    assert args_default.verbose is False


# ── load_config tests ──────────────────────────────────────────────


def test_load_config_yaml_overrides(tmp_path: Path) -> None:
    """load_config reads YAML overrides and sets seed/device from CLI."""
    yml = tmp_path / "cfg.yaml"
    yml.write_text("budget_efficiency_cap: 99.0\nw_connectivity: 0.5\nunknown_key: 123\n")
    cfg = load_config(str(yml), seed=7, device="cpu")
    assert cfg.budget_efficiency_cap == 99.0
    assert cfg.w_connectivity == 0.5
    assert cfg.seed == 7
    assert cfg.device == "cpu"
    # Unknown key ignored without error


def test_load_config_none_returns_defaults(tmp_path: Path) -> None:
    """load_config(None, ...) returns Config with seed/device overridden."""
    cfg = load_config(None, seed=42, device="cuda")
    assert cfg.seed == 42
    assert cfg.device == "cuda"
    # Other fields at defaults
    assert cfg.budget_efficiency_cap == 200.0


# ── _safe_label tests ──────────────────────────────────────────────


def test_safe_label() -> None:
    """_safe_label converts strings to safe filesystem labels."""
    assert _safe_label("Otley, UK") == "otley_uk"
    assert _safe_label("bbox_1.0_2.0_3.0_4.0") == "bbox_1_0_2_0_3_0_4_0"
    assert _safe_label("  Hello  World  ") == "hello_world"
    assert _safe_label("abc") == "abc"


# ── _should_plot_training_rewards tests ────────────────────────────


def test_should_plot_training_rewards_true() -> None:
    """Returns True when no_plots is False and rewards is non-empty."""
    assert _should_plot_training_rewards(no_plots=False, rewards=[1.0, 2.0]) is True


def test_should_plot_training_rewards_false_when_no_plots() -> None:
    """Returns False when no_plots is True regardless of rewards."""
    assert _should_plot_training_rewards(no_plots=True, rewards=[1.0]) is False


def test_should_plot_training_rewards_false_when_empty() -> None:
    """Returns False when rewards is empty regardless of no_plots."""
    assert _should_plot_training_rewards(no_plots=False, rewards=[]) is False


# ── _format_runtime tests ──────────────────────────────────────────


def test_format_runtime() -> None:
    """_format_runtime converts seconds to Hh Mm Ss."""
    assert _format_runtime(3661) == "1h 1m 1s"
    assert _format_runtime(0) == "0h 0m 0s"
    assert _format_runtime(3600) == "1h 0m 0s"
    assert _format_runtime(61) == "0h 1m 1s"


# ── End-to-end: skip-training path ─────────────────────────────────


def test_main_skip_training_loads_model_and_evaluates(
    tmp_path: Path,
    tiny_bike_graph: _NXGraph,
    tiny_walk_graph: _NXGraph,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--skip-training loads a model and runs evaluation; writes run_summary.txt.

    OSM loaders are stubbed to return tiny graphs; MaskablePPO.load returns
    a fake model. Evaluation runs for real on the tiny graphs with
    no_plots=True, exercising the integration.
    """
    # Patch OSM loaders (cli imports these names into its own namespace)
    monkeypatch.setattr(
        "bike_rl.cli.load_city_graph",
        lambda city, cfg, network_type="bike": (
            tiny_bike_graph if network_type == "bike" else tiny_walk_graph
        ),
    )
    # Patch MaskablePPO.load to return a fake model
    monkeypatch.setattr(
        "sb3_contrib.MaskablePPO.load",
        lambda path, **kwargs: _FakeModel(),
    )

    ret = main(
        [
            "--city",
            "Tiny",
            "--skip-training",
            "--model-path",
            "dummy.zip",
            "--no-plots",
            "--out-dir",
            str(tmp_path),
            "--budget",
            "100000",
            "--eval-episodes",
            "1",
        ]
    )
    assert ret == 0

    # Verify run_summary.txt was written
    run_dirs = list(tmp_path.glob("tiny_*"))
    assert len(run_dirs) == 1, f"Expected exactly one run dir, found {run_dirs}"
    summary = run_dirs[0] / "run_summary.txt"
    assert summary.exists()
    text = summary.read_text()
    assert "skipped" in text  # timesteps: skipped
    assert "best_reward" in text
    # No final_model.zip (skip-training)
    assert not (run_dirs[0] / "final_model.zip").exists()


# ── End-to-end: training path ─────────────────────────────────────


def test_main_training_path_writes_artefacts(
    tmp_path: Path,
    tiny_bike_graph: _NXGraph,
    tiny_walk_graph: _NXGraph,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Training path creates run dir, run_summary.txt, and calls evaluate.

    OSM loaders and train_model are stubbed to avoid real computation.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv

    # Patch OSM loaders
    monkeypatch.setattr(
        "bike_rl.cli.load_city_graph",
        lambda city, cfg, network_type="bike": (
            tiny_bike_graph if network_type == "bike" else tiny_walk_graph
        ),
    )
    # Patch SubprocVecEnv -> DummyVecEnv to avoid spawning processes
    monkeypatch.setattr(
        "stable_baselines3.common.vec_env.SubprocVecEnv",
        DummyVecEnv,
    )

    # Patch train_model to return a fake model (no real training)
    def _fake_train_model(
        envs: object,
        cfg: object,
        run_context: object,
        total_timesteps: int,
        progress_callback: object = None,
    ) -> _FakeModel:
        return _FakeModel()

    monkeypatch.setattr("bike_rl.training.train_model", _fake_train_model)

    ret = main(
        [
            "--city",
            "Tiny",
            "--timesteps",
            "1024",
            "--n-envs",
            "1",
            "--no-plots",
            "--out-dir",
            str(tmp_path),
            "--budget",
            "100000",
            "--eval-episodes",
            "1",
        ]
    )
    assert ret == 0

    run_dirs = list(tmp_path.glob("tiny_*"))
    assert len(run_dirs) == 1, f"Expected exactly one run dir, found {run_dirs}"
    summary = run_dirs[0] / "run_summary.txt"
    assert summary.exists()
    text = summary.read_text()
    assert "timesteps: 1024" in text


# ── --no-plots skips plotting ──────────────────────────────────────


def test_main_no_plots_skips_plotting(
    tmp_path: Path,
    tiny_bike_graph: _NXGraph,
    tiny_walk_graph: _NXGraph,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--no-plots produces no .png or .geojson files in the run dir."""
    monkeypatch.setattr(
        "bike_rl.cli.load_city_graph",
        lambda city, cfg, network_type="bike": (
            tiny_bike_graph if network_type == "bike" else tiny_walk_graph
        ),
    )
    monkeypatch.setattr(
        "sb3_contrib.MaskablePPO.load",
        lambda path, **kwargs: _FakeModel(),
    )

    ret = main(
        [
            "--city",
            "Tiny",
            "--skip-training",
            "--model-path",
            "dummy.zip",
            "--no-plots",
            "--out-dir",
            str(tmp_path),
            "--budget",
            "100000",
            "--eval-episodes",
            "1",
        ]
    )
    assert ret == 0

    run_dirs = list(tmp_path.glob("tiny_*"))
    assert len(run_dirs) == 1
    png_files = list(run_dirs[0].glob("*.png"))
    geojson_files = list(run_dirs[0].glob("*.geojson"))
    assert len(png_files) == 0, f"Expected no PNGs, found {png_files}"
    assert len(geojson_files) == 0, f"Expected no GeoJSONs, found {geojson_files}"


# ── --export-geojson flag ──────────────────────────────────────────


def test_main_export_geojson_flag(
    tmp_path: Path,
    tiny_bike_graph: _NXGraph,
    tiny_walk_graph: _NXGraph,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--export-geojson writes suggested_bike_paths.geojson."""
    monkeypatch.setattr(
        "bike_rl.cli.load_city_graph",
        lambda city, cfg, network_type="bike": (
            tiny_bike_graph if network_type == "bike" else tiny_walk_graph
        ),
    )
    monkeypatch.setattr(
        "sb3_contrib.MaskablePPO.load",
        lambda path, **kwargs: _FakeModel(),
    )

    ret = main(
        [
            "--city",
            "Tiny",
            "--skip-training",
            "--model-path",
            "dummy.zip",
            "--export-geojson",
            "--out-dir",
            str(tmp_path),
            "--budget",
            "100000",
            "--eval-episodes",
            "1",
        ]
    )
    assert ret == 0

    run_dirs = list(tmp_path.glob("tiny_*"))
    assert len(run_dirs) == 1
    geojson = run_dirs[0] / "suggested_bike_paths.geojson"
    assert geojson.exists()
