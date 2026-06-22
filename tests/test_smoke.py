"""Smoke tests: the package imports and core dataclasses instantiate."""

from bike_rl.config import Config
from bike_rl.objective import ObjectiveWeights
from bike_rl.run_context import RunContext


def test_config_defaults() -> None:
    """Verify Config default field values match the spec."""
    cfg = Config()
    assert cfg.edge_cost_factor == 10.0
    assert cfg.road_priorities["primary"] == 5
    assert cfg.max_candidates_for_ilp == 300


def test_objective_weights_defaults() -> None:
    """Verify ObjectiveWeights default values sum to 1.0."""
    w = ObjectiveWeights()
    assert abs(w.connectivity - 0.4) < 1e-9
    assert abs(w.coverage - 0.4) < 1e-9
    assert abs(w.fragmentation - 0.2) < 1e-9


def test_run_context_create(tmp_path) -> None:
    """Verify RunContext.create produces a valid id and creates the output dir."""
    rc = RunContext.create(tmp_path, "test")
    assert rc.run_id.startswith("test_")
    rc.ensure_output_dir()
    assert rc.output_dir.exists()
