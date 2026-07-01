"""Tests for ``scripts/run_comparison.py`` CLI behaviour (plan 017).

Exercises the parser and the ``_objective_weights`` helper without network
access (``main()`` loads OSM graphs, so we test private helpers instead).
"""

from __future__ import annotations

import pytest

from scripts.run_comparison import _build_parser, _objective_weights

# ── Parser: default and legacy flags ──────────────────────────────────────


def test_default_weights_are_coverage_only() -> None:
    """Default invocation (no objective flag) yields coverage-only weights."""
    args = _build_parser().parse_args(["--city", "Otley, UK"])
    weights = _objective_weights(args)
    assert weights.connectivity == 0.0
    assert weights.coverage == 1.0
    assert weights.fragmentation == 0.0
    # coverage-only is now the default, so the flag is False by default.
    assert args.coverage_only is False
    assert args.weighted_objective is False


def test_weighted_objective_flag_yields_legacy_weights() -> None:
    """``--weighted-objective`` opts into the legacy weighted objective."""
    args = _build_parser().parse_args(["--city", "Otley, UK", "--weighted-objective"])
    weights = _objective_weights(args)
    assert weights.connectivity == 0.4
    assert weights.coverage == 0.4
    assert weights.fragmentation == 0.2


def test_coverage_only_flag_accepted_and_is_noop() -> None:
    """``--coverage-only`` is accepted for backward compatibility and is a no-op."""
    args = _build_parser().parse_args(["--city", "Otley, UK", "--coverage-only"])
    weights = _objective_weights(args)
    # Same coverage-only weights as the default — the flag changes nothing.
    assert weights.connectivity == 0.0
    assert weights.coverage == 1.0
    assert weights.fragmentation == 0.0


def test_coverage_only_and_weighted_are_mutually_exclusive() -> None:
    """Passing both objective flags should be rejected by argparse."""
    with pytest.raises(SystemExit):
        _build_parser().parse_args(
            ["--city", "Otley, UK", "--coverage-only", "--weighted-objective"]
        )


def test_weighted_flag_with_bbox() -> None:
    """``--weighted-objective`` works with the bbox path too."""
    args = _build_parser().parse_args(
        ["--bbox", "53.9", "53.9", "-1.69", "-1.69", "--weighted-objective"]
    )
    weights = _objective_weights(args)
    assert weights.connectivity == 0.4
    assert weights.coverage == 0.4
    assert weights.fragmentation == 0.2