"""Candidate extraction for bike-network expansion. See RECREATE_SPEC.md §3.2, §5.8."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bike_rl.config import Config


@dataclass(frozen=True)
class Candidate:
    """A candidate edge for bike-lane addition."""

    u: int | str
    v: int | str
    length: float
    road_priority: int
    connects_to_bike_path: bool
    data: dict[str, object] = field(default_factory=dict)


def extract_candidates(bike_graph: object, walk_graph: object, cfg: Config) -> list[Candidate]:
    """Extract candidate edges from walking graph not in bike graph."""
    raise NotImplementedError("Plan 002")


def recompute_connects(candidates: list[Candidate], bike_nodes: set[int | str]) -> list[Candidate]:
    """Recompute connects_to_bike_path for all candidates."""
    raise NotImplementedError("Plan 002")
