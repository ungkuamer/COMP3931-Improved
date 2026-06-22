"""RunContext: per-run output directory and identity, replacing global state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class RunContext:
    """Per-run context: identity and output directory.

    Attributes:
        run_id: Stable run identifier (e.g. ``<safe_city>_<timestamp>``).
        output_dir: Directory for this run's artefacts (model, figures, summary).
        timestamp: When the run started (UTC).
    """

    run_id: str
    output_dir: Path
    timestamp: datetime

    @classmethod
    def create(cls, base_dir: Path, label: str, timestamp: datetime | None = None) -> RunContext:
        """Create a RunContext with a ``<label>_<timestamp>`` id under ``base_dir``."""
        ts = timestamp or datetime.now(timezone.utc)
        run_id = f"{label}_{ts.strftime('%Y%m%d_%H%M%S')}"
        return cls(run_id=run_id, output_dir=base_dir / run_id, timestamp=ts)

    def ensure_output_dir(self) -> Path:
        """Create ``output_dir`` if missing and return it."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir
