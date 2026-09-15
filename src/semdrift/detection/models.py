"""
Detection layer result data contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DriftResult:
    """Structured immutable outcome representing the detection decision for a single function.

    Preserves source metadata from the underlying CodeDocumentPair / ModelPrediction,
    alongside the raw model probability and the explicit binary/status detection decision.

    Attributes:
        file_path: Normalized path to the source file.
        qualified_name: Fully-qualified deterministic name of the function.
        line_number: 1-indexed start line of definition.
        status: Detection outcome: 'drifted', 'aligned', or 'undocumented'.
        is_drift: True if status == 'drifted'; False if 'aligned' or 'undocumented'.
        drift_probability: Raw model probability in [0.0, 1.0], or None if undocumented.
        threshold: The decision threshold applied.
    """

    file_path: str
    qualified_name: str
    line_number: int
    status: str
    is_drift: bool
    drift_probability: Optional[float]
    threshold: float

    @property
    def is_undocumented(self) -> bool:
        """Return True if the function was undocumented."""
        return self.status == "undocumented"
