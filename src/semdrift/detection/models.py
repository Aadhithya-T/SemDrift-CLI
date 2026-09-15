"""
Detection layer result data contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
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

    def __post_init__(self) -> None:
        if self.status not in ("drifted", "aligned", "undocumented"):
            raise ValueError(
                f"DriftResult status must be one of ('drifted', 'aligned', 'undocumented'), got {self.status!r}"
            )
        if not isinstance(self.is_drift, bool):
            raise TypeError(
                f"DriftResult is_drift must be bool, got {type(self.is_drift).__name__}"
            )
        if self.status == "undocumented":
            if self.drift_probability is not None:
                raise ValueError(
                    f"DriftResult drift_probability must be None when status is 'undocumented', got {self.drift_probability}"
                )
        else:
            if self.drift_probability is None:
                raise ValueError(
                    f"DriftResult drift_probability must not be None when status is {self.status!r}"
                )
            if isinstance(self.drift_probability, bool) or not isinstance(self.drift_probability, (int, float)):
                raise TypeError(
                    f"DriftResult drift_probability must be numeric float or int, got {type(self.drift_probability).__name__}"
                )
            prob = float(self.drift_probability)
            if math.isnan(prob) or math.isinf(prob) or not (0.0 <= prob <= 1.0):
                raise ValueError(
                    f"DriftResult drift_probability must be within [0.0, 1.0], got {self.drift_probability}"
                )

    @property
    def is_undocumented(self) -> bool:
        """Return True if the function was undocumented."""
        return self.status == "undocumented"

