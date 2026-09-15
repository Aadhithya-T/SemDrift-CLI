"""
Configuration dataclass for the SemDrift detection layer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from semdrift.detection.exceptions import DetectionConfigError


@dataclass(frozen=True)
class DetectionConfig:
    """Configuration settings for semantic drift decision logic.

    Attributes:
        drift_threshold: Threshold probability in [0.0, 1.0] above or equal to which
            a function is classified as 'drifted'. Defaults to 0.5 (a technical decision
            boundary default, not an empirically validated production optimum).
    """

    drift_threshold: float = 0.5

    def __post_init__(self) -> None:
        val = self.drift_threshold
        # Reject booleans (bool is a subclass of int in Python)
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise DetectionConfigError(
                f"drift_threshold must be a numeric float or int, got {type(val).__name__} ({val!r})"
            )

        if math.isnan(val) or math.isinf(val):
            raise DetectionConfigError(
                f"drift_threshold cannot be NaN or Infinite, got {val}"
            )

        if not (0.0 <= val <= 1.0):
            raise DetectionConfigError(
                f"drift_threshold must be within [0.0, 1.0], got {val}"
            )

        object.__setattr__(self, "drift_threshold", float(val))
