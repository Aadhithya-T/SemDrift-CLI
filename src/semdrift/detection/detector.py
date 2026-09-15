"""
Semantic drift detector implementation.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

from semdrift.detection.config import DetectionConfig
from semdrift.detection.exceptions import DetectionInputError
from semdrift.detection.models import DriftResult
from semdrift.model.models import ModelPrediction


class DriftDetector:
    """Applies detection policy to ModelPrediction records to produce DriftResult outcomes.

    Translates raw continuous drift probabilities into explicit binary / status detection
    decisions ('drifted', 'aligned', or 'undocumented') based on a configurable threshold.

    Parameters:
        config: Optional DetectionConfig. Defaults to default threshold (0.5).
    """

    def __init__(self, config: Optional[DetectionConfig] = None) -> None:
        self.config = config or DetectionConfig()

    def detect(self, predictions: Sequence[ModelPrediction]) -> List[DriftResult]:
        """Apply detection policy across a sequence of ModelPrediction records.

        Preserves 1-to-1 input ordering. Does not filter, group, or mutate input objects.

        Parameters:
            predictions: Sequence of ModelPrediction objects from the Model layer.

        Returns:
            List of DriftResult objects corresponding to input predictions in identical order.

        Raises:
            DetectionInputError: If any prediction object is malformed or carries invalid
                probability values.
        """
        if not predictions:
            return []

        results: List[DriftResult] = []

        for p in predictions:
            self._validate_prediction_structure(p)

            if p.is_undocumented:
                # Undocumented functions have no meaningful drift probability and are
                # explicitly marked as undocumented rather than falsely flagged as drift.
                results.append(
                    DriftResult(
                        file_path=p.file_path,
                        qualified_name=p.qualified_name,
                        line_number=p.line_number,
                        status="undocumented",
                        is_drift=False,
                        drift_probability=None,
                        threshold=self.config.drift_threshold,
                    )
                )
            else:
                prob = self._validate_documented_probability(p)
                is_drift = prob >= self.config.drift_threshold
                status = "drifted" if is_drift else "aligned"

                results.append(
                    DriftResult(
                        file_path=p.file_path,
                        qualified_name=p.qualified_name,
                        line_number=p.line_number,
                        status=status,
                        is_drift=is_drift,
                        drift_probability=prob,
                        threshold=self.config.drift_threshold,
                    )
                )

        return results

    def _validate_prediction_structure(self, p: ModelPrediction) -> None:
        """Verify presence and basic sanity of prediction fields."""
        if not hasattr(p, "file_path") or not isinstance(p.file_path, str):
            raise DetectionInputError("Missing or invalid 'file_path' attribute on prediction")
        if not hasattr(p, "qualified_name") or not isinstance(p.qualified_name, str):
            raise DetectionInputError("Missing or invalid 'qualified_name' attribute on prediction")
        if not hasattr(p, "line_number") or not isinstance(p.line_number, int):
            raise DetectionInputError("Missing or invalid 'line_number' attribute on prediction")
        if not hasattr(p, "is_undocumented") or not isinstance(p.is_undocumented, bool):
            raise DetectionInputError("Missing or invalid 'is_undocumented' attribute on prediction")

    def _validate_documented_probability(self, p: ModelPrediction) -> float:
        """Validate drift_probability on documented predictions."""
        prob = p.drift_probability

        if prob is None:
            raise DetectionInputError(
                "drift_probability cannot be None for a documented function (is_undocumented=False)",
                file_path=p.file_path,
                qualified_name=p.qualified_name,
            )

        # Reject boolean values (bool is a subclass of int in Python)
        if isinstance(prob, bool) or not isinstance(prob, (int, float)):
            raise DetectionInputError(
                f"drift_probability must be numeric (not bool), got {type(prob).__name__} ({prob!r})",
                file_path=p.file_path,
                qualified_name=p.qualified_name,
            )

        if math.isnan(prob) or math.isinf(prob):
            raise DetectionInputError(
                f"drift_probability cannot be NaN or Infinite, got {prob}",
                file_path=p.file_path,
                qualified_name=p.qualified_name,
            )

        if not (0.0 <= prob <= 1.0):
            raise DetectionInputError(
                f"drift_probability must be within [0.0, 1.0], got {prob}",
                file_path=p.file_path,
                qualified_name=p.qualified_name,
            )

        return float(prob)
