"""
JSON reporting for SemDrift detection results.
"""

from __future__ import annotations

import json
from typing import Sequence

from semdrift.detection.models import DriftResult
from semdrift.reporting.exceptions import ReportingInputError


class JsonReporter:
    """Renders DriftResult sequences into deterministic, structured JSON."""

    def render(self, results: Sequence[DriftResult]) -> str:
        """Render detection results as formatted JSON.

        Args:
            results: Sequence of DriftResult instances.

        Returns:
            Formatted JSON string representing the results array.

        Raises:
            ReportingInputError: If any element in results is not a DriftResult.
        """
        records = []
        for idx, item in enumerate(results):
            if not isinstance(item, DriftResult):
                raise ReportingInputError(
                    f"Expected DriftResult instance, got {type(item).__name__}",
                    index=idx,
                )
            records.append(
                {
                    "file_path": item.file_path,
                    "qualified_name": item.qualified_name,
                    "line_number": item.line_number,
                    "status": item.status,
                    "is_drift": item.is_drift,
                    "drift_probability": item.drift_probability,
                    "threshold": item.threshold,
                }
            )

        return json.dumps(records, indent=2)
