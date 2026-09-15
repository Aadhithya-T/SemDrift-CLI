"""
Terminal plain-text reporting for SemDrift detection results.
"""

from __future__ import annotations

from typing import Sequence

from semdrift.detection.models import DriftResult
from semdrift.reporting.exceptions import ReportingInputError


class TerminalReporter:
    """Renders DriftResult sequences into concise plain-text terminal output."""

    def render(self, results: Sequence[DriftResult]) -> str:
        """Render detection results as plain-text terminal output.

        Args:
            results: Sequence of DriftResult instances.

        Returns:
            Formatted plain-text string suitable for terminal display.

        Raises:
            ReportingInputError: If any element in results is not a DriftResult.
        """
        for idx, item in enumerate(results):
            if not isinstance(item, DriftResult):
                raise ReportingInputError(
                    f"Expected DriftResult instance, got {type(item).__name__}",
                    index=idx,
                )

        total = len(results)
        drifted = sum(1 for r in results if r.status == "drifted")
        aligned = sum(1 for r in results if r.status == "aligned")
        undocumented = sum(1 for r in results if r.status == "undocumented")

        lines = [
            "SemDrift Detection Summary",
            "==========================",
            f"Total:        {total}",
            f"Drifted:      {drifted}",
            f"Aligned:      {aligned}",
            f"Undocumented: {undocumented}",
        ]

        if not results:
            lines.append("")
            lines.append("No results found.")
            return "\n".join(lines) + "\n"

        lines.append("")
        lines.append("Results:")
        for r in results:
            prob_str = f"{r.drift_probability:.4f}" if r.drift_probability is not None else "N/A"
            thresh_str = f"{r.threshold:.2f}"
            lines.append(
                f"  [{r.status}] {r.file_path}:{r.line_number} - {r.qualified_name} "
                f"(probability: {prob_str}, threshold: {thresh_str})"
            )

        return "\n".join(lines) + "\n"
