"""
Markdown reporting for SemDrift detection results.
"""

from __future__ import annotations

from typing import Sequence

from semdrift.detection.models import DriftResult
from semdrift.reporting.exceptions import ReportingInputError


def _escape_markdown(value: str) -> str:
    """Escape Markdown table delimiter characters."""
    return str(value).replace("|", "\\|")


class MarkdownReporter:
    """Renders DriftResult sequences into structured GitHub Flavored Markdown."""

    def render(self, results: Sequence[DriftResult]) -> str:
        """Render detection results as a Markdown report.

        Args:
            results: Sequence of DriftResult instances.

        Returns:
            Formatted Markdown document string with summary and results tables.

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
            "# SemDrift Report",
            "",
            "## Summary",
            "",
            "| Metric | Count |",
            "|---|---:|",
            f"| Total | {total} |",
            f"| Drifted | {drifted} |",
            f"| Aligned | {aligned} |",
            f"| Undocumented | {undocumented} |",
            "",
            "## Results",
            "",
            "| File | Function | Line | Status | Drift Probability | Threshold |",
            "|---|---|---:|---|---:|---:|",
        ]

        for r in results:
            file_escaped = _escape_markdown(r.file_path)
            func_escaped = _escape_markdown(r.qualified_name)
            prob_str = f"{r.drift_probability:.4f}" if r.drift_probability is not None else "N/A"
            thresh_str = f"{r.threshold:.2f}"
            lines.append(
                f"| {file_escaped} | {func_escaped} | {r.line_number} | {r.status} | {prob_str} | {thresh_str} |"
            )

        return "\n".join(lines) + "\n"
