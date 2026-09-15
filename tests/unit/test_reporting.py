"""
Unit tests for semdrift.reporting.
"""

from __future__ import annotations

import json
import pytest

from semdrift.detection.models import DriftResult
from semdrift.reporting.exceptions import ReportingError, ReportingInputError
from semdrift.reporting.json import JsonReporter
from semdrift.reporting.markdown import MarkdownReporter
from semdrift.reporting.terminal import TerminalReporter


@pytest.fixture
def sample_results() -> list[DriftResult]:
    """Provide representative DriftResult fixtures covering drifted, aligned, and undocumented."""
    return [
        DriftResult(
            file_path="src/auth.py",
            qualified_name="login",
            line_number=15,
            status="drifted",
            is_drift=True,
            drift_probability=0.82,
            threshold=0.5,
        ),
        DriftResult(
            file_path="src/auth.py",
            qualified_name="logout",
            line_number=45,
            status="aligned",
            is_drift=False,
            drift_probability=0.23,
            threshold=0.5,
        ),
        DriftResult(
            file_path="src/utils.py",
            qualified_name="format_date",
            line_number=10,
            status="undocumented",
            is_drift=False,
            drift_probability=None,
            threshold=0.5,
        ),
    ]


@pytest.fixture
def boundary_results() -> list[DriftResult]:
    """Provide a result exactly at threshold boundary (0.50) with status='drifted'."""
    return [
        DriftResult(
            file_path="src/calc.py",
            qualified_name="add",
            line_number=5,
            status="drifted",
            is_drift=True,
            drift_probability=0.50,
            threshold=0.5,
        )
    ]


# ============================================================================
# JSON Reporter Tests
# ============================================================================


class TestJsonReporter:
    """Verify JsonReporter output formatting, determinism, and contracts."""

    def test_json_output_is_valid_json(self, sample_results):
        reporter = JsonReporter()
        rendered = reporter.render(sample_results)
        parsed = json.loads(rendered)
        assert isinstance(parsed, list)
        assert len(parsed) == 3

    def test_json_preserves_all_fields(self, sample_results):
        reporter = JsonReporter()
        rendered = reporter.render(sample_results)
        parsed = json.loads(rendered)

        first = parsed[0]
        assert first["file_path"] == "src/auth.py"
        assert first["qualified_name"] == "login"
        assert first["line_number"] == 15
        assert first["status"] == "drifted"
        assert first["is_drift"] is True
        assert first["drift_probability"] == 0.82
        assert first["threshold"] == 0.5

    def test_json_preserves_null_for_undocumented(self, sample_results):
        reporter = JsonReporter()
        rendered = reporter.render(sample_results)
        parsed = json.loads(rendered)

        undoc = parsed[2]
        assert undoc["status"] == "undocumented"
        assert undoc["is_drift"] is False
        assert undoc["drift_probability"] is None
        # Verify JSON string contains literal null rather than 0
        assert '"drift_probability": null' in rendered

    def test_json_empty_input(self):
        reporter = JsonReporter()
        rendered = reporter.render([])
        assert json.loads(rendered) == []
        assert rendered.strip() == "[]"

    def test_json_order_preservation(self, sample_results):
        reporter = JsonReporter()
        rendered = reporter.render(sample_results)
        parsed = json.loads(rendered)

        assert [p["qualified_name"] for p in parsed] == ["login", "logout", "format_date"]

    def test_json_determinism(self, sample_results):
        reporter = JsonReporter()
        run1 = reporter.render(sample_results)
        run2 = reporter.render(sample_results)
        assert run1 == run2

    def test_json_malformed_input_raises(self):
        reporter = JsonReporter()
        with pytest.raises(ReportingInputError) as exc_info:
            reporter.render([{"not": "a DriftResult"}])  # type: ignore
        assert "Expected DriftResult instance" in str(exc_info.value)
        assert exc_info.value.index == 0


# ============================================================================
# Markdown Reporter Tests
# ============================================================================


class TestMarkdownReporter:
    """Verify MarkdownReporter tables, counts, formatting, and safety."""

    def test_markdown_summary_counts(self, sample_results):
        reporter = MarkdownReporter()
        rendered = reporter.render(sample_results)

        assert "| Total | 3 |" in rendered
        assert "| Drifted | 1 |" in rendered
        assert "| Aligned | 1 |" in rendered
        assert "| Undocumented | 1 |" in rendered

    def test_markdown_represents_drifted_result(self, sample_results):
        reporter = MarkdownReporter()
        rendered = reporter.render(sample_results)
        assert "| src/auth.py | login | 15 | drifted | 0.8200 | 0.50 |" in rendered

    def test_markdown_represents_aligned_result(self, sample_results):
        reporter = MarkdownReporter()
        rendered = reporter.render(sample_results)
        assert "| src/auth.py | logout | 45 | aligned | 0.2300 | 0.50 |" in rendered

    def test_markdown_represents_undocumented_result(self, sample_results):
        reporter = MarkdownReporter()
        rendered = reporter.render(sample_results)
        assert "| src/utils.py | format_date | 10 | undocumented | N/A | 0.50 |" in rendered

    def test_markdown_empty_input(self):
        reporter = MarkdownReporter()
        rendered = reporter.render([])

        assert "# SemDrift Report" in rendered
        assert "| Total | 0 |" in rendered
        assert "| Drifted | 0 |" in rendered
        assert "| Aligned | 0 |" in rendered
        assert "| Undocumented | 0 |" in rendered
        assert "## Results" in rendered

    def test_markdown_order_preservation(self, sample_results):
        reporter = MarkdownReporter()
        rendered = reporter.render(sample_results)

        pos_login = rendered.find("login")
        pos_logout = rendered.find("logout")
        pos_format = rendered.find("format_date")

        assert pos_login < pos_logout < pos_format

    def test_markdown_escapes_pipes(self):
        reporter = MarkdownReporter()
        result_with_pipes = [
            DriftResult(
                file_path="src/foo|bar.py",
                qualified_name="Class|Method.call",
                line_number=1,
                status="drifted",
                is_drift=True,
                drift_probability=0.75,
                threshold=0.5,
            )
        ]
        rendered = reporter.render(result_with_pipes)
        assert "src/foo\\|bar.py" in rendered
        assert "Class\\|Method.call" in rendered

    def test_markdown_determinism(self, sample_results):
        reporter = MarkdownReporter()
        run1 = reporter.render(sample_results)
        run2 = reporter.render(sample_results)
        assert run1 == run2

    def test_markdown_malformed_input_raises(self):
        reporter = MarkdownReporter()
        with pytest.raises(ReportingInputError) as exc_info:
            reporter.render(["not a DriftResult"])  # type: ignore
        assert "Expected DriftResult instance" in str(exc_info.value)
        assert exc_info.value.index == 0


# ============================================================================
# Terminal Reporter Tests
# ============================================================================


class TestTerminalReporter:
    """Verify TerminalReporter formatting, counts, and output clarity."""

    def test_terminal_summary_and_results(self, sample_results):
        reporter = TerminalReporter()
        rendered = reporter.render(sample_results)

        assert "SemDrift Detection Summary" in rendered
        assert "Total:        3" in rendered
        assert "Drifted:      1" in rendered
        assert "Aligned:      1" in rendered
        assert "Undocumented: 1" in rendered
        assert "[drifted] src/auth.py:15 - login (probability: 0.8200, threshold: 0.50)" in rendered
        assert "[aligned] src/auth.py:45 - logout (probability: 0.2300, threshold: 0.50)" in rendered
        assert "[undocumented] src/utils.py:10 - format_date (probability: N/A, threshold: 0.50)" in rendered

    def test_terminal_empty_input(self):
        reporter = TerminalReporter()
        rendered = reporter.render([])

        assert "Total:        0" in rendered
        assert "Drifted:      0" in rendered
        assert "No results found." in rendered

    def test_terminal_order_preservation(self, sample_results):
        reporter = TerminalReporter()
        rendered = reporter.render(sample_results)

        pos_login = rendered.find("login")
        pos_logout = rendered.find("logout")
        pos_format = rendered.find("format_date")

        assert pos_login < pos_logout < pos_format

    def test_terminal_determinism(self, sample_results):
        reporter = TerminalReporter()
        run1 = reporter.render(sample_results)
        run2 = reporter.render(sample_results)
        assert run1 == run2

    def test_terminal_malformed_input_raises(self):
        reporter = TerminalReporter()
        with pytest.raises(ReportingInputError) as exc_info:
            reporter.render([None])  # type: ignore
        assert "Expected DriftResult instance" in str(exc_info.value)
        assert exc_info.value.index == 0


# ============================================================================
# Cross-Format & Behavioral Integrity Tests
# ============================================================================


class TestReportingBehavioralIntegrity:
    """Verify cross-format consistency, pure presentation boundary, and immutability."""

    def test_boundary_result_retains_drifted_decision(self, boundary_results):
        """Boundary result at prob=0.50 and thresh=0.50 with status='drifted' must remain drifted."""
        json_out = JsonReporter().render(boundary_results)
        md_out = MarkdownReporter().render(boundary_results)
        term_out = TerminalReporter().render(boundary_results)

        assert '"status": "drifted"' in json_out
        assert "| drifted | 0.5000 | 0.50 |" in md_out
        assert "[drifted]" in term_out

    def test_reporting_does_not_recalculate_detection(self):
        """Even if probability < threshold, reporter must honor the existing status and is_drift."""
        counter_intuitive_result = [
            DriftResult(
                file_path="src/override.py",
                qualified_name="forced_drift",
                line_number=1,
                status="drifted",
                is_drift=True,
                drift_probability=0.20,  # Below threshold, but status is explicitly 'drifted'
                threshold=0.50,
            )
        ]

        json_out = JsonReporter().render(counter_intuitive_result)
        md_out = MarkdownReporter().render(counter_intuitive_result)
        term_out = TerminalReporter().render(counter_intuitive_result)

        # JSON
        parsed = json.loads(json_out)
        assert parsed[0]["status"] == "drifted"
        assert parsed[0]["is_drift"] is True

        # Markdown
        assert "| Drifted | 1 |" in md_out
        assert "| Aligned | 0 |" in md_out
        assert "| drifted | 0.2000 | 0.50 |" in md_out

        # Terminal
        assert "Drifted:      1" in term_out
        assert "Aligned:      0" in term_out
        assert "[drifted] src/override.py:1 - forced_drift" in term_out

    def test_reporting_does_not_mutate_input(self, sample_results):
        """Input sequence and objects must remain unmodified after rendering."""
        original_states = [
            (r.file_path, r.qualified_name, r.line_number, r.status, r.is_drift, r.drift_probability, r.threshold)
            for r in sample_results
        ]

        JsonReporter().render(sample_results)
        MarkdownReporter().render(sample_results)
        TerminalReporter().render(sample_results)

        current_states = [
            (r.file_path, r.qualified_name, r.line_number, r.status, r.is_drift, r.drift_probability, r.threshold)
            for r in sample_results
        ]

        assert original_states == current_states
