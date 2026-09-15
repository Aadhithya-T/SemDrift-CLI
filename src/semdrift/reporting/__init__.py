"""
semdrift.reporting — Output formatting and serialization layer.

Transforms DriftResult sequences into structured JSON, Markdown, and plain-text
terminal presentations without performing detection, inference, or file I/O.
"""

from semdrift.reporting.exceptions import (
    ReportingError,
    ReportingInputError,
)
from semdrift.reporting.json import JsonReporter
from semdrift.reporting.markdown import MarkdownReporter
from semdrift.reporting.terminal import TerminalReporter

__all__ = [
    "JsonReporter",
    "MarkdownReporter",
    "TerminalReporter",
    "ReportingError",
    "ReportingInputError",
]
