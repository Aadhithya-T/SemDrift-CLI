"""
Exceptions for the SemDrift reporting layer.
"""

from __future__ import annotations

from typing import Optional


class ReportingError(Exception):
    """Base exception for all reporting errors in SemDrift."""

    def __init__(self, message: str, cause: Optional[BaseException] = None) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause

    def __str__(self) -> str:
        cause_str = f" (caused by {type(self.cause).__name__}: {self.cause})" if self.cause else ""
        return f"{self.message}{cause_str}"


class ReportingInputError(ReportingError):
    """Raised when an object in the input sequence is not a valid DriftResult."""

    def __init__(
        self,
        message: str,
        index: Optional[int] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        idx_str = f" at index {index}" if index is not None else ""
        super().__init__(f"Invalid reporting input{idx_str}: {message}", cause)
        self.index = index
