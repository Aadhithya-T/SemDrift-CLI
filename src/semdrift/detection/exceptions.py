"""
Exceptions for the SemDrift detection layer.
"""

from __future__ import annotations

from typing import Optional


class DetectionError(Exception):
    """Base exception for all detection errors in SemDrift."""

    def __init__(self, message: str, cause: Optional[BaseException] = None) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause

    def __str__(self) -> str:
        cause_str = f" (caused by {type(self.cause).__name__}: {self.cause})" if self.cause else ""
        return f"{self.message}{cause_str}"


class DetectionConfigError(DetectionError):
    """Raised when detection configuration parameters are invalid."""

    def __init__(self, message: str, cause: Optional[BaseException] = None) -> None:
        super().__init__(f"Invalid detection configuration: {message}", cause)


class DetectionInputError(DetectionError):
    """Raised when a ModelPrediction object contains invalid or malformed data."""

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        qualified_name: Optional[str] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        context_parts = []
        if file_path:
            context_parts.append(f"file='{file_path}'")
        if qualified_name:
            context_parts.append(f"qualified_name='{qualified_name}'")
        context_str = f" [{', '.join(context_parts)}]" if context_parts else ""

        super().__init__(f"Invalid prediction input{context_str}: {message}", cause)
        self.file_path = file_path
        self.qualified_name = qualified_name
