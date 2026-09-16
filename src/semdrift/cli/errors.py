"""
CLI-specific exceptions and error types.
"""

from __future__ import annotations

from typing import Optional


class CLIError(Exception):
    """Base exception for CLI errors."""

    def __init__(self, message: str, exit_code: int = 1, cause: Optional[BaseException] = None) -> None:
        super().__init__(message)
        self.message = message
        self.exit_code = exit_code
        self.cause = cause

    def __str__(self) -> str:
        return self.message


class CLIArgumentError(CLIError):
    """Raised when command-line arguments or configurations are invalid or help is requested."""

    def __init__(self, message: str, exit_code: int = 2, cause: Optional[BaseException] = None) -> None:
        super().__init__(message=message, exit_code=exit_code, cause=cause)


class CLIRuntimeError(CLIError):
    """Raised when an operational or environmental error occurs during CLI execution."""

    def __init__(self, message: str, exit_code: int = 1, cause: Optional[BaseException] = None) -> None:
        super().__init__(message=message, exit_code=exit_code, cause=cause)
