"""
Exceptions for the SemDrift model layer.
"""

from __future__ import annotations

from typing import Optional


class ModelError(Exception):
    """Base exception for all model and inference errors in SemDrift."""

    def __init__(self, message: str, cause: Optional[BaseException] = None) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause

    def __str__(self) -> str:
        cause_str = f" (caused by {type(self.cause).__name__}: {self.cause})" if self.cause else ""
        return f"{self.message}{cause_str}"


class ModelLoadError(ModelError):
    """Raised when model checkpoint or tokenizer fails to load."""

    def __init__(
        self,
        message: str,
        checkpoint_path: Optional[str] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message, cause)
        self.checkpoint_path = checkpoint_path

    def __str__(self) -> str:
        loc = f" [checkpoint='{self.checkpoint_path}']" if self.checkpoint_path else ""
        cause_str = f" (caused by {type(self.cause).__name__}: {self.cause})" if self.cause else ""
        return f"ModelLoadError{loc}: {self.message}{cause_str}"


class ModelInferenceError(ModelError):
    """Raised when tensor creation or model forward pass fails during inference."""

    def __init__(
        self,
        message: str,
        batch_size: Optional[int] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message, cause)
        self.batch_size = batch_size

    def __str__(self) -> str:
        bs_str = f" [batch_size={self.batch_size}]" if self.batch_size is not None else ""
        cause_str = f" (caused by {type(self.cause).__name__}: {self.cause})" if self.cause else ""
        return f"ModelInferenceError{bs_str}: {self.message}{cause_str}"
