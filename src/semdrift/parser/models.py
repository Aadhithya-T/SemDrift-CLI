"""
Core data models and exceptions for the SemDrift AST parser.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CodeDocumentPair:
    """Immutable representation of an extracted function/method and its documentation.

    This dataclass forms the core contract between the source AST parser and downstream
    analysis, embeddings, and drift detection stages.

    Attributes:
        file_path: Normalized path to the source file containing the function.
        qualified_name: Deterministic hierarchical name (e.g. 'top_func', 'Class.method',
            'outer.inner', 'Class.method.nested_helper').
        line_number: 1-indexed line number where the function definition starts.
        code: Extracted source code of the function, including decorators and body,
            preserving original relative indentation.
        docstring: Cleaned docstring content without surrounding quotes, or None if undocumented.
        end_line_number: 1-indexed line number where the function ends.
        is_method: True if directly defined within a class body.
        is_async: True if defined with `async def`.
        class_name: Name of the immediate enclosing class if this is a method, else None.
    """

    file_path: str
    qualified_name: str
    line_number: int
    code: str
    docstring: Optional[str] = None
    end_line_number: Optional[int] = None
    is_method: bool = False
    is_async: bool = False
    class_name: Optional[str] = None


class ParseError(Exception):
    """Exception raised when parsing or reading a Python source file fails.

    Attributes:
        file_path: Path to the file that failed parsing.
        message: Human-readable description of the error.
        lineno: Source line number where the syntax or parse error occurred, if available.
        cause: The underlying exception (e.g. SyntaxError, UnicodeDecodeError, OSError).
    """

    def __init__(
        self,
        message: str,
        file_path: str,
        lineno: Optional[int] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.file_path = file_path
        self.lineno = lineno
        self.cause = cause

    def __str__(self) -> str:
        loc = f":{self.lineno}" if self.lineno is not None else ""
        cause_suffix = f" (caused by {type(self.cause).__name__}: {self.cause})" if self.cause else ""
        return f"ParseError: {self.file_path}{loc}: {self.message}{cause_suffix}"
