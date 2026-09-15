"""
Python standard-library AST parser for extracting functions, methods, and docstrings.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Optional, Union

from semdrift.parser.models import CodeDocumentPair, ParseError


class PythonASTParser:
    """Extracts function and method definitions from Python source code using the AST.

    Discovers top-level functions, class methods, async functions/methods, and nested
    functions, capturing source code, docstrings, deterministic qualified names, and
    line locations into immutable CodeDocumentPair objects.

    Parameters:
        max_file_size_bytes: Maximum allowed file size in bytes before rejecting parsing.
            Defaults to 1,000,000 bytes (1 MB).
    """

    def __init__(self, max_file_size_bytes: int = 1_000_000) -> None:
        self.max_file_size_bytes = max_file_size_bytes

    def parse_source(
        self,
        source: str,
        file_path: str = "<string>",
    ) -> List[CodeDocumentPair]:
        """Parse source code string and extract all function/method definitions.

        Parameters:
            source: The raw Python source code string.
            file_path: The file path to attribute extracted records to.

        Returns:
            List of immutable CodeDocumentPair objects in source order.

        Raises:
            ParseError: If source contains Python syntax errors or parsing fails.
        """
        try:
            tree = ast.parse(source, filename=file_path)
        except SyntaxError as exc:
            raise ParseError(
                message=exc.msg or "Syntax error in source code",
                file_path=file_path,
                lineno=exc.lineno,
                cause=exc,
            ) from exc

        source_lines = source.splitlines()
        pairs: List[CodeDocumentPair] = []
        self._walk_scope(
            body=tree.body,
            source_lines=source_lines,
            file_path=file_path,
            scope=[],
            enclosing_class=None,
            out=pairs,
        )
        return pairs

    def parse_file(
        self,
        file_path: Union[str, Path],
        relative_to: Optional[Union[str, Path]] = None,
    ) -> List[CodeDocumentPair]:
        """Parse a Python source file from disk.

        Parameters:
            file_path: Path to the Python source file.
            relative_to: Optional base path to resolve relative file paths against.

        Returns:
            List of CodeDocumentPair instances.

        Raises:
            ParseError: If the file cannot be read, exceeds size limits, or contains syntax errors.
        """
        path = Path(file_path)

        # Normalize display path
        if relative_to is not None:
            try:
                display_path = path.resolve().relative_to(Path(relative_to).resolve()).as_posix()
            except ValueError:
                display_path = path.as_posix()
        else:
            display_path = path.as_posix()

        # Check file size limit
        try:
            file_size = path.stat().st_size
        except OSError as exc:
            raise ParseError(
                message=f"Cannot access file: {exc}",
                file_path=display_path,
                cause=exc,
            ) from exc

        if file_size > self.max_file_size_bytes:
            raise ParseError(
                message=f"File size ({file_size} bytes) exceeds limit of {self.max_file_size_bytes} bytes",
                file_path=display_path,
            )

        # Read source content
        try:
            source = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            raise ParseError(
                message=f"Failed to read file: {exc}",
                file_path=display_path,
                cause=exc,
            ) from exc

        return self.parse_source(source, file_path=display_path)

    def _walk_scope(
        self,
        body: list[ast.stmt],
        source_lines: list[str],
        file_path: str,
        scope: list[str],
        enclosing_class: Optional[str],
        out: list[CodeDocumentPair],
    ) -> None:
        """Walk AST statement lists recursively, tracking lexical scope and enclosing classes."""
        for node in body:
            if isinstance(node, ast.ClassDef):
                # Recurse into class body with class name appended to scope
                self._walk_scope(
                    body=node.body,
                    source_lines=source_lines,
                    file_path=file_path,
                    scope=scope + [node.name],
                    enclosing_class=node.name,
                    out=out,
                )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Determine qualified name
                qname = ".".join(scope + [node.name]) if scope else node.name
                is_method = enclosing_class is not None and bool(scope) and scope[-1] == enclosing_class
                class_name = enclosing_class if is_method else None
                is_async = isinstance(node, ast.AsyncFunctionDef)

                # Extract source code segment including decorators
                code = self._extract_source_segment(node, source_lines)

                # Docstring via standard ast.get_docstring (returns None if no docstring)
                docstring = ast.get_docstring(node)

                pair = CodeDocumentPair(
                    file_path=file_path,
                    qualified_name=qname,
                    line_number=node.lineno,
                    end_line_number=node.end_lineno,
                    code=code,
                    docstring=docstring,
                    is_method=is_method,
                    is_async=is_async,
                    class_name=class_name,
                )
                out.append(pair)

                # Recurse into function body to find nested functions/classes
                self._walk_scope(
                    body=node.body,
                    source_lines=source_lines,
                    file_path=file_path,
                    scope=scope + [node.name],
                    enclosing_class=None,  # Functions inside methods are nested functions, not methods
                    out=out,
                )

    def _extract_source_segment(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        source_lines: list[str],
    ) -> str:
        """Extract the exact source code range for a function including any decorators.

        Preserves original indentation relative to the surrounding context.
        """
        if node.decorator_list:
            start_line = node.decorator_list[0].lineno
        else:
            start_line = node.lineno

        end_line = node.end_lineno if node.end_lineno is not None else node.lineno

        # Slice lines (1-indexed inclusive to 0-indexed slice)
        extracted = source_lines[start_line - 1 : end_line]
        return "\n".join(extracted)
