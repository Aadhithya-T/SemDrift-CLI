"""
Repository scanner for discovering Python source files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Set, Union


class ScanError(Exception):
    """Exception raised when a scan target cannot be accessed or resolved."""

    def __init__(self, message: str, target_path: str, cause: Optional[BaseException] = None) -> None:
        super().__init__(message)
        self.message = message
        self.target_path = target_path
        self.cause = cause

    def __str__(self) -> str:
        cause_suffix = f" (caused by {type(self.cause).__name__}: {self.cause})" if self.cause else ""
        return f"ScanError: {self.target_path}: {self.message}{cause_suffix}"


class RepositoryScanner:
    """Discovers Python source files in a target repository or directory.

    Recursively traverses directory trees while pruning configured exclusion directories,
    filters for `.py` files, respects maximum file size limits, and supports targeting
    a single Python source file.

    Parameters:
        exclude_dirs: Set of directory names to skip during recursive traversal.
            Defaults to `{".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache", "node_modules"}`.
        max_file_size_bytes: Maximum allowed file size in bytes. Files exceeding this
            size are excluded from discovery. Defaults to 1,000,000 bytes (1 MB).
    """

    DEFAULT_EXCLUDES: Set[str] = {
        ".git",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".pytest_cache",
        "node_modules",
    }

    def __init__(
        self,
        exclude_dirs: Optional[Set[str]] = None,
        max_file_size_bytes: int = 1_000_000,
    ) -> None:
        self.exclude_dirs: Set[str] = set(exclude_dirs) if exclude_dirs is not None else set(self.DEFAULT_EXCLUDES)
        self.max_file_size_bytes: int = max_file_size_bytes

    def discover_files(self, target: Union[str, Path]) -> List[Path]:
        """Discover Python files for a given directory or single file path.

        Parameters:
            target: Path to a directory or a single `.py` file.

        Returns:
            A deterministically sorted list of Path objects for discovered `.py` files.

        Raises:
            ScanError: If the target path does not exist or cannot be accessed.
        """
        path = Path(target)

        if not path.exists():
            raise ScanError(
                message=f"Target path '{target}' does not exist",
                target_path=str(target),
            )

        # Single file target
        if path.is_file():
            if path.suffix.lower() != ".py":
                return []
            try:
                if path.stat().st_size > self.max_file_size_bytes:
                    return []
            except OSError as exc:
                raise ScanError(
                    message=f"Cannot stat file '{target}': {exc}",
                    target_path=str(target),
                    cause=exc,
                ) from exc
            return [path]

        # Directory target
        if not path.is_dir():
            return []

        discovered: List[Path] = []

        try:
            for root, dirs, files in os.walk(path):
                # Prune excluded directories in-place to avoid descending into them
                dirs[:] = [d for d in dirs if d not in self.exclude_dirs]

                for fname in sorted(files):
                    if not fname.endswith(".py"):
                        continue

                    full_path = Path(root) / fname

                    # Check file size guard
                    try:
                        if full_path.stat().st_size > self.max_file_size_bytes:
                            continue
                    except OSError:
                        # Skip files that disappear or cannot be stated during traversal
                        continue

                    discovered.append(full_path)
        except OSError as exc:
            raise ScanError(
                message=f"Failed during directory traversal: {exc}",
                target_path=str(target),
                cause=exc,
            ) from exc

        return sorted(discovered)
