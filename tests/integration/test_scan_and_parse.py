"""
Integration test demonstrating the end-to-end scanner -> parser pipeline.

Verifies that RepositoryScanner discovers files independently and PythonASTParser
extracts CodeDocumentPair records across multiple files, while gracefully handling
individual broken files without crashing the broader scan workflow.
"""

from pathlib import Path

from semdrift.parser.ast_parser import PythonASTParser
from semdrift.parser.models import CodeDocumentPair, ParseError
from semdrift.scanner.repository import RepositoryScanner


def test_end_to_end_scan_and_parse(tmp_path: Path):
    """Verify repository scanning and AST parsing on a multi-file project."""
    # 1. Scaffolding a realistic multi-file repository
    repo_root = tmp_path / "mock_project"
    repo_root.mkdir()

    # Valid module A
    mod_a = repo_root / "service.py"
    mod_a.write_text(
        '''"""Service module."""

def start_service(port: int = 8080) -> bool:
    """Start the network service on the specified port."""
    return True

class Worker:
    """Background worker class."""

    def run(self) -> None:
        """Run worker loop."""
        pass
''',
        encoding="utf-8",
    )

    # Valid module B in nested directory
    nested_dir = repo_root / "utils"
    nested_dir.mkdir()
    mod_b = nested_dir / "math_ops.py"
    mod_b.write_text(
        '''"""Math operations."""

async def async_multiply(a: int, b: int) -> int:
    """Multiply two integers asynchronously."""
    return a * b

def undocumented_helper(x: int) -> int:
    return x + 1
''',
        encoding="utf-8",
    )

    # Malformed module (syntax error)
    bad_mod = repo_root / "broken.py"
    bad_mod.write_text(
        '''def syntax_fail(
    x, y
    return
''',
        encoding="utf-8",
    )

    # Excluded directory with Python file
    venv_dir = repo_root / ".venv"
    venv_dir.mkdir()
    (venv_dir / "internal.py").write_text("def hidden(): pass\n", encoding="utf-8")

    # 2. Scanner execution (independent component)
    scanner = RepositoryScanner()
    discovered_files = scanner.discover_files(repo_root)

    discovered_names = [f.name for f in discovered_files]
    assert "service.py" in discovered_names
    assert "math_ops.py" in discovered_names
    assert "broken.py" in discovered_names
    assert "internal.py" not in discovered_names  # Excluded

    # 3. Parser execution on discovered files (independent component)
    parser = PythonASTParser()
    all_pairs: list[CodeDocumentPair] = []
    failed_files: list[tuple[str, ParseError]] = []

    for file_path in discovered_files:
        try:
            pairs = parser.parse_file(file_path, relative_to=repo_root)
            all_pairs.extend(pairs)
        except ParseError as exc:
            failed_files.append((str(file_path), exc))

    # 4. Verify outcomes
    # Exactly one file failed (broken.py)
    assert len(failed_files) == 1
    failed_path, parse_error = failed_files[0]
    assert "broken.py" in failed_path
    assert parse_error.lineno is not None

    # Successfully extracted records from valid files
    assert len(all_pairs) == 4
    qnames = {p.qualified_name for p in all_pairs}
    assert qnames == {
        "start_service",
        "Worker.run",
        "async_multiply",
        "undocumented_helper",
    }

    # Verify attributes on extracted pairs
    start_svc = next(p for p in all_pairs if p.qualified_name == "start_service")
    assert start_svc.file_path == "service.py"
    assert start_svc.docstring == "Start the network service on the specified port."
    assert start_svc.is_method is False

    worker_run = next(p for p in all_pairs if p.qualified_name == "Worker.run")
    assert worker_run.file_path == "service.py"
    assert worker_run.is_method is True
    assert worker_run.class_name == "Worker"
    assert worker_run.docstring == "Run worker loop."

    async_mult = next(p for p in all_pairs if p.qualified_name == "async_multiply")
    assert async_mult.file_path == "utils/math_ops.py"
    assert async_mult.is_async is True
    assert async_mult.docstring == "Multiply two integers asynchronously."

    undoc = next(p for p in all_pairs if p.qualified_name == "undocumented_helper")
    assert undoc.file_path == "utils/math_ops.py"
    assert undoc.docstring is None
