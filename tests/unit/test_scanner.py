"""
Unit tests for semdrift.scanner.repository.
"""

from pathlib import Path
import pytest

from semdrift.scanner.repository import RepositoryScanner, ScanError


FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


class TestRepositoryScanner:
    """Verify RepositoryScanner file discovery, filtering, and exclusion behaviors."""

    @pytest.fixture
    def scanner(self):
        return RepositoryScanner()

    def test_single_file_discovery(self, scanner):
        target = FIXTURES_DIR / "simple_function.py"
        files = scanner.discover_files(target)

        assert len(files) == 1
        assert files[0].resolve() == target.resolve()

    def test_single_non_python_file_returns_empty(self, scanner, tmp_path):
        text_file = tmp_path / "notes.txt"
        text_file.write_text("hello", encoding="utf-8")

        files = scanner.discover_files(text_file)
        assert files == []

    def test_recursive_directory_discovery(self, scanner):
        files = scanner.discover_files(FIXTURES_DIR)
        assert len(files) > 0

        # All discovered files must end with .py
        for f in files:
            assert f.suffix == ".py"

        names = [f.name for f in files]
        assert "simple_function.py" in names
        assert "class_methods.py" in names
        assert "submodule.py" in names

    def test_default_exclusions(self, tmp_path):
        # Setup repo with excluded and included directories
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "git_hook.py").write_text("x = 1", encoding="utf-8")

        (tmp_path / "venv").mkdir()
        (tmp_path / "venv" / "lib.py").write_text("x = 1", encoding="utf-8")

        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "cached.py").write_text("x = 1", encoding="utf-8")

        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "valid.py").write_text("x = 1", encoding="utf-8")

        scanner = RepositoryScanner()
        discovered = scanner.discover_files(tmp_path)

        assert len(discovered) == 1
        assert discovered[0].name == "valid.py"

    def test_custom_exclusions(self, tmp_path):
        (tmp_path / "skip_me").mkdir()
        (tmp_path / "skip_me" / "ignored.py").write_text("x = 1", encoding="utf-8")

        (tmp_path / "keep_me").mkdir()
        (tmp_path / "keep_me" / "kept.py").write_text("x = 1", encoding="utf-8")

        scanner = RepositoryScanner(exclude_dirs={"skip_me"})
        discovered = scanner.discover_files(tmp_path)

        assert len(discovered) == 1
        assert discovered[0].name == "kept.py"

    def test_max_file_size_bytes(self, tmp_path):
        normal_file = tmp_path / "normal.py"
        normal_file.write_text("x = 1\n", encoding="utf-8")

        large_file = tmp_path / "large.py"
        large_file.write_text("x = 1\n" * 500, encoding="utf-8")

        scanner = RepositoryScanner(max_file_size_bytes=50)
        discovered = scanner.discover_files(tmp_path)

        assert len(discovered) == 1
        assert discovered[0].name == "normal.py"

    def test_nonexistent_path_raises_scan_error(self, scanner):
        with pytest.raises(ScanError) as exc_info:
            scanner.discover_files("nonexistent_path_xyz_123")

        assert "does not exist" in exc_info.value.message
