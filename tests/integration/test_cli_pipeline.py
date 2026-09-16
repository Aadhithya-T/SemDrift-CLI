"""
Integration tests for the SemDrift end-to-end CLI workflow.

Verifies the complete pipeline:
Scanner -> Parser -> Model -> Detection -> Reporting -> stdout/stderr
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import pytest

from semdrift.cli.main import main
from semdrift.model.inference import SemDriftModel
from semdrift.model.models import ModelPrediction
from semdrift.parser.models import CodeDocumentPair


@pytest.fixture
def fixture_repo(tmp_path: Path) -> Path:
    """Create a sample Python repository with documented and undocumented functions."""
    repo = tmp_path / "sample_project"
    repo.mkdir()

    # File 1: auth.py
    auth_file = repo / "auth.py"
    auth_file.write_text(
        'def login(user, password):\n    """Authenticate user."""\n    return True\n\n'
        "def _hash_token(token):\n    return hash(token)\n",
        encoding="utf-8",
    )

    # File 2: calc.py
    calc_file = repo / "calc.py"
    calc_file.write_text(
        'def add(a, b):\n    """Return sum of a and b."""\n    return a + b\n',
        encoding="utf-8",
    )

    return repo


@pytest.fixture
def mock_model(monkeypatch):
    """Inject a deterministic mock model for offline integration testing."""
    class MockInferenceModel:
        def __init__(self, checkpoint_path, device="cpu"):
            self.checkpoint_path = checkpoint_path
            self.device = device
            self.last_batch_size = None

        def predict(self, pairs: list[CodeDocumentPair], batch_size: int | None = None) -> list[ModelPrediction]:
            self.last_batch_size = batch_size
            preds = []
            for pair in pairs:
                if pair.docstring is None:
                    preds.append(
                        ModelPrediction(
                            file_path=pair.file_path,
                            qualified_name=pair.qualified_name,
                            line_number=pair.line_number,
                            drift_probability=None,
                            prediction="undocumented",
                            is_undocumented=True,
                        )
                    )
                elif pair.qualified_name == "login":
                    # Simulated high drift probability
                    preds.append(
                        ModelPrediction(
                            file_path=pair.file_path,
                            qualified_name=pair.qualified_name,
                            line_number=pair.line_number,
                            drift_probability=0.85,
                            prediction="drifted",
                            is_undocumented=False,
                        )
                    )
                else:
                    # Simulated aligned probability
                    preds.append(
                        ModelPrediction(
                            file_path=pair.file_path,
                            qualified_name=pair.qualified_name,
                            line_number=pair.line_number,
                            drift_probability=0.15,
                            prediction="aligned",
                            is_undocumented=False,
                        )
                    )
            return preds


    active_instances = []

    def mock_from_checkpoint(checkpoint_path, config=None, device=None, **kwargs):
        # Validate that checkpoint exists on disk if tested
        p = Path(checkpoint_path)
        if not p.exists():
            from semdrift.model.exceptions import ModelLoadError
            raise ModelLoadError(
                message=f"Checkpoint file not found: '{checkpoint_path}'",
                checkpoint_path=str(checkpoint_path),
            )
        instance = MockInferenceModel(checkpoint_path=checkpoint_path, device=device or "cpu")
        active_instances.append(instance)
        return instance

    monkeypatch.setattr(SemDriftModel, "from_checkpoint", mock_from_checkpoint)
    return active_instances


@pytest.fixture
def dummy_checkpoint(tmp_path: Path) -> Path:
    ckpt = tmp_path / "model_weights.pt"
    ckpt.write_text("mock checkpoint content", encoding="utf-8")
    return ckpt


class TestEndToEndPipeline:
    """Verify full end-to-end scanning, inference, detection, and reporting."""

    def test_terminal_report_output(self, fixture_repo, dummy_checkpoint, mock_model):
        stdout = io.StringIO()
        stderr = io.StringIO()
        exit_code = main(
            ["scan", str(fixture_repo), "--checkpoint", str(dummy_checkpoint)],
            stdout=stdout,
            stderr=stderr,
        )

        assert exit_code == 0
        assert stderr.getvalue() == ""
        out = stdout.getvalue()
        assert "SemDrift Detection Summary" in out
        assert "Total:        3" in out
        assert "Drifted:      1" in out
        assert "Aligned:      1" in out
        assert "Undocumented: 1" in out
        assert "[drifted]" in out
        assert "login" in out
        assert "[undocumented]" in out
        assert "_hash_token" in out

    def test_markdown_report_output(self, fixture_repo, dummy_checkpoint, mock_model):
        stdout = io.StringIO()
        stderr = io.StringIO()
        exit_code = main(
            [
                "scan",
                str(fixture_repo),
                "--checkpoint",
                str(dummy_checkpoint),
                "--format",
                "markdown",
            ],
            stdout=stdout,
            stderr=stderr,
        )

        assert exit_code == 0
        assert stderr.getvalue() == ""
        out = stdout.getvalue()
        assert "# SemDrift Report" in out
        assert "## Summary" in out
        assert "| Total | 3 |" in out
        assert "| Drifted | 1 |" in out
        assert "| Aligned | 1 |" in out
        assert "| Undocumented | 1 |" in out
        assert "## Results" in out

    def test_json_report_output_cleanliness(self, fixture_repo, dummy_checkpoint, mock_model):
        stdout = io.StringIO()
        stderr = io.StringIO()
        exit_code = main(
            [
                "scan",
                str(fixture_repo),
                "--checkpoint",
                str(dummy_checkpoint),
                "--format",
                "json",
            ],
            stdout=stdout,
            stderr=stderr,
        )

        assert exit_code == 0
        assert stderr.getvalue() == ""

        # Verify stdout is 100% clean, parseable JSON with no debug text or banners
        raw_stdout = stdout.getvalue()
        parsed = json.loads(raw_stdout)
        assert isinstance(parsed, list)
        assert len(parsed) == 3

        # Verify undocumented null representation
        undoc = next(p for p in parsed if p["qualified_name"] == "_hash_token")
        assert undoc["status"] == "undocumented"
        assert undoc["drift_probability"] is None
        assert '"drift_probability": null' in raw_stdout

        # Verify drifted representation
        drifted = next(p for p in parsed if p["qualified_name"] == "login")
        assert drifted["status"] == "drifted"
        assert drifted["drift_probability"] == 0.85

    def test_threshold_propagation(self, fixture_repo, dummy_checkpoint, mock_model):
        """When threshold is raised to 0.90, login (prob 0.85) should now be aligned."""
        stdout = io.StringIO()
        stderr = io.StringIO()
        exit_code = main(
            [
                "scan",
                str(fixture_repo),
                "--checkpoint",
                str(dummy_checkpoint),
                "--threshold",
                "0.90",
                "--format",
                "json",
            ],
            stdout=stdout,
            stderr=stderr,
        )

        assert exit_code == 0
        parsed = json.loads(stdout.getvalue())
        login_res = next(p for p in parsed if p["qualified_name"] == "login")
        assert login_res["status"] == "aligned"
        assert login_res["is_drift"] is False
        assert login_res["threshold"] == 0.90

    def test_batch_size_propagation(self, fixture_repo, dummy_checkpoint, mock_model):
        stdout = io.StringIO()
        exit_code = main(
            [
                "scan",
                str(fixture_repo),
                "--checkpoint",
                str(dummy_checkpoint),
                "--batch-size",
                "32",
            ],
            stdout=stdout,
        )
        assert exit_code == 0
        assert mock_model[-1].last_batch_size == 32

    def test_device_propagation_cpu(self, fixture_repo, dummy_checkpoint, mock_model):
        stdout = io.StringIO()
        exit_code = main(
            [
                "scan",
                str(fixture_repo),
                "--checkpoint",
                str(dummy_checkpoint),
                "--device",
                "cpu",
            ],
            stdout=stdout,
        )
        assert exit_code == 0
        assert mock_model[-1].device == "cpu"

    def test_empty_repository_scan(self, tmp_path: Path, dummy_checkpoint, mock_model):
        empty_dir = tmp_path / "empty_repo"
        empty_dir.mkdir()

        stdout = io.StringIO()
        stderr = io.StringIO()
        exit_code = main(
            [
                "scan",
                str(empty_dir),
                "--checkpoint",
                str(dummy_checkpoint),
                "--format",
                "json",
            ],
            stdout=stdout,
            stderr=stderr,
        )

        assert exit_code == 0
        assert stderr.getvalue() == ""
        parsed = json.loads(stdout.getvalue())
        assert parsed == []

    def test_syntax_error_file_warns_and_continues(self, fixture_repo, dummy_checkpoint, mock_model):
        # Create a malformed python file
        bad_file = fixture_repo / "broken_syntax.py"
        bad_file.write_text("def broken( :::\n", encoding="utf-8")

        stdout = io.StringIO()
        stderr = io.StringIO()
        exit_code = main(
            [
                "scan",
                str(fixture_repo),
                "--checkpoint",
                str(dummy_checkpoint),
                "--format",
                "json",
            ],
            stdout=stdout,
            stderr=stderr,
        )

        assert exit_code == 0
        # Check that warning appeared on stderr
        err_text = stderr.getvalue()
        assert "Warning: Failed to parse" in err_text
        assert "broken_syntax.py" in err_text

        # Check that valid files were still processed on stdout
        parsed = json.loads(stdout.getvalue())
        assert len(parsed) == 3

    def test_nonexistent_scan_path_returns_exit_code_1(self, dummy_checkpoint, mock_model):
        stdout = io.StringIO()
        stderr = io.StringIO()
        exit_code = main(
            ["scan", "nonexistent_directory_12345", "--checkpoint", str(dummy_checkpoint)],
            stdout=stdout,
            stderr=stderr,
        )
        assert exit_code == 1
        assert stdout.getvalue() == ""
        assert "Error:" in stderr.getvalue()

    def test_nonexistent_checkpoint_returns_exit_code_1(self, fixture_repo, mock_model):
        stdout = io.StringIO()
        stderr = io.StringIO()
        exit_code = main(
            ["scan", str(fixture_repo), "--checkpoint", "missing_checkpoint.pt"],
            stdout=stdout,
            stderr=stderr,
        )
        assert exit_code == 1
        assert stdout.getvalue() == ""
        assert "Error:" in stderr.getvalue()
        assert "Checkpoint file not found" in stderr.getvalue()
