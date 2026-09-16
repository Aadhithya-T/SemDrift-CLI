"""
Unit tests for semdrift.cli argument parsing, validation, and error handling.
"""

from __future__ import annotations

import io
from pathlib import Path
import pytest
import torch

from semdrift.cli.config import CLIConfig
from semdrift.cli.errors import CLIArgumentError, CLIRuntimeError
from semdrift.cli.main import main, resolve_device
from semdrift.cli.parser import build_parser, parse_args


class TestCLIParser:
    """Verify CLI argument parsing and validation contracts."""

    def test_default_options(self):
        config = parse_args(["scan", "my_repo", "--checkpoint", "model.pt"])
        assert config.path == Path("my_repo")
        assert config.checkpoint == Path("model.pt")
        assert config.format == "terminal"
        assert config.threshold == 0.50
        assert config.batch_size == 16
        assert config.device == "auto"

    def test_custom_options(self):
        config = parse_args(
            [
                "scan",
                "src/pkg",
                "--checkpoint",
                "weights/ckpt.pt",
                "--format",
                "json",
                "--threshold",
                "0.85",
                "--batch-size",
                "32",
                "--device",
                "cpu",
            ]
        )
        assert config.path == Path("src/pkg")
        assert config.checkpoint == Path("weights/ckpt.pt")
        assert config.format == "json"
        assert config.threshold == 0.85
        assert config.batch_size == 32
        assert config.device == "cpu"

    def test_missing_command_raises(self):
        with pytest.raises(CLIArgumentError) as exc_info:
            parse_args([])
        assert exc_info.value.exit_code == 2

    def test_missing_checkpoint_raises(self):
        with pytest.raises(CLIArgumentError) as exc_info:
            parse_args(["scan", "my_repo"])
        assert exc_info.value.exit_code == 2
        assert "checkpoint" in str(exc_info.value)

    def test_empty_checkpoint_string_raises(self):
        with pytest.raises(CLIArgumentError) as exc_info:
            parse_args(["scan", "my_repo", "--checkpoint", "   "])
        assert exc_info.value.exit_code == 2
        assert "checkpoint" in str(exc_info.value)

    def test_invalid_batch_size_zero_or_negative(self):
        with pytest.raises(CLIArgumentError) as exc_info:
            parse_args(["scan", "repo", "--checkpoint", "m.pt", "--batch-size", "0"])
        assert exc_info.value.exit_code == 2
        assert "--batch-size must be a positive integer" in str(exc_info.value)

        with pytest.raises(CLIArgumentError) as exc_info:
            parse_args(["scan", "repo", "--checkpoint", "m.pt", "--batch-size", "-4"])
        assert exc_info.value.exit_code == 2
        assert "--batch-size must be a positive integer" in str(exc_info.value)

    def test_invalid_threshold_bounds(self):
        with pytest.raises(CLIArgumentError) as exc_info:
            parse_args(["scan", "repo", "--checkpoint", "m.pt", "--threshold", "-0.01"])
        assert exc_info.value.exit_code == 2
        assert "threshold must be within [0.0, 1.0]" in str(exc_info.value)

        with pytest.raises(CLIArgumentError) as exc_info:
            parse_args(["scan", "repo", "--checkpoint", "m.pt", "--threshold", "1.01"])
        assert exc_info.value.exit_code == 2
        assert "threshold must be within [0.0, 1.0]" in str(exc_info.value)

    def test_invalid_format_choice(self):
        with pytest.raises(CLIArgumentError) as exc_info:
            parse_args(["scan", "repo", "--checkpoint", "m.pt", "--format", "xml"])
        assert exc_info.value.exit_code == 2

    def test_invalid_device_choice(self):
        with pytest.raises(CLIArgumentError) as exc_info:
            parse_args(["scan", "repo", "--checkpoint", "m.pt", "--device", "tpu"])
        assert exc_info.value.exit_code == 2


class TestDeviceResolution:
    """Verify device selection and CUDA availability validation."""

    def test_resolve_cpu(self):
        assert resolve_device("cpu") == "cpu"
        assert resolve_device("CPU") == "cpu"

    def test_resolve_auto(self):
        resolved = resolve_device("auto")
        if torch.cuda.is_available():
            assert resolved == "cuda"
        else:
            assert resolved == "cpu"

    def test_resolve_cuda_explicit(self, monkeypatch):
        if torch.cuda.is_available():
            assert resolve_device("cuda") == "cuda"
        else:
            with pytest.raises(CLIRuntimeError) as exc_info:
                resolve_device("cuda")
            assert "CUDA was requested" in str(exc_info.value)


class TestMainEntrypointBehaviors:
    """Verify main() return codes, stream isolation, and exception propagation."""

    def test_help_flag_returns_zero(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        exit_code = main(["--help"], stdout=stdout, stderr=stderr)
        assert exit_code == 0

    def test_scan_help_flag_returns_zero(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        exit_code = main(["scan", "--help"], stdout=stdout, stderr=stderr)
        assert exit_code == 0

    def test_argument_error_returns_two(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        exit_code = main(["scan"], stdout=stdout, stderr=stderr)
        assert exit_code == 2
        assert stdout.getvalue() == ""
        assert "error:" in stderr.getvalue().lower() or "usage:" in stderr.getvalue().lower()

    def test_unexpected_exception_bubbles_up(self, monkeypatch):
        import sys
        mod = sys.modules["semdrift.cli.main"]

        def buggy_scan(*args, **kwargs):
            raise IndexError("Unexpected list index bug")

        monkeypatch.setattr(mod, "orchestrate_scan", buggy_scan)
        with pytest.raises(IndexError, match="Unexpected list index bug"):
            main(["scan", "src/", "--checkpoint", "m.pt"])


