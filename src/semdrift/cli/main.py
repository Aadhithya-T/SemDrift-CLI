"""
Main CLI entry point and end-to-end pipeline orchestrator for SemDrift.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional, Sequence, TextIO
import torch

from semdrift.cli.config import CLIConfig
from semdrift.cli.errors import CLIArgumentError, CLIError, CLIRuntimeError
from semdrift.cli.parser import parse_args
from semdrift.detection.config import DetectionConfig
from semdrift.detection.detector import DriftDetector
from semdrift.detection.exceptions import DetectionError
from semdrift.model.exceptions import ModelError
from semdrift.model.inference import SemDriftModel
from semdrift.parser.ast_parser import PythonASTParser
from semdrift.parser.models import CodeDocumentPair, ParseError
from semdrift.reporting.exceptions import ReportingError
from semdrift.reporting.json import JsonReporter
from semdrift.reporting.markdown import MarkdownReporter
from semdrift.reporting.terminal import TerminalReporter
from semdrift.scanner.repository import RepositoryScanner, ScanError


def resolve_device(device_str: str) -> str:
    """Resolve target execution device from CLI parameter.

    Args:
        device_str: Device identifier string ('auto', 'cpu', or 'cuda').

    Returns:
        Resolved device string ('cpu' or 'cuda').

    Raises:
        CLIRuntimeError: If CUDA is explicitly requested but unavailable.
        CLIArgumentError: If device string is unsupported.
    """
    dev = device_str.lower()
    if dev == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    elif dev == "cuda":
        if not torch.cuda.is_available():
            raise CLIRuntimeError(
                "CUDA was requested via --device cuda, but CUDA is not available on this system."
            )
        return "cuda"
    elif dev == "cpu":
        return "cpu"
    else:
        raise CLIArgumentError(f"error: unsupported device: '{device_str}'", exit_code=2)


def orchestrate_scan(
    config: CLIConfig,
    stdout: Optional[TextIO] = None,
    stderr: Optional[TextIO] = None,
) -> int:
    """Execute the end-to-end SemDrift scan workflow.

    Pipeline sequence:
        Scanner -> Parser -> Model -> Detection -> Reporting -> stdout

    Args:
        config: Validated runtime configuration.
        stdout: Output stream for rendered reports (defaults to sys.stdout).
        stderr: Error stream for warnings and diagnostics (defaults to sys.stderr).

    Returns:
        Process exit code (0 on successful scan completion).
    """
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr

    # 1. Scanner Layer
    scanner = RepositoryScanner()
    file_paths = scanner.discover_files(config.path)

    # 2. Parser Layer
    parser = PythonASTParser()
    all_pairs: list[CodeDocumentPair] = []
    for fp in file_paths:
        try:
            pairs = parser.parse_file(fp, relative_to=config.path)
            all_pairs.extend(pairs)
        except ParseError as exc:
            # Non-fatal: notify on stderr and continue processing remaining files
            err.write(f"Warning: Failed to parse '{fp}': {exc.message}\n")

    # 3. Model Layer
    target_device = resolve_device(config.device)
    model = SemDriftModel.from_checkpoint(
        checkpoint_path=config.checkpoint,
        device=target_device,
    )
    predictions = model.predict(all_pairs, batch_size=config.batch_size)

    # 4. Detection Layer
    det_config = DetectionConfig(drift_threshold=config.threshold)
    detector = DriftDetector(config=det_config)
    results = detector.detect(predictions)

    # 5. Reporting Layer
    if config.format == "json":
        reporter = JsonReporter()
    elif config.format == "markdown":
        reporter = MarkdownReporter()
    else:
        reporter = TerminalReporter()

    rendered = reporter.render(results)

    # 6. Output to stdout
    out.write(rendered if rendered.endswith("\n") else rendered + "\n")
    out.flush()
    return 0


def main(
    argv: Optional[Sequence[str]] = None,
    stdout: Optional[TextIO] = None,
    stderr: Optional[TextIO] = None,
) -> int:
    """CLI application entry point.

    Parses command-line arguments, catches expected application-layer errors,
    and returns deterministic process exit codes.

    Args:
        argv: Optional command-line arguments (defaults to sys.argv[1:]).
        stdout: Output stream for reports (defaults to sys.stdout).
        stderr: Error stream for diagnostics (defaults to sys.stderr).

    Returns:
        0 on success, 1 on runtime error, 2 on argument/configuration error.
    """
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr

    try:
        config = parse_args(argv)
        return orchestrate_scan(config, stdout=out, stderr=err)
    except CLIArgumentError as exc:
        if exc.exit_code == 0:
            return 0
        if exc.message:
            err.write(f"{exc.message}\n")
        return exc.exit_code
    except (
        CLIRuntimeError,
        ScanError,
        ModelError,
        DetectionError,
        ReportingError,
    ) as exc:
        err.write(f"Error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
