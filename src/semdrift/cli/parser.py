"""
Command-line argument parser for SemDrift.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Optional, Sequence

from semdrift.cli.config import CLIConfig
from semdrift.cli.errors import CLIArgumentError


class CLIArgumentParser(argparse.ArgumentParser):
    """Custom ArgumentParser that raises CLIArgumentError instead of sys.exit()."""

    def exit(self, status: int = 0, message: Optional[str] = None) -> None:
        if message:
            self._print_message(message, sys.stderr)
        raise CLIArgumentError(message=message or "", exit_code=status)

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        raise CLIArgumentError(message=f"error: {message}", exit_code=2)


def build_parser() -> CLIArgumentParser:
    """Construct the SemDrift command-line parser hierarchy.

    Returns:
        Configured CLIArgumentParser instance.
    """
    parser = CLIArgumentParser(
        prog="semdrift",
        description="Detect semantic drift between source code and documentation in Python repositories.",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Available commands",
        required=True,
    )

    # scan command
    scan_parser = subparsers.add_parser(
        "scan",
        help="Scan a repository or Python file for semantic drift.",
        description="Scan Python files, parse definitions, evaluate drift using CodeBERT, and report results.",
    )
    scan_parser.add_argument(
        "path",
        type=str,
        help="Path to repository directory or single Python file to scan.",
    )
    scan_parser.add_argument(
        "--format",
        choices=["terminal", "markdown", "json"],
        default="terminal",
        help="Output report format (default: terminal).",
    )
    scan_parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Decision threshold for drift detection in [0.0, 1.0] (default: 0.50).",
    )
    scan_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained PyTorch model checkpoint (.pt).",
    )
    scan_parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        dest="batch_size",
        help="Inference batch size (default: 16).",
    )
    scan_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device: auto (CUDA if available, else CPU), cpu, or cuda (default: auto).",
    )

    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> CLIConfig:
    """Parse and validate command-line arguments into a CLIConfig.

    Args:
        argv: Optional list of command-line argument strings. Defaults to sys.argv[1:].

    Returns:
        Validated immutable CLIConfig instance.

    Raises:
        CLIArgumentError: If arguments are invalid, missing, or help is requested.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Validate batch_size
    if args.batch_size <= 0:
        raise CLIArgumentError(
            f"error: --batch-size must be a positive integer, got {args.batch_size}",
            exit_code=2,
        )

    # Validate threshold
    if math.isnan(args.threshold) or math.isinf(args.threshold) or not (0.0 <= args.threshold <= 1.0):
        raise CLIArgumentError(
            f"error: --threshold must be within [0.0, 1.0], got {args.threshold}",
            exit_code=2,
        )

    # Validate checkpoint string
    checkpoint_str = str(args.checkpoint).strip()
    if not checkpoint_str:
        raise CLIArgumentError("error: --checkpoint path must not be empty", exit_code=2)

    return CLIConfig(
        path=Path(args.path),
        checkpoint=Path(checkpoint_str),
        format=args.format,
        threshold=float(args.threshold),
        batch_size=int(args.batch_size),
        device=args.device,
    )
