"""
Runtime configuration contract for the SemDrift CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CLIConfig:
    """Immutable runtime configuration resolved from CLI options and arguments.

    Attributes:
        path: Target directory or file path to scan.
        checkpoint: Path to the PyTorch model checkpoint (.pt).
        format: Desired output format ('terminal', 'markdown', or 'json').
        threshold: Decision threshold for semantic drift detection (0.0 to 1.0).
        batch_size: Number of code-doc pairs to process per model inference batch.
        device: Target execution device ('auto', 'cpu', or 'cuda').
    """

    path: Path
    checkpoint: Path
    format: str = "terminal"
    threshold: float = 0.50
    batch_size: int = 16
    device: str = "auto"
