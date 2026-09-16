"""
semdrift.cli — Command-line interface and end-to-end pipeline orchestration.
"""

from semdrift.cli.config import CLIConfig
from semdrift.cli.errors import CLIArgumentError, CLIError, CLIRuntimeError
from semdrift.cli.main import main, orchestrate_scan
from semdrift.cli.parser import build_parser, parse_args

__all__ = [
    "CLIConfig",
    "CLIError",
    "CLIArgumentError",
    "CLIRuntimeError",
    "build_parser",
    "parse_args",
    "orchestrate_scan",
    "main",
]
