"""
SemDrift — Semantic Drift Detection for Python Repositories.

This package provides the production implementation of SemDrift,
a tool for detecting semantic drift between source code and
documentation in Python codebases.

The research, datasets, training, and benchmarking material live in
the original research repository:
https://github.com/Aadhithya-T/SemDrift
"""

from semdrift.parser.models import CodeDocumentPair, ParseError
from semdrift.parser.ast_parser import PythonASTParser
from semdrift.scanner.repository import RepositoryScanner, ScanError
from semdrift.model.config import ModelConfig
from semdrift.model.models import ModelPrediction
from semdrift.model.inference import SemDriftModel
from semdrift.model.exceptions import (
    ModelError,
    ModelLoadError,
    ModelInferenceError,
)
from semdrift.detection.config import DetectionConfig
from semdrift.detection.models import DriftResult
from semdrift.detection.detector import DriftDetector
from semdrift.detection.exceptions import (
    DetectionError,
    DetectionConfigError,
    DetectionInputError,
)
from semdrift.reporting.json import JsonReporter
from semdrift.reporting.markdown import MarkdownReporter
from semdrift.reporting.terminal import TerminalReporter
from semdrift.reporting.exceptions import (
    ReportingError,
    ReportingInputError,
)
from semdrift.cli.main import main
from semdrift.cli.config import CLIConfig
from semdrift.cli.errors import (
    CLIError,
    CLIArgumentError,
    CLIRuntimeError,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "CodeDocumentPair",
    "ParseError",
    "PythonASTParser",
    "RepositoryScanner",
    "ScanError",
    "ModelConfig",
    "ModelPrediction",
    "SemDriftModel",
    "ModelError",
    "ModelLoadError",
    "ModelInferenceError",
    "DetectionConfig",
    "DriftResult",
    "DriftDetector",
    "DetectionError",
    "DetectionConfigError",
    "DetectionInputError",
    "JsonReporter",
    "MarkdownReporter",
    "TerminalReporter",
    "ReportingError",
    "ReportingInputError",
    "main",
    "CLIConfig",
    "CLIError",
    "CLIArgumentError",
    "CLIRuntimeError",
]


