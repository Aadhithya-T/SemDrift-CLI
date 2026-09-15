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
]
