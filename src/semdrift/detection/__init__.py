"""
semdrift.detection — Semantic drift detection and policy evaluation layer.

Provides configurable threshold-based classification of ModelPrediction records
into structured DriftResult outcomes.
"""

from semdrift.detection.config import DetectionConfig
from semdrift.detection.models import DriftResult
from semdrift.detection.detector import DriftDetector
from semdrift.detection.exceptions import (
    DetectionError,
    DetectionConfigError,
    DetectionInputError,
)

__all__ = [
    "DetectionConfig",
    "DriftResult",
    "DriftDetector",
    "DetectionError",
    "DetectionConfigError",
    "DetectionInputError",
]
