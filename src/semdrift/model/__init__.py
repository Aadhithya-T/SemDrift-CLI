"""
semdrift.model — Model architecture, preprocessing, and inference layer.

Provides CodeBERT Joint-Encoder inference for semantic drift detection.
"""

from semdrift.model.architecture import JointEncoderModel
from semdrift.model.config import ModelConfig
from semdrift.model.exceptions import (
    ModelError,
    ModelInferenceError,
    ModelLoadError,
)
from semdrift.model.inference import SemDriftModel
from semdrift.model.manager import ModelManager
from semdrift.model.models import ModelPrediction
from semdrift.model.preprocessing import (
    collate_joint_batch,
    extract_docstring_summary,
    prepare_joint_tokens,
)

__all__ = [
    "ModelConfig",
    "ModelPrediction",
    "JointEncoderModel",
    "SemDriftModel",
    "ModelManager",
    "ModelError",
    "ModelLoadError",
    "ModelInferenceError",
    "extract_docstring_summary",
    "prepare_joint_tokens",
    "collate_joint_batch",
]
