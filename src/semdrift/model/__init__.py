"""
semdrift.model — Model architecture, preprocessing, and inference layer.

Provides CodeBERT Joint-Encoder inference for semantic drift detection.
"""

from semdrift.model.config import ModelConfig
from semdrift.model.models import ModelPrediction
from semdrift.model.architecture import JointEncoderModel
from semdrift.model.inference import SemDriftModel
from semdrift.model.exceptions import (
    ModelError,
    ModelLoadError,
    ModelInferenceError,
)
from semdrift.model.preprocessing import (
    extract_docstring_summary,
    prepare_joint_tokens,
    collate_joint_batch,
)

__all__ = [
    "ModelConfig",
    "ModelPrediction",
    "JointEncoderModel",
    "SemDriftModel",
    "ModelError",
    "ModelLoadError",
    "ModelInferenceError",
    "extract_docstring_summary",
    "prepare_joint_tokens",
    "collate_joint_batch",
]
