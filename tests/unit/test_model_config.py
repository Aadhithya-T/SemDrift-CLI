"""
Unit tests for semdrift.model.config, semdrift.model.models, and semdrift.model.exceptions.
"""

from dataclasses import FrozenInstanceError
import pytest

from semdrift.model.config import ModelConfig
from semdrift.model.exceptions import (
    ModelError,
    ModelInferenceError,
    ModelLoadError,
)
from semdrift.model.models import ModelPrediction


class TestModelConfig:
    """Verify ModelConfig default values and immutability."""

    def test_default_values(self):
        cfg = ModelConfig()
        assert cfg.model_name == "microsoft/codebert-base"
        assert cfg.max_length == 512
        assert cfg.doc_max_tokens == 96
        assert cfg.code_truncation == "head_tail"
        assert cfg.pooling == "cls"
        assert cfg.num_labels == 2
        assert cfg.dropout == 0.1
        assert cfg.device == "cpu"
        assert cfg.batch_size == 16
        assert cfg.clean_docstrings is True

    def test_immutability(self):
        cfg = ModelConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.device = "cuda"  # type: ignore


class TestModelPrediction:
    """Verify ModelPrediction contract and immutability."""

    def test_documented_prediction(self):
        pred = ModelPrediction(
            file_path="src/calc.py",
            qualified_name="Calculator.add",
            line_number=10,
            drift_probability=0.8421,
            prediction="drifted",
            is_undocumented=False,
            raw_logits=(-1.2, 1.8),
        )
        assert pred.file_path == "src/calc.py"
        assert pred.qualified_name == "Calculator.add"
        assert pred.line_number == 10
        assert pred.drift_probability == 0.8421
        assert pred.prediction == "drifted"
        assert pred.is_undocumented is False
        assert pred.raw_logits == (-1.2, 1.8)

    def test_undocumented_prediction(self):
        pred = ModelPrediction(
            file_path="src/calc.py",
            qualified_name="Calculator.undocumented_helper",
            line_number=25,
            drift_probability=None,
            prediction="undocumented",
            is_undocumented=True,
            raw_logits=None,
        )
        assert pred.drift_probability is None
        assert pred.prediction == "undocumented"
        assert pred.is_undocumented is True
        assert pred.raw_logits is None

    def test_immutability(self):
        pred = ModelPrediction(
            file_path="a.py",
            qualified_name="foo",
            line_number=1,
            drift_probability=0.1,
            prediction="aligned",
        )
        with pytest.raises(FrozenInstanceError):
            pred.prediction = "drifted"  # type: ignore


class TestModelExceptions:
    """Verify model exception hierarchy and formatting."""

    def test_model_error(self):
        cause = ValueError("bad value")
        err = ModelError("something went wrong", cause=cause)
        assert "something went wrong" in str(err)
        assert "ValueError" in str(err)

    def test_model_load_error(self):
        err = ModelLoadError("file corrupt", checkpoint_path="model.pt")
        assert "model.pt" in str(err)
        assert "file corrupt" in str(err)

    def test_model_inference_error(self):
        err = ModelInferenceError("CUDA OOM", batch_size=32)
        assert "CUDA OOM" in str(err)
        assert "batch_size=32" in str(err)
