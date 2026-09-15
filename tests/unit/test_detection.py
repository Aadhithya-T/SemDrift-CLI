"""
Unit tests for semdrift.detection.
"""

from dataclasses import FrozenInstanceError
import pytest

from semdrift.detection.config import DetectionConfig
from semdrift.detection.detector import DriftDetector
from semdrift.detection.exceptions import (
    DetectionConfigError,
    DetectionError,
    DetectionInputError,
)
from semdrift.detection.models import DriftResult
from semdrift.model.models import ModelPrediction


class TestDetectionConfig:
    """Verify DetectionConfig validation and immutability."""

    def test_default_threshold(self):
        cfg = DetectionConfig()
        assert cfg.drift_threshold == 0.5

    def test_valid_custom_threshold(self):
        cfg = DetectionConfig(drift_threshold=0.75)
        assert cfg.drift_threshold == 0.75

    def test_threshold_integer_converted_to_float(self):
        cfg0 = DetectionConfig(drift_threshold=0)
        assert cfg0.drift_threshold == 0.0
        assert isinstance(cfg0.drift_threshold, float)

        cfg1 = DetectionConfig(drift_threshold=1)
        assert cfg1.drift_threshold == 1.0
        assert isinstance(cfg1.drift_threshold, float)

    def test_threshold_below_zero_raises(self):
        with pytest.raises(DetectionConfigError) as exc_info:
            DetectionConfig(drift_threshold=-0.01)
        assert "within [0.0, 1.0]" in str(exc_info.value)

    def test_threshold_above_one_raises(self):
        with pytest.raises(DetectionConfigError) as exc_info:
            DetectionConfig(drift_threshold=1.0001)
        assert "within [0.0, 1.0]" in str(exc_info.value)

    def test_boolean_threshold_rejected(self):
        with pytest.raises(DetectionConfigError) as exc_info:
            DetectionConfig(drift_threshold=True)  # type: ignore
        assert "must be a numeric float or int" in str(exc_info.value)

        with pytest.raises(DetectionConfigError) as exc_info:
            DetectionConfig(drift_threshold=False)  # type: ignore
        assert "must be a numeric float or int" in str(exc_info.value)

    def test_non_numeric_threshold_rejected(self):
        with pytest.raises(DetectionConfigError):
            DetectionConfig(drift_threshold="0.5")  # type: ignore

    def test_nan_or_inf_threshold_rejected(self):
        with pytest.raises(DetectionConfigError):
            DetectionConfig(drift_threshold=float("nan"))
        with pytest.raises(DetectionConfigError):
            DetectionConfig(drift_threshold=float("inf"))

    def test_immutability(self):
        cfg = DetectionConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.drift_threshold = 0.8  # type: ignore


class TestDriftResult:
    """Verify DriftResult contract and immutability."""

    def test_drift_result_fields(self):
        res = DriftResult(
            file_path="src/app.py",
            qualified_name="login",
            line_number=12,
            status="drifted",
            is_drift=True,
            drift_probability=0.88,
            threshold=0.5,
        )
        assert res.file_path == "src/app.py"
        assert res.qualified_name == "login"
        assert res.line_number == 12
        assert res.status == "drifted"
        assert res.is_drift is True
        assert res.drift_probability == 0.88
        assert res.threshold == 0.5
        assert res.is_undocumented is False

    def test_undocumented_result(self):
        res = DriftResult(
            file_path="src/app.py",
            qualified_name="helper",
            line_number=20,
            status="undocumented",
            is_drift=False,
            drift_probability=None,
            threshold=0.5,
        )
        assert res.status == "undocumented"
        assert res.is_drift is False
        assert res.drift_probability is None
        assert res.is_undocumented is True

    def test_immutability(self):
        res = DriftResult(
            file_path="a.py",
            qualified_name="f",
            line_number=1,
            status="aligned",
            is_drift=False,
            drift_probability=0.1,
            threshold=0.5,
        )
        with pytest.raises(FrozenInstanceError):
            res.status = "drifted"  # type: ignore


class TestDriftDetectorLogic:
    """Verify threshold evaluation, boundaries, ordering, and error handling."""

    @pytest.fixture
    def detector_default(self):
        return DriftDetector()  # threshold = 0.5

    # ------------------------------------------------------------------
    # Threshold Boundary Tests
    # ------------------------------------------------------------------

    def test_threshold_below_boundary(self, detector_default):
        pred = ModelPrediction(
            file_path="calc.py",
            qualified_name="sub",
            line_number=5,
            drift_probability=0.4999,
            prediction="aligned",
            is_undocumented=False,
        )
        results = detector_default.detect([pred])
        assert len(results) == 1
        assert results[0].status == "aligned"
        assert results[0].is_drift is False
        assert results[0].drift_probability == 0.4999
        assert results[0].threshold == 0.5

    def test_threshold_exactly_at_boundary(self, detector_default):
        pred = ModelPrediction(
            file_path="calc.py",
            qualified_name="add",
            line_number=1,
            drift_probability=0.50,
            prediction="drifted",
            is_undocumented=False,
        )
        results = detector_default.detect([pred])
        assert len(results) == 1
        # Exactly equal to threshold evaluates to drift
        assert results[0].status == "drifted"
        assert results[0].is_drift is True
        assert results[0].drift_probability == 0.50

    def test_threshold_above_boundary(self, detector_default):
        pred = ModelPrediction(
            file_path="calc.py",
            qualified_name="mul",
            line_number=10,
            drift_probability=0.5001,
            prediction="drifted",
            is_undocumented=False,
        )
        results = detector_default.detect([pred])
        assert len(results) == 1
        assert results[0].status == "drifted"
        assert results[0].is_drift is True
        assert results[0].drift_probability == 0.5001

    def test_configurable_threshold_changes_decision(self):
        pred = ModelPrediction(
            file_path="calc.py",
            qualified_name="compute",
            line_number=1,
            drift_probability=0.65,
            prediction="drifted",
            is_undocumented=False,
        )

        detector_low = DriftDetector(DetectionConfig(drift_threshold=0.60))
        detector_high = DriftDetector(DetectionConfig(drift_threshold=0.70))

        result_low = detector_low.detect([pred])[0]
        result_high = detector_high.detect([pred])[0]

        # 0.65 >= 0.60 -> drifted
        assert result_low.status == "drifted"
        assert result_low.is_drift is True
        assert result_low.threshold == 0.60

        # 0.65 < 0.70 -> aligned
        assert result_high.status == "aligned"
        assert result_high.is_drift is False
        assert result_high.threshold == 0.70

    # ------------------------------------------------------------------
    # Undocumented Functions Handling
    # ------------------------------------------------------------------

    def test_undocumented_function_remains_undocumented(self, detector_default):
        pred = ModelPrediction(
            file_path="utils.py",
            qualified_name="undoc_helper",
            line_number=42,
            drift_probability=None,
            prediction="undocumented",
            is_undocumented=True,
        )
        results = detector_default.detect([pred])
        assert len(results) == 1
        r = results[0]
        assert r.status == "undocumented"
        assert r.is_drift is False
        assert r.drift_probability is None
        assert r.is_undocumented is True
        assert r.threshold == 0.5

    def test_undocumented_function_never_treated_as_drift(self):
        # Even with a zero threshold (where any non-negative float would drift)
        detector_zero = DriftDetector(DetectionConfig(drift_threshold=0.0))
        pred = ModelPrediction(
            file_path="utils.py",
            qualified_name="undoc_helper",
            line_number=42,
            drift_probability=None,
            prediction="undocumented",
            is_undocumented=True,
        )
        results = detector_zero.detect([pred])
        assert len(results) == 1
        assert results[0].status == "undocumented"
        assert results[0].is_drift is False

    # ------------------------------------------------------------------
    # Input Ordering & Metadata Preservation
    # ------------------------------------------------------------------

    def test_empty_input_returns_empty_list(self, detector_default):
        assert detector_default.detect([]) == []

    def test_multiple_predictions_all_converted_preserving_order(self, detector_default):
        preds = [
            ModelPrediction(
                file_path="mod_a.py",
                qualified_name="func_a",
                line_number=1,
                drift_probability=0.90,
                prediction="drifted",
                is_undocumented=False,
            ),
            ModelPrediction(
                file_path="mod_b.py",
                qualified_name="func_b",
                line_number=10,
                drift_probability=None,
                prediction="undocumented",
                is_undocumented=True,
            ),
            ModelPrediction(
                file_path="mod_c.py",
                qualified_name="func_c",
                line_number=20,
                drift_probability=0.10,
                prediction="aligned",
                is_undocumented=False,
            ),
        ]

        results = detector_default.detect(preds)
        assert len(results) == 3

        # Exactly matches input indices and metadata
        assert results[0].file_path == "mod_a.py"
        assert results[0].qualified_name == "func_a"
        assert results[0].line_number == 1
        assert results[0].status == "drifted"
        assert results[0].is_drift is True

        assert results[1].file_path == "mod_b.py"
        assert results[1].qualified_name == "func_b"
        assert results[1].line_number == 10
        assert results[1].status == "undocumented"
        assert results[1].is_drift is False

        assert results[2].file_path == "mod_c.py"
        assert results[2].qualified_name == "func_c"
        assert results[2].line_number == 20
        assert results[2].status == "aligned"
        assert results[2].is_drift is False

    def test_deterministic_repeated_detection(self, detector_default):
        preds = [
            ModelPrediction(
                file_path="mod.py",
                qualified_name="fn1",
                line_number=1,
                drift_probability=0.72,
                prediction="drifted",
                is_undocumented=False,
            ),
            ModelPrediction(
                file_path="mod.py",
                qualified_name="fn2",
                line_number=15,
                drift_probability=None,
                prediction="undocumented",
                is_undocumented=True,
            ),
        ]

        out1 = detector_default.detect(preds)
        out2 = detector_default.detect(preds)
        assert out1 == out2

    # ------------------------------------------------------------------
    # Invalid Prediction Input Tests
    # ------------------------------------------------------------------

    def test_documented_prediction_with_none_probability_raises(self, detector_default):
        pred = ModelPrediction(
            file_path="mod.py",
            qualified_name="fn",
            line_number=1,
            drift_probability=None,  # Invalid when is_undocumented=False
            prediction="drifted",
            is_undocumented=False,
        )
        with pytest.raises(DetectionInputError) as exc_info:
            detector_default.detect([pred])
        assert "cannot be None for a documented function" in str(exc_info.value)
        assert "fn" in str(exc_info.value)

    def test_probability_below_zero_raises(self, detector_default):
        pred = ModelPrediction(
            file_path="mod.py",
            qualified_name="fn",
            line_number=1,
            drift_probability=-0.05,
            prediction="aligned",
            is_undocumented=False,
        )
        with pytest.raises(DetectionInputError) as exc_info:
            detector_default.detect([pred])
        assert "within [0.0, 1.0]" in str(exc_info.value)

    def test_probability_above_one_raises(self, detector_default):
        pred = ModelPrediction(
            file_path="mod.py",
            qualified_name="fn",
            line_number=1,
            drift_probability=1.05,
            prediction="drifted",
            is_undocumented=False,
        )
        with pytest.raises(DetectionInputError) as exc_info:
            detector_default.detect([pred])
        assert "within [0.0, 1.0]" in str(exc_info.value)

    def test_boolean_probability_raises(self, detector_default):
        pred_true = ModelPrediction(
            file_path="mod.py",
            qualified_name="fn",
            line_number=1,
            drift_probability=True,  # type: ignore
            prediction="drifted",
            is_undocumented=False,
        )
        with pytest.raises(DetectionInputError) as exc_info:
            detector_default.detect([pred_true])
        assert "must be numeric (not bool)" in str(exc_info.value)

        pred_false = ModelPrediction(
            file_path="mod.py",
            qualified_name="fn",
            line_number=1,
            drift_probability=False,  # type: ignore
            prediction="aligned",
            is_undocumented=False,
        )
        with pytest.raises(DetectionInputError) as exc_info:
            detector_default.detect([pred_false])
        assert "must be numeric (not bool)" in str(exc_info.value)

    def test_malformed_prediction_missing_attribute_raises(self, detector_default):
        class MalformedPrediction:
            file_path = "mod.py"
            # Missing qualified_name, line_number, etc.

        with pytest.raises(DetectionInputError) as exc_info:
            detector_default.detect([MalformedPrediction()])  # type: ignore
        assert "Missing or invalid" in str(exc_info.value)
