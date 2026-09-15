"""
Unit tests for semdrift.model.architecture and semdrift.model.inference.

All tests run completely offline using dependency-injected DummyBackbone and MockTokenizer.
No external Hugging Face model downloads are made.
"""

from pathlib import Path
import pytest
import torch
import torch.nn as nn

from semdrift.model.architecture import JointEncoderModel
from semdrift.model.config import ModelConfig
from semdrift.model.exceptions import ModelInferenceError, ModelLoadError
from semdrift.model.inference import SemDriftModel
from semdrift.parser.models import CodeDocumentPair


class DummyBackbone(nn.Module):
    """Lightweight offline mock transformer backbone for unit testing."""

    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(2000, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        # Clip IDs to vocabulary range
        safe_ids = torch.clamp(input_ids, 0, 1999)
        emb = self.embedding(safe_ids)
        # Emulate HuggingFace ModelOutput object
        return type("Output", (), {"last_hidden_state": emb})()


class MockTokenizer:
    """Deterministic offline mock tokenizer."""

    def __init__(self):
        self.cls_token_id = 0
        self.pad_token_id = 1
        self.sep_token_id = 2
        self.mask_token_id = 3
        self.unk_token_id = 4

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if not text:
            return []
        return [10 + (hash(w) % 1000) for w in text.split()]


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


@pytest.fixture
def dummy_backbone():
    return DummyBackbone(hidden_size=16)


@pytest.fixture
def dummy_checkpoint(tmp_path, dummy_backbone):
    """Creates a temporary valid state_dict checkpoint for JointEncoderModel."""
    model = JointEncoderModel(encoder=dummy_backbone, num_labels=2)
    ckpt_path = tmp_path / "dummy_checkpoint.pt"
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path


class TestJointEncoderArchitecture:
    """Verify PyTorch JointEncoderModel forward pass and pooling."""

    def test_forward_pass_shapes(self, dummy_backbone):
        model = JointEncoderModel(encoder=dummy_backbone, num_labels=2)
        model.eval()

        input_ids = torch.tensor([[0, 10, 2, 2, 20, 2], [0, 15, 2, 2, 25, 2]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]], dtype=torch.long)

        logits = model({"input_ids": input_ids, "attention_mask": attention_mask})
        assert logits.shape == (2, 2)
        assert logits.dtype == torch.float32


class TestSemDriftModelInference:
    """Verify SemDriftModel checkpoint loading, ordering, and batching."""

    def test_from_checkpoint_loading(self, dummy_checkpoint, mock_tokenizer):
        model = SemDriftModel.from_checkpoint(
            checkpoint_path=dummy_checkpoint,
            encoder=DummyBackbone(16),
            tokenizer=mock_tokenizer,
            device="cpu",
        )
        assert model.device == "cpu"
        assert not model.model.training  # model.eval() was called

    def test_nonexistent_checkpoint_raises_model_load_error(self, mock_tokenizer):
        with pytest.raises(ModelLoadError) as exc_info:
            SemDriftModel.from_checkpoint(
                checkpoint_path="nonexistent_path_123.pt",
                encoder=DummyBackbone(16),
                tokenizer=mock_tokenizer,
            )
        assert "not found" in exc_info.value.message

    def test_corrupt_checkpoint_raises_model_load_error(self, tmp_path, mock_tokenizer):
        corrupt_path = tmp_path / "corrupt.pt"
        corrupt_path.write_text("not a valid torch file", encoding="utf-8")

        with pytest.raises(ModelLoadError) as exc_info:
            SemDriftModel.from_checkpoint(
                checkpoint_path=corrupt_path,
                encoder=DummyBackbone(16),
                tokenizer=mock_tokenizer,
            )
        assert "Failed to load checkpoint" in exc_info.value.message

    def test_predict_preserves_input_order_and_handles_undocumented(
        self, dummy_checkpoint, mock_tokenizer
    ):
        model = SemDriftModel.from_checkpoint(
            checkpoint_path=dummy_checkpoint,
            encoder=DummyBackbone(16),
            tokenizer=mock_tokenizer,
            device="cpu",
        )

        pairs = [
            CodeDocumentPair(
                file_path="service.py",
                qualified_name="documented_1",
                line_number=1,
                code="def documented_1(): return 1",
                docstring="Docstring 1.",
            ),
            CodeDocumentPair(
                file_path="service.py",
                qualified_name="undocumented_2",
                line_number=10,
                code="def undocumented_2(): return 2",
                docstring=None,  # Undocumented
            ),
            CodeDocumentPair(
                file_path="service.py",
                qualified_name="documented_3",
                line_number=20,
                code="def documented_3(): return 3",
                docstring="Docstring 3.",
            ),
            CodeDocumentPair(
                file_path="service.py",
                qualified_name="undocumented_4",
                line_number=30,
                code="def undocumented_4(): return 4",
                docstring=None,  # Undocumented
            ),
        ]

        preds = model.predict(pairs, batch_size=2)
        assert len(preds) == 4

        # Verify ordering is 1-to-1 preserved
        assert preds[0].qualified_name == "documented_1"
        assert preds[0].is_undocumented is False
        assert preds[0].drift_probability is not None
        assert 0.0 <= preds[0].drift_probability <= 1.0
        assert preds[0].prediction in ("drifted", "aligned")
        assert preds[0].raw_logits is not None
        assert len(preds[0].raw_logits) == 2

        assert preds[1].qualified_name == "undocumented_2"
        assert preds[1].is_undocumented is True
        assert preds[1].drift_probability is None
        assert preds[1].prediction == "undocumented"
        assert preds[1].raw_logits is None

        assert preds[2].qualified_name == "documented_3"
        assert preds[2].is_undocumented is False
        assert preds[2].drift_probability is not None

        assert preds[3].qualified_name == "undocumented_4"
        assert preds[3].is_undocumented is True
        assert preds[3].drift_probability is None
        assert preds[3].prediction == "undocumented"

    def test_batching_produces_consistent_results(self, dummy_checkpoint, mock_tokenizer):
        model = SemDriftModel.from_checkpoint(
            checkpoint_path=dummy_checkpoint,
            encoder=DummyBackbone(16),
            tokenizer=mock_tokenizer,
            device="cpu",
        )

        pairs = [
            CodeDocumentPair(
                file_path="math.py",
                qualified_name=f"func_{i}",
                line_number=i * 5,
                code=f"def func_{i}(): return {i}",
                docstring=f"Docstring for function {i}.",
            )
            for i in range(5)
        ]

        # Predict with batch_size=1
        preds_b1 = model.predict(pairs, batch_size=1)
        # Predict with batch_size=3
        preds_b3 = model.predict(pairs, batch_size=3)

        assert len(preds_b1) == len(preds_b3) == 5
        for p1, p3 in zip(preds_b1, preds_b3):
            assert p1.qualified_name == p3.qualified_name
            assert p1.drift_probability == p3.drift_probability
            assert p1.prediction == p3.prediction

    def test_empty_pairs_returns_empty(self, dummy_checkpoint, mock_tokenizer):
        model = SemDriftModel.from_checkpoint(
            checkpoint_path=dummy_checkpoint,
            encoder=DummyBackbone(16),
            tokenizer=mock_tokenizer,
            device="cpu",
        )
        assert model.predict([]) == []

    def test_all_undocumented_pairs(self, dummy_checkpoint, mock_tokenizer):
        model = SemDriftModel.from_checkpoint(
            checkpoint_path=dummy_checkpoint,
            encoder=DummyBackbone(16),
            tokenizer=mock_tokenizer,
            device="cpu",
        )

        pairs = [
            CodeDocumentPair(
                file_path="a.py",
                qualified_name="undoc_a",
                line_number=1,
                code="pass",
                docstring=None,
            ),
            CodeDocumentPair(
                file_path="b.py",
                qualified_name="undoc_b",
                line_number=5,
                code="pass",
                docstring=None,
            ),
        ]
        preds = model.predict(pairs)
        assert len(preds) == 2
        assert all(p.is_undocumented for p in preds)
        assert all(p.prediction == "undocumented" for p in preds)

    def test_inference_error_handling(self, dummy_checkpoint, mock_tokenizer):
        class BrokenEncoder(DummyBackbone):
            def forward(self, *args, **kwargs):
                raise RuntimeError("Tensor allocation failure during forward")

        model = SemDriftModel.from_checkpoint(
            checkpoint_path=dummy_checkpoint,
            encoder=BrokenEncoder(16),
            tokenizer=mock_tokenizer,
            device="cpu",
        )

        pairs = [
            CodeDocumentPair(
                file_path="a.py",
                qualified_name="func",
                line_number=1,
                code="pass",
                docstring="Doc.",
            )
        ]

        with pytest.raises(ModelInferenceError) as exc_info:
            model.predict(pairs)
        assert "Inference execution failed" in exc_info.value.message
