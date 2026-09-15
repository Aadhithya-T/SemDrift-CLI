"""
Inference runner for SemDrift models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence, Union
import torch
import torch.nn.functional as F

from semdrift.model.architecture import JointEncoderModel
from semdrift.model.config import ModelConfig
from semdrift.model.exceptions import ModelInferenceError, ModelLoadError
from semdrift.model.models import ModelPrediction
from semdrift.model.preprocessing import (
    collate_joint_batch,
    extract_docstring_summary,
    prepare_joint_tokens,
)
from semdrift.parser.models import CodeDocumentPair


class SemDriftModel:
    """Production runtime inference model for SemDrift semantic drift detection.

    Consumes CodeDocumentPair objects from the parser, preprocesses code and docstrings,
    and executes inference through the fine-tuned CodeBERT Joint Encoder to produce
    ModelPrediction results.

    Parameters:
        model: Initialized and loaded JointEncoderModel PyTorch module.
        tokenizer: Tokenizer for encoding input texts.
        config: ModelConfig containing inference hyperparameters.
        device: Target execution device.
    """

    def __init__(
        self,
        model: JointEncoderModel,
        tokenizer: Any,
        config: ModelConfig,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        config: Optional[ModelConfig] = None,
        device: Optional[str] = None,
        encoder: Optional[torch.nn.Module] = None,
        tokenizer: Optional[Any] = None,
    ) -> SemDriftModel:
        """Instantiate a SemDriftModel by loading weights from a checkpoint file.

        Parameters:
            checkpoint_path: Path to the trained PyTorch state_dict checkpoint (.pt).
            config: Optional ModelConfig override. If None, default ModelConfig is used.
            device: Optional device override ('cpu' or 'cuda').
            encoder: Optional pre-constructed encoder backbone (useful for offline unit testing).
            tokenizer: Optional pre-constructed tokenizer (useful for offline unit testing).

        Returns:
            An evaluation-ready SemDriftModel instance.

        Raises:
            ModelLoadError: If checkpoint file does not exist, cannot be read, or fails to load.
        """
        cfg = config or ModelConfig()
        target_device = device or cfg.device

        path = Path(checkpoint_path)
        if not path.exists():
            raise ModelLoadError(
                message=f"Checkpoint file not found: '{checkpoint_path}'",
                checkpoint_path=str(checkpoint_path),
            )

        # 1. Load checkpoint state_dict
        try:
            state_dict = torch.load(path, map_location=target_device)
        except Exception as exc:
            raise ModelLoadError(
                message=f"Failed to load checkpoint file: {exc}",
                checkpoint_path=str(checkpoint_path),
                cause=exc,
            ) from exc

        # 2. Initialize tokenizer
        if tokenizer is None:
            try:
                from transformers import AutoTokenizer

                tok = AutoTokenizer.from_pretrained(cfg.model_name)
            except Exception as exc:
                raise ModelLoadError(
                    message=f"Failed to load tokenizer '{cfg.model_name}': {exc}",
                    checkpoint_path=str(checkpoint_path),
                    cause=exc,
                ) from exc
        else:
            tok = tokenizer

        # 3. Initialize PyTorch model architecture
        try:
            joint_model = JointEncoderModel(
                model_name=cfg.model_name,
                num_labels=cfg.num_labels,
                dropout=cfg.dropout,
                encoder=encoder,
            )
            joint_model.load_state_dict(state_dict)
            joint_model.to(target_device)
            joint_model.eval()
        except Exception as exc:
            raise ModelLoadError(
                message=f"Failed to initialize or load weights into JointEncoderModel: {exc}",
                checkpoint_path=str(checkpoint_path),
                cause=exc,
            ) from exc

        return cls(model=joint_model, tokenizer=tok, config=cfg, device=target_device)

    def predict(
        self,
        pairs: Sequence[CodeDocumentPair],
        batch_size: Optional[int] = None,
    ) -> List[ModelPrediction]:
        """Execute semantic drift inference across a sequence of CodeDocumentPair records.

        Preserves the exact input ordering. For functions without docstrings (docstring=None),
        inference is skipped and a prediction with is_undocumented=True is returned.

        Parameters:
            pairs: Sequence of CodeDocumentPair objects from the parser layer.
            batch_size: Batch size for inference. Defaults to config.batch_size.

        Returns:
            List of ModelPrediction objects corresponding 1-to-1 with the input pairs.

        Raises:
            ModelInferenceError: If tensor creation or model execution fails.
        """
        if not pairs:
            return []

        eff_batch_size = batch_size or self.config.batch_size
        results: List[Optional[ModelPrediction]] = [None] * len(pairs)

        # Separate documented pairs from undocumented pairs to preserve exact ordering
        documented_items: List[tuple[int, CodeDocumentPair, str]] = []

        for idx, pair in enumerate(pairs):
            if pair.docstring is None:
                results[idx] = ModelPrediction(
                    file_path=pair.file_path,
                    qualified_name=pair.qualified_name,
                    line_number=pair.line_number,
                    drift_probability=None,
                    prediction="undocumented",
                    is_undocumented=True,
                    raw_logits=None,
                )
            else:
                doc_text = pair.docstring
                if self.config.clean_docstrings:
                    doc_text = extract_docstring_summary(doc_text)
                documented_items.append((idx, pair, doc_text))

        if not documented_items:
            # All pairs were undocumented
            return [r for r in results if r is not None]

        # Execute batch inference on documented pairs
        try:
            with torch.no_grad():
                for start_idx in range(0, len(documented_items), eff_batch_size):
                    batch_chunk = documented_items[start_idx : start_idx + eff_batch_size]

                    # Tokenize batch
                    token_sequences = [
                        prepare_joint_tokens(
                            doc=item[2],
                            code=item[1].code,
                            tokenizer=self.tokenizer,
                            max_length=self.config.max_length,
                            doc_max_tokens=self.config.doc_max_tokens,
                            truncation_strategy=self.config.code_truncation,
                        )
                        for item in batch_chunk
                    ]

                    # Collate into tensors and move to device
                    batch_inputs = collate_joint_batch(token_sequences, self.tokenizer)
                    batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}

                    # Model forward pass
                    logits = self.model(batch_inputs)
                    probs = F.softmax(logits, dim=-1)

                    drift_probs = probs[:, 1].cpu().tolist()
                    raw_logits_list = logits.cpu().tolist()

                    for chunk_idx, (orig_idx, p, _) in enumerate(batch_chunk):
                        d_prob = round(drift_probs[chunk_idx], 4)
                        label = "drifted" if d_prob >= 0.5 else "aligned"
                        r_logits = (
                            round(raw_logits_list[chunk_idx][0], 4),
                            round(raw_logits_list[chunk_idx][1], 4),
                        )

                        results[orig_idx] = ModelPrediction(
                            file_path=p.file_path,
                            qualified_name=p.qualified_name,
                            line_number=p.line_number,
                            drift_probability=d_prob,
                            prediction=label,
                            is_undocumented=False,
                            raw_logits=r_logits,
                        )
        except Exception as exc:
            raise ModelInferenceError(
                message=f"Inference execution failed: {exc}",
                batch_size=eff_batch_size,
                cause=exc,
            ) from exc

        return [r for r in results if r is not None]
