"""
Configuration dataclass for the SemDrift model layer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Configuration settings for SemDrift model loading, preprocessing, and inference.

    All default parameters match the trained CodeBERT Joint Encoder checkpoint
    from the SemDrift research repository.

    Attributes:
        model_name: HuggingFace model backbone identifier or local directory.
        max_length: Maximum total token length for joint (docstring + code) input. Cap: 512.
        doc_max_tokens: Maximum token budget allocated to the docstring.
        code_truncation: Truncation strategy when code exceeds remaining budget
            ('head_tail', 'head', or 'tail').
        pooling: Pooling strategy for the sequence representation ('cls').
        num_labels: Number of output classification logits (2: [aligned, drifted]).
        dropout: Dropout probability before classifier linear layer (inactive during eval).
        device: Target execution device ('cpu', 'cuda', etc.).
        batch_size: Default batch size for batch inference.
        clean_docstrings: Whether to strip REPL/section headers using extract_docstring_summary.
    """

    model_name: str = "microsoft/codebert-base"
    max_length: int = 512
    doc_max_tokens: int = 96
    code_truncation: str = "head_tail"
    pooling: str = "cls"
    num_labels: int = 2
    dropout: float = 0.1
    device: str = "cpu"
    batch_size: int = 16
    clean_docstrings: bool = True
