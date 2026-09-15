"""
PyTorch neural network architecture for the CodeBERT Joint-Encoder.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import torch
import torch.nn as nn


class JointEncoderModel(nn.Module):
    """Joint encoder for semantic drift detection using Joint Code-Documentation Self-Attention.

    Architecture matches the research Model B:
        Concatenates docstring and code into a single sequence:
            [CLS] docstring_tokens [SEP] [SEP] code_tokens [SEP]
        Executes a single forward pass through CodeBERT where bidirectional self-attention
        allows every token across both sequences to interact directly across all transformer layers.
        The [CLS] token representation is pooled and passed through a linear classifier head:
            [CLS] -> Dropout -> Linear(768 -> 2) -> logits

    Parameters:
        model_name: HuggingFace model backbone name or path. Defaults to 'microsoft/codebert-base'.
        num_labels: Number of output labels (2: [aligned, drifted]).
        dropout: Dropout rate before classification head.
        encoder: Optional pre-instantiated or mock encoder backbone (e.g. for offline unit tests).
    """

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        num_labels: int = 2,
        dropout: float = 0.1,
        encoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if encoder is not None:
            self.encoder = encoder
            hidden_size = getattr(encoder, "hidden_size", 768)
        else:
            from transformers import AutoModel

            self.encoder = AutoModel.from_pretrained(model_name)
            hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Execute forward pass.

        Parameters:
            inputs: Dictionary containing 'input_ids' and 'attention_mask' tensors.

        Returns:
            Logits tensor of shape (batch_size, num_labels).
        """
        outputs = self.encoder(**inputs)

        # Standard Hugging Face ModelOutput has .last_hidden_state; tuples have index 0
        if hasattr(outputs, "last_hidden_state"):
            last_hidden_state = outputs.last_hidden_state
        elif isinstance(outputs, (tuple, list)):
            last_hidden_state = outputs[0]
        else:
            raise ValueError(f"Unexpected encoder output type: {type(outputs)}")

        # [CLS] token pooling at index 0 (matching trained research checkpoint)
        pooled = last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits
