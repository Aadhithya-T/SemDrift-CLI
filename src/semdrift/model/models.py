"""
Model layer prediction data contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class ModelPrediction:
    """Immutable prediction output from the SemDrift model inference layer.

    This dataclass represents the raw inference result for a single CodeDocumentPair,
    serving as the boundary contract between the Model layer and downstream Detection.

    Attributes:
        file_path: Normalized file path of the evaluated source function.
        qualified_name: Fully-qualified deterministic name of the function.
        line_number: 1-indexed start line number of the function.
        drift_probability: Model-estimated probability of semantic drift in [0.0, 1.0],
            or None if the function is undocumented and inference was skipped.
        prediction: Raw label prediction: 'drifted', 'aligned', or 'undocumented'.
        is_undocumented: True if the source function has docstring=None.
        raw_logits: Optional diagnostic tuple containing unnormalized logits
            (aligned_logit, drifted_logit).
    """

    file_path: str
    qualified_name: str
    line_number: int
    drift_probability: Optional[float] = None
    prediction: str = "aligned"
    is_undocumented: bool = False
    raw_logits: Optional[Tuple[float, float]] = None
