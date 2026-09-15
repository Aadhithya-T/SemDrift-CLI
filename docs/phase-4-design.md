# SemDrift — Phase 4 Design: Detection Layer

This document details the responsibilities, data contracts, decision logic, and threshold provenance for the Detection layer (Phase 4) of `SemDrift-CLI`.

---

## 1. Overview & Responsibility

The Detection layer sits strictly between the Model layer (Phase 3) and the future Reporting layer (Phase 5):

```
Repository
  ↓
Scanner
  ↓
AST Parser
  ↓
CodeDocumentPair[]
  ↓
SemDriftModel
  ↓
ModelPrediction[]
  ↓
[Detection: DriftDetector]
  ↓
DriftResult[]
  ↓
[Reporting: Phase 5]
  ↓
[CLI: Phase 6]
```

### Responsibility
The sole responsibility of the Detection layer is to take raw model predictions (`ModelPrediction[]`) and apply an explicit, configurable decision policy (`DetectionConfig`) to produce structured detection results (`DriftResult[]`).

### Boundaries
The Detection layer does **NOT**:
- Scan repositories or traverse directories
- Parse Python source or inspect ASTs
- Tokenize code or docstrings
- Load PyTorch models or perform inference
- Train, evaluate, or fine-tune models
- Format terminal output, Markdown, or JSON reports
- Implement CLI commands or argument parsing
- Group or filter results out (it is a pure, deterministic 1-to-1 transformation)

---

## 2. Input and Output Contracts

### Input Contract: `ModelPrediction`
Defined in `semdrift.model.models.ModelPrediction`:
- `file_path: str`
- `qualified_name: str`
- `line_number: int`
- `drift_probability: Optional[float]` in `[0.0, 1.0]` (or `None` if undocumented)
- `prediction: str` (`"drifted"`, `"aligned"`, or `"undocumented"`)
- `is_undocumented: bool`
- `raw_logits: Optional[Tuple[float, float]]`

### Output Contract: `DriftResult`
Defined in `semdrift.detection.models.DriftResult` (`@dataclass(frozen=True)`):
| Field | Type | Description |
|---|---|---|
| `file_path` | `str` | Normalized path to the source file. |
| `qualified_name` | `str` | Fully-qualified deterministic function name. |
| `line_number` | `int` | 1-indexed definition start line. |
| `status` | `str` | `"drifted"`, `"aligned"`, or `"undocumented"`. |
| `is_drift` | `bool` | `True` if `status == "drifted"`, `False` if `"aligned"` or `"undocumented"`. |
| `drift_probability` | `Optional[float]` | Continuous model score in `[0.0, 1.0]`, or `None` if undocumented. |
| `threshold` | `float` | The decision threshold applied. |

Convenience property:
- `is_undocumented: bool`: Returns `True` if `status == "undocumented"`.

---

## 3. Configuration & Threshold Semantics

### `DetectionConfig` (`@dataclass(frozen=True)`)
```python
@dataclass(frozen=True)
class DetectionConfig:
    drift_threshold: float = 0.5
```

### Validation Invariants
`DetectionConfig` validates in `__post_init__`:
1. `drift_threshold` must be a numeric `float` or `int` (booleans are explicitly rejected).
2. `drift_threshold` cannot be NaN or Infinite.
3. `drift_threshold` must satisfy `0.0 <= drift_threshold <= 1.0`.
4. Violations raise `DetectionConfigError`.

### Decision Boundary
For documented functions:
```python
if drift_probability >= drift_threshold:
    status = "drifted"
    is_drift = True
else:
    status = "aligned"
    is_drift = False
```
- Exactly at the threshold (`drift_probability == drift_threshold`), the function is classified as **drifted**.
- Below the threshold (`drift_probability < drift_threshold`), the function is classified as **aligned**.

---

## 4. Model Probability vs. Detection Decision

A fundamental design principle of SemDrift-CLI is the strict conceptual separation between:

1. **Model Output**:
   A continuous probability estimate $P(\text{drift} \mid \text{code}, \text{doc}) \in [0.0, 1.0]$ produced by the CodeBERT Joint-Encoder.
2. **Detection Policy**:
   A discrete business/quality decision applied via `DetectionConfig(drift_threshold=...)`.

The model layer outputs what the neural network observed; the detection layer decides what action to recommend based on caller risk tolerance. Downstream tools can tune `drift_threshold` (e.g. higher threshold for high-precision gating in CI; lower threshold for thorough auditing) without altering model weights or inference results.

---

## 5. Threshold Provenance: V1 Synthetic vs. V2 Real-World Benchmarks

The research repository (`scripts/scan_repo.py`) defaulted to `0.50` (with `0.60` referenced in example docstrings).

### Empirical Context from Research
- **V1 Controlled Synthetic Benchmark**:
  Showed high separability and strong macro-F1 scores around a balanced 0.5 decision boundary.
- **V2 Real-World Grounded Diagnostic Benchmark**:
  Demonstrated that real-world semantic drift exhibits substantially more subtle divergence (e.g. subtle param mutations, implicit return changes, stale docstrings). On authentic software repositories, optimal thresholds depend heavily on whether a team prioritizes precision (avoiding alert fatigue) or recall (catching every subtle divergence).

### Architectural Conclusion
The default `drift_threshold = 0.5` in SemDrift-CLI is established strictly as a **standard technical decision boundary default**, **NOT** as an empirically calibrated or validated production optimum.
Detection does not claim 0.5 is optimal; it provides an explicit, tunable parameter via `DetectionConfig`.

---

## 6. Undocumented Functions Policy

In Phase 2 and Phase 3, functions without docstrings are explicitly modeled with `docstring=None` and `ModelPrediction(is_undocumented=True, drift_probability=None, prediction="undocumented")`.

In Detection:
1. Undocumented functions are **NOT** evaluated against the numerical threshold.
2. Detection does **NOT** fabricate artificial probabilities (`0.0` or `1.0`).
3. Undocumented functions produce:
   - `status = "undocumented"`
   - `is_drift = False`
   - `drift_probability = None`
4. This ensures undocumented functions are never misclassified as code/doc drift, leaving reporting policy (e.g. warnings vs. filters) to Phase 5.

---

## 7. Ordering and Determinism Guarantees

- **1-to-1 Mapping**: Each input `ModelPrediction` produces exactly one `DriftResult`.
- **Order Preservation**: Input list order is strictly maintained. Detection never reorders, groups by file, or filters by threshold.
- **Determinism**: Given identical `ModelPrediction` sequences and `DetectionConfig`, `DriftDetector.detect()` is 100% deterministic and idempotent.

---

## 8. Error Handling

- `DetectionError`: Base exception for all detection issues.
- `DetectionConfigError`: Raised on invalid threshold parameters (e.g., negative, >1.0, non-numeric, NaN).
- `DetectionInputError`: Raised on malformed `ModelPrediction` objects:
  - Missing required fields (`file_path`, `qualified_name`, `line_number`, `is_undocumented`).
  - Documented predictions where `drift_probability is None`.
  - Predictions where `drift_probability` is boolean, non-numeric, NaN/Inf, or outside `[0.0, 1.0]`.

---

## 9. Deliberately Excluded Functionality

- **No Reporting**: Formatting as JSON, Markdown, or terminal tables belongs to Phase 5.
- **No Severity Scales**: Arbitrary "low/medium/high" severity categorizations are not introduced.
- **No CLI**: Command line entry points, options parsing, and exit codes belong to Phase 6.
- **No Model Inference**: Zero transformer or PyTorch dependencies exist in `semdrift.detection`.
