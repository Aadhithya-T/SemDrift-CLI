# SemDrift — Architecture

This document describes the modular architecture of the SemDrift implementation repository.

## Design Principle

The implementation decomposes semantic drift detection into isolated, single-responsibility modules. There is no monolithic pipeline — the CLI orchestrates separate stages rather than containing their implementation.

## Data Flow

```
Repository
    ↓
Scanner         — discovers Python source files
    ↓
Parser          — extracts code/documentation pairs via AST
    ↓
Code/Documentation pairs
    ↓
Model           — loads trained model, performs inference
    ↓
Detection       — calculates drift scores, applies thresholds
    ↓
Reporting       — formats results (terminal, JSON, Markdown)
    ↓
CLI             — user-facing command-line interface
```

---

## Module Responsibilities

### `semdrift.parser` (Implemented in Phase 2)

Responsible for understanding Python source code via Python's standard-library `ast` module.

**Implemented components:**
- `PythonASTParser`: Walks AST, identifies top-level functions, class methods, async functions/methods, and nested functions.
- `CodeDocumentPair`: Immutable data contract representing extracted functions, source code (including decorators and relative indentation), docstrings, deterministic qualified names, and 1-indexed line locations.
- `ParseError`: Structured exception tracking file path, line number, message, and underlying cause for syntax/read errors.

**Must NOT:**
- Load ML models or perform inference
- Generate reports
- Implement CLI behavior

---

### `semdrift.scanner` (Implemented in Phase 2)

Responsible for discovering Python source files across directories or single file targets.

**Implemented components:**
- `RepositoryScanner`: Recursively discovers `*.py` files, prunes excluded directories (`.git`, `.venv`, `venv`, `env`, `__pycache__`, `.pytest_cache`, `node_modules`), supports single-file inputs, enforces a configurable file size guard (`max_file_size_bytes`), and deterministically sorts discovered paths.
- `ScanError`: Exception tracking target path and access/traversal failures.

**Must NOT:**
- Perform ML inference
- Parse AST or inspect function internals
- Calculate drift scores
- Format CLI output

---

### `semdrift.model` (Implemented in Phase 3)

Responsible only for model loading, token preprocessing, and inference.

**Implemented components:**
- `SemDriftModel`: High-level inference class with `from_checkpoint()` and `predict()`, supporting batching and ordering preservation.
- `JointEncoderModel`: PyTorch `nn.Module` implementing joint code-doc self-attention with CodeBERT backbone and `[CLS]` token classification.
- `ModelConfig`: Immutable configuration settings matching the trained research model.
- `ModelPrediction`: Immutable prediction contract carrying drift probabilities, predicted class, and undocumented indicators.
- `ModelError`, `ModelLoadError`, `ModelInferenceError`: Structured exception hierarchy.
- Preprocessing: `extract_docstring_summary` and `prepare_joint_tokens` with `head_tail` code budget allocation.

**Must NOT:**
- Scan repositories or parse ASTs
- Decide drift thresholds or severity levels
- Print terminal output or format reports
- Include training, optimization, or evaluation routines

---

### `semdrift.detection` (Implemented in Phase 4)

Responsible for converting raw model predictions into structured drift detection decisions.

**Implemented components:**
- `DriftDetector`: Core detection engine applying configured threshold policy to `ModelPrediction` records.
- `DetectionConfig`: Immutable configuration (`drift_threshold`) with strict numeric and bounds validation.
- `DriftResult`: Immutable result contract maintaining 1-to-1 input ordering, probability, status (`'drifted'`, `'aligned'`, or `'undocumented'`), and metadata.
- `DetectionError`, `DetectionConfigError`, `DetectionInputError`: Structured exception hierarchy.
- Undocumented policy: Functions with `docstring is None` are preserved as `status="undocumented"` and `is_drift=False` without threshold evaluation.

**Must NOT:**
- Traverse repositories or parse source code
- Load PyTorch models or perform inference
- Format terminal output or export reports
- Contain CLI-specific logic

---

### `semdrift.reporting`

Responsible for converting detection results into user-facing formats.

**Will eventually support:**
- Terminal output (rich formatted tables, panels)
- JSON export
- Markdown export

**Must NOT:**
- Perform model inference
- Parse source code
- Scan repositories

---

### `semdrift.cli`

Responsible only for the command-line interface.

**Will eventually expose commands such as:**
```
semdrift scan .
semdrift scan ./src --threshold 0.6 --output json
```

**Design rule:** The CLI orchestrates the other modules. It does not directly implement AST parsing, model inference, or drift scoring.

**Must NOT:**
- Contain parsing logic
- Contain model inference logic
- Contain drift scoring logic
- Contain report formatting logic beyond dispatching to the reporting module
