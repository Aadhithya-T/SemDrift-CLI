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

### `semdrift.parser`

Responsible for understanding Python source code.

**Will eventually:**
- Parse Python source using AST
- Identify functions and methods
- Extract docstrings
- Preserve source location information (file, line number)
- Produce normalized code/documentation pairs

**Must NOT:**
- Load ML models
- Perform inference
- Generate reports
- Implement CLI behavior

---

### `semdrift.scanner`

Responsible for traversing a user's repository and determining which source files and functions should be analyzed.

**Will eventually:**
- Walk directories recursively
- Identify Python files
- Respect exclusion patterns (e.g., `venv/`, `__pycache__/`, test files)
- Pass discovered source files to the parser

**Must NOT:**
- Perform ML inference
- Calculate drift scores
- Format CLI output

---

### `semdrift.model`

Responsible only for model loading and inference.

**Will eventually:**
- Load the selected trained SemDrift model (e.g., CodeBERT Joint Encoder)
- Preprocess model inputs (tokenization, truncation)
- Perform inference (forward pass)
- Return raw model predictions (logits or probabilities)

**Must NOT:**
- Scan repositories
- Traverse directories
- Print terminal output
- Decide how results are presented

---

### `semdrift.detection`

Responsible for converting model predictions into SemDrift detection results.

**Will eventually:**
- Calculate drift scores from model output
- Apply configurable thresholds
- Classify drift (e.g., aligned / drifted)
- Assign severity and/or confidence levels if supported by the model

**Must NOT:**
- Traverse repositories
- Print output
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
