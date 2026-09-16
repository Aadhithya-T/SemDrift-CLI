# Phase 6 Design — CLI & End-to-End Orchestration

## 1. Scope & Responsibilities

The CLI layer (`semdrift.cli`) is the orchestration and user-facing entry point for SemDrift. It coordinates the underlying pipeline layers into a cohesive end-to-end execution workflow.

### Core Responsibilities
- Parsing command-line arguments and flags via standard-library `argparse`.
- Validating runtime configuration options (`batch_size > 0`, `0.0 <= threshold <= 1.0`, required checkpoint).
- Resolving execution devices (`auto`, `cpu`, `cuda`).
- Orchestrating the pipeline without duplicating domain logic:
  ```
  Repository / Path
        ↓
     Scanner
        ↓
  Discovered *.py files
        ↓
      Parser
        ↓
  CodeDocumentPair[]
        ↓
      Model
        ↓
  ModelPrediction[]
        ↓
    Detection
        ↓
   DriftResult[]
        ↓
    Reporting
        ↓
     stdout
  ```
- Dispatching to the requested reporter (`TerminalReporter`, `MarkdownReporter`, `JsonReporter`).
- Emitting reports exclusively to `stdout`.
- Emitting diagnostic warnings and errors exclusively to `stderr`.
- Returning deterministic process exit codes (`0`, `1`, `2`).

### Strict Non-Responsibilities (Architectural Boundaries)
The CLI layer is strictly prohibited from:
- Parsing Python ASTs directly or extracting function docstrings.
- Traversing directories or filtering files directly (delegates to `RepositoryScanner`).
- Running model embeddings or tensor operations directly (delegates to `SemDriftModel`).
- Computing drift probabilities or evaluating detection thresholds (delegates to `DriftDetector`).
- Constructing report text, markdown tables, or JSON formatting manually (delegates to `semdrift.reporting`).
- Sorting, grouping, or modifying the order of results produced by upstream layers.
- Persisting reports to disk or accepting file-writing flags (`--output`).

---

## 2. CLI Interface & Options

### Primary Command
```bash
semdrift scan <path> [options]
```
Or via Python module execution:
```bash
python -m semdrift scan <path> [options]
```

### Positional Arguments
- `path`: Path to a repository directory or a single Python source file (`.py`).

### Options & Defaults
| Option | Type / Choices | Default | Description |
|---|---|---|---|
| `--format` | `terminal`, `markdown`, `json` | `terminal` | Format of the report rendered to `stdout`. |
| `--threshold` | `float` in `[0.0, 1.0]` | `0.50` | Decision threshold applied by `DriftDetector`. |
| `--checkpoint` | `path` (string) | **Required** | Path to the trained PyTorch state_dict checkpoint (`.pt`). |
| `--batch-size` | `int` (`> 0`) | `16` | Inference batch size passed to `SemDriftModel.predict`. |
| `--device` | `auto`, `cpu`, `cuda` | `auto` | Execution device. `auto` selects CUDA if available, else CPU. |

---

## 3. Exit Codes

The CLI enforces deterministic process exit codes:
- **`0` — Success**:
  The scan completed successfully and the report was written to `stdout`.
  *(Note: Finding drift is a normal analytical outcome, not a failure; scans detecting drift exit with `0`)*.
- **`1` — Operational / Runtime Error**:
  An expected application-layer error occurred during execution:
  - Checkpoint file not found or corrupted (`ModelLoadError`).
  - Target scan path does not exist or cannot be read (`ScanError`).
  - CUDA was requested explicitly via `--device cuda` on a system without CUDA (`CLIRuntimeError`).
  - Model inference or detection failed on input data (`ModelError`, `DetectionError`).
- **`2` — Argument / Configuration Error**:
  Command-line arguments were invalid:
  - Missing required `--checkpoint` argument.
  - Unknown subcommand or flag.
  - `--batch-size <= 0`.
  - `--threshold` outside `[0.0, 1.0]`.
  - Invalid choice for `--format` or `--device`.

*Note: Invocations requesting `--help` return exit code `0`.*

---

## 4. stdout / stderr Separation

To ensure reliable automation and scripting, stream separation is strictly maintained:
- **`stdout`**:
  Receives **only** the rendered report. No log banners, debugging text, progress updates, or error messages are written to `stdout`.
  For `--format json`, `stdout` is guaranteed to be parseable with `json.loads(sys.stdin.read())`.
- **`stderr`**:
  Receives all diagnostic warnings and error messages.
  - If an individual file in a scanned repository cannot be parsed due to a syntax error, a non-fatal warning is written to `stderr` (`Warning: Failed to parse '...': ...`), while the remaining files are processed and reported normally.
  - User-facing error messages on exit codes `1` or `2` are written to `stderr`.

---

## 5. Traceback & Error Handling Policy

Unexpected bugs or internal programmer errors (e.g. `AttributeError`, `KeyError`, `IndexError`, unhandled exceptions) are allowed to bubble up naturally with complete tracebacks. The CLI only catches anticipated domain-level errors:
- `CLIRuntimeError`
- `ScanError`
- `ModelError`
- `DetectionError`
- `ReportingError`
- `CLIArgumentError`
