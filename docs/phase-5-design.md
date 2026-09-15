# Phase 5 Design — Reporting Layer

## 1. Scope & Responsibilities

The reporting layer (`semdrift.reporting`) converts structured detection results produced by Phase 4 (`DriftResult`) into serialized and formatted representations for human consumption and programmatic integration.

### Core Responsibility
- Pure presentation and serialization of `Sequence[DriftResult]`.
- Provide format-specific renderers:
  - `JsonReporter`: Deterministic machine-readable JSON array.
  - `MarkdownReporter`: GitHub Flavored Markdown with summary and results tables.
  - `TerminalReporter`: Concise, plain-text developer summary and results listing.

### Strict Non-Responsibilities (Architectural Boundaries)
The reporting layer is strictly prohibited from:
- Re-evaluating or modifying detection decisions (`probability >= threshold`).
- Modifying or inventing severity levels, confidence ratings, or risk scores.
- Scanning repositories or reading files from disk.
- Parsing Python code or ASTs.
- Loading PyTorch models or running inference.
- Handling command-line arguments or exit codes.
- Performing file I/O or writing directly to terminal streams via `print()`.

---

## 2. Input / Output Contracts

### Input Contract: `Sequence[DriftResult]`
Reporters accept an iterable sequence of immutable `DriftResult` objects.
Each `DriftResult` enforces the following invariants at creation (`DriftResult.__post_init__`):
- `status`: One of `'drifted'`, `'aligned'`, `'undocumented'`.
- `is_drift`: `bool` (rejects non-boolean values).
- `drift_probability`: `None` if `status == 'undocumented'`; otherwise a float within `[0.0, 1.0]`.
- `threshold`: Float decision threshold applied.

Reporters validate that every element in the input sequence is an instance of `DriftResult`, raising `ReportingInputError` for invalid or foreign objects.

### Output Contract: `str`
Each reporter exposes a unified interface:
```python
def render(self, results: Sequence[DriftResult]) -> str:
    ...
```
All reporters return a formatted string. File writing and standard output redirection are deferred to future CLI orchestration (Phase 6).

---

## 3. Format Specifications

### JSON Format (`JsonReporter`)
Produces a standard JSON array of objects using semantic field ordering:
```json
[
  {
    "file_path": "src/auth.py",
    "qualified_name": "login",
    "line_number": 15,
    "status": "drifted",
    "is_drift": true,
    "drift_probability": 0.82,
    "threshold": 0.5
  },
  {
    "file_path": "src/utils.py",
    "qualified_name": "format_date",
    "line_number": 10,
    "status": "undocumented",
    "is_drift": false,
    "drift_probability": null,
    "threshold": 0.5
  }
]
```
- Undocumented functions preserve `"drift_probability": null` (never replaced with `0.0`).
- Empty input `[]` renders as `"[]"`.

### Markdown Format (`MarkdownReporter`)
Generates GitHub Flavored Markdown with two main sections:
1. **Summary Table**:
   ```markdown
   # SemDrift Report

   ## Summary

   | Metric | Count |
   |---|---:|
   | Total | 2 |
   | Drifted | 1 |
   | Aligned | 0 |
   | Undocumented | 1 |
   ```
2. **Results Table**:
   ```markdown
   ## Results

   | File | Function | Line | Status | Drift Probability | Threshold |
   |---|---|---:|---|---:|---:|
   | src/auth.py | login | 15 | drifted | 0.8200 | 0.50 |
   | src/utils.py | format_date | 10 | undocumented | N/A | 0.50 |
   ```
- Pipe delimiter characters (`|`) in file paths and qualified names are escaped as `\|`.
- Undocumented entries display `N/A` for drift probability.

### Terminal Format (`TerminalReporter`)
Generates concise, human-readable plain text without third-party dependencies or raw terminal escapes:
```text
SemDrift Detection Summary
==========================
Total:        2
Drifted:      1
Aligned:      0
Undocumented: 1

Results:
  [drifted] src/auth.py:15 - login (probability: 0.8200, threshold: 0.50)
  [undocumented] src/utils.py:10 - format_date (probability: N/A, threshold: 0.50)
```
- When no results are provided (`[]`), outputs summary counts of 0 and `"No results found."`.

---

## 4. Design Decisions & Guarantees

### Strict Summary Semantics & No Recalculation
Summary counts are derived **solely** by counting `DriftResult.status` values:
- `Total`: `len(results)`
- `Drifted`: `sum(1 for r in results if r.status == 'drifted')`
- `Aligned`: `sum(1 for r in results if r.status == 'aligned')`
- `Undocumented`: `sum(1 for r in results if r.status == 'undocumented')`

Reporters never re-apply the threshold rule (`drift_probability >= threshold`). If a detection outcome has `drift_probability = 0.20` but `status = 'drifted'`, the reporter faithfully presents it as drifted.

### Input Order Preservation
All reporters process and render results in the exact order provided by the caller. No sorting (alphabetical, path-based, probability-based) or grouping is performed.

### Determinism & Immutability
- Given identical inputs, each reporter produces byte-for-byte identical output across repeated invocations.
- Input sequences and `DriftResult` objects are treated as strictly immutable and never modified.

### Error Handling
- `ReportingError`: Base exception for reporting failures.
- `ReportingInputError`: Raised when non-`DriftResult` objects are passed in the results sequence, identifying the offending index.
