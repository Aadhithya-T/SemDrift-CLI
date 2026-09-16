# SemDrift — Implementation

**Detect semantic drift between source code and documentation in Python repositories.**

SemDrift identifies docstrings that no longer match the behavior of their associated code — a common maintenance problem in real-world software projects.

---

## Repository Purpose

This repository contains the **production-oriented implementation** of SemDrift as an end-to-end developer tool.

The original [SemDrift research repository](https://github.com/Aadhithya-T/SemDrift) contains:
- Research framework and empirical methodology
- Datasets (V1 synthetic, V2 real-world grounded)
- Training scripts and model architectures
- Benchmarking results and analysis
- Experiment reproducibility artifacts

This implementation repository is **derived from and informed by** the research repository. It is being developed incrementally — not all planned functionality exists yet.

## Current Status

> **v1.0 Production-Ready CLI Implementation**
>
> SemDrift provides a complete end-to-end command-line workflow:
> repository scanning (`semdrift.scanner`), Python AST extraction (`semdrift.parser`),
> CodeBERT Joint-Encoder model inference (`semdrift.model`), threshold-based drift detection (`semdrift.detection`),
> multi-format reporting (`semdrift.reporting`), and command-line execution (`semdrift.cli`).

## Architecture

The implementation follows a strict, single-responsibility pipeline architecture:

```
CLI → Scanner → Parser → Code/Doc pairs → Model → Detection → Reporting → stdout
```

Each stage is an isolated module with clearly defined boundaries:
- `semdrift.scanner`: Discovers Python files, pruning excluded directories (`.git`, `.venv`, etc.).
- `semdrift.parser`: Extracts function/method definitions, relative source code, and docstrings via AST.
- `semdrift.model`: Preprocesses joint tokens and runs CodeBERT inference to yield drift probabilities.
- `semdrift.detection`: Applies configurable decision thresholds without recalculating inference.
- `semdrift.reporting`: Pure presentation layer producing Terminal, Markdown, or JSON reports.
- `semdrift.cli`: User-facing argument parsing, configuration validation, and stream orchestration.

See [docs/architecture.md](docs/architecture.md) for full architectural specifications.

## Installation

```bash
# Clone the repository
git clone https://github.com/Aadhithya-T/SemDrift-CLI.git
cd SemDrift-CLI

# Install package in development mode
pip install -e ".[dev]"
```

## Model Checkpoint Requirement

SemDrift-CLI is an inference and developer tool that intentionally does **not** bundle large model checkpoint weights into the package repository. 

To run a scan, you must supply a trained CodeBERT Joint-Encoder checkpoint (`.pt` file) via the `--checkpoint` option:

```bash
semdrift scan ./src --checkpoint /path/to/joint_encoder_checkpoint.pt
```

Checkpoints must contain the `state_dict` of the fine-tuned CodeBERT Joint Encoder architecture as trained in the [SemDrift research repository](https://github.com/Aadhithya-T/SemDrift).

## Usage

Run a scan using the `semdrift` command or through module execution (`python -m semdrift`):

```bash
# 1. Terminal output (default)
semdrift scan ./my-project --checkpoint ./weights/joint_encoder.pt

# 2. Markdown output (suitable for PR comments or CI logs)
semdrift scan ./my-project --checkpoint ./weights/joint_encoder.pt --format markdown

# 3. Machine-readable JSON output (pristine stdout, pipeable to jq)
semdrift scan ./my-project --checkpoint ./weights/joint_encoder.pt --format json

# 4. Custom threshold and batch size on CPU
semdrift scan ./my-project --checkpoint ./weights/joint_encoder.pt --threshold 0.70 --batch-size 32 --device cpu

# 5. Scanning a single Python file
semdrift scan ./my-project/auth.py --checkpoint ./weights/joint_encoder.pt
```

### CLI Options

| Option | Type / Choices | Default | Description |
|---|---|---|---|
| `path` | Positional `str` / `Path` | *Required* | Path to target repository directory or single `.py` file to scan. |
| `--checkpoint PATH` | `str` / `Path` | *Required* | Path to the trained PyTorch state_dict checkpoint (`.pt`). |
| `--format` | `terminal`, `markdown`, `json` | `terminal` | Format of the report rendered to `stdout`. |
| `--threshold FLOAT` | `float` in `[0.0, 1.0]` | `0.50` | Decision threshold applied by `DriftDetector`. |
| `--batch-size INT` | `int` (`> 0`) | `16` | Inference batch size passed to the model inference layer. |
| `--device` | `auto`, `cpu`, `cuda` | `auto` | Execution device. `auto` uses CUDA if available, else CPU. |

### Output Streams

- **`stdout`**: Exclusively receives the formatted report. When using `--format json`, `stdout` is guaranteed to be clean, parseable JSON with no log banners or progress noise.
- **`stderr`**: Exclusively receives diagnostic warnings (such as skipped unparseable files) and operational errors.

### Process Exit Codes

| Exit Code | Meaning | Description |
|---|---|---|
| `0` | **Success** | Scan completed successfully and report was emitted. *(Note: detecting semantic drift is a normal analytical finding, not a process failure)*. |
| `1` | **Runtime Error** | Checkpoint missing/corrupt, target path unreadable, explicit CUDA unavailable, or model failure. |
| `2` | **Argument Error** | Invalid CLI options, missing required `--checkpoint`, negative batch size, or out-of-bounds threshold. |

## Model Limitations & Empirical Considerations

1. **Inference Tool Scope**: SemDrift-CLI is an execution and reporting runtime. Model accuracy and generalization depend strictly on the weights provided via `--checkpoint`.
2. **Technical Default Threshold**: The default `--threshold 0.50` is a technical baseline, **not** an empirically calibrated production optimum.
3. **Research-Grounded Reality**: Empirical research on semantic drift models reveals that models evaluated solely on synthetic transformations (e.g. synthetic identifier perturbations) show performance degradation on authentic, real-world maintenance drift. High scores on synthetic benchmarks do not guarantee real-world generalization.
4. **Validation Recommended**: Users should evaluate and calibrate the detection threshold against representative code/documentation samples from their own repository before relying on results for automated enforcement.

## Development & Testing

```bash
# Run the complete test suite (unit + integration)
pytest tests/ -v
```

## Research & Implementation Boundary

For a detailed classification of which research components are ported to this implementation and which remain research-only (datasets, training loops, ablation studies, benchmarking), see [docs/research-vs-runtime.md](docs/research-vs-runtime.md).

## License

MIT — see [LICENSE](LICENSE).

