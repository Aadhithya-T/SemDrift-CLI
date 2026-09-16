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

> **Phase 6 — Complete End-to-End Runtime Pipeline**
>
> SemDrift provides a complete end-to-end command-line workflow:
> repository scanning (`semdrift.scanner`), Python AST extraction (`semdrift.parser`),
> CodeBERT Joint-Encoder model inference (`semdrift.model`), threshold-based drift detection (`semdrift.detection`),
> multi-format reporting (`semdrift.reporting`), and command-line execution (`semdrift.cli`).

## Architecture

The implementation follows a modular pipeline architecture:

```
CLI → Scanner → Parser → Code/Doc pairs → Model → Detection → Reporting → stdout
```

Each stage is a separate module with clearly defined responsibilities. See [docs/architecture.md](docs/architecture.md) for details.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd SemDrift-CLI

# Install in development mode
pip install -e ".[dev]"
```

## Usage

Run a scan using the `semdrift` command or through module execution:

```bash
# Terminal output (default)
semdrift scan ./my-project --checkpoint ./model/joint_encoder_checkpoint.pt

# Markdown output
semdrift scan ./my-project --checkpoint ./model/joint_encoder_checkpoint.pt --format markdown

# Machine-readable JSON output
semdrift scan ./my-project --checkpoint ./model/joint_encoder_checkpoint.pt --format json

# Python module execution
python -m semdrift scan ./my-project --checkpoint ./model/joint_encoder_checkpoint.pt
```

### CLI Options

- `path`: Positional directory or `.py` file to scan.
- `--checkpoint PATH`: **(Required)** Path to trained PyTorch model checkpoint (`.pt`).
- `--format {terminal,markdown,json}`: Output report format (default: `terminal`).
- `--threshold FLOAT`: Decision threshold in `[0.0, 1.0]` (default: `0.50`).
- `--batch-size INT`: Inference batch size (default: `16`).
- `--device {auto,cpu,cuda}`: Execution device (default: `auto`).


## Development

```bash
# Run tests
pytest tests/ -v
```

## Research & Implementation Boundary

For a detailed classification of which research components are being ported to this implementation and which remain research-only, see [docs/research-vs-runtime.md](docs/research-vs-runtime.md).

## License

MIT — see [LICENSE](LICENSE).
