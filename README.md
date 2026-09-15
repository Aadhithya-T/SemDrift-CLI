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

> **Phase 1 — Foundation** (current)
>
> The repository has a clean package structure, module boundaries are defined,
> and the project can be installed as a local Python package.
> No scanning, parsing, model inference, or CLI commands are implemented yet.

## Architecture

The implementation follows a modular pipeline architecture:

```
Repository → Scanner → Parser → Code/Doc pairs → Model → Detection → Reporting
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

## Development

```bash
# Run tests
pytest tests/ -v
```

## Research & Implementation Boundary

For a detailed classification of which research components are being ported to this implementation and which remain research-only, see [docs/research-vs-runtime.md](docs/research-vs-runtime.md).

## License

MIT — see [LICENSE](LICENSE).
