# SemDrift — Research vs. Runtime Boundary

This document classifies components from the [SemDrift research repository](https://github.com/Aadhithya-T/SemDrift) and defines which are being ported to the implementation repository.

## Repository Purposes

### Research Repository (`Aadhithya-T/SemDrift`)
- Empirical framework for investigating semantic drift detection
- Benchmarking classical ML, zero-shot, dual-encoder, and joint-encoder approaches
- Dataset curation (V1 synthetic, V2 real-world grounded)
- Training scripts and experiment reproducibility
- Analysis and visualization of results

### Implementation Repository (`SemDrift-CLI`)
- Production-oriented developer tool for detecting semantic drift
- Clean, modular architecture (scanner → parser → model → detection → reporting → CLI)
- Installable Python package with a command-line interface
- Designed for use in real-world development workflows

---

## Component Classification

### 1. Runtime / Implementation (will be ported)

| Research Component | Implementation Module | Notes |
|---|---|---|
| `semdrift/parser/ast_parser.py` | `semdrift.parser` | Core AST parsing logic — will be refactored for the modular architecture |
| `semdrift/parser/docstring_parser.py` | `semdrift.parser` | Docstring extraction — runtime-relevant |
| `semdrift/parser/formatter.py` | `semdrift.parser` | Input formatting for model consumption |
| `semdrift/models/joint_encoder.py` | `semdrift.model` | Model architecture needed for inference (training code will be stripped) |
| `scripts/scan_repo.py` (partial) | `semdrift.scanner`, `semdrift.cli` | Repository traversal and CLI logic — will be decomposed across modules |

### 2. Training-Only (will NOT be ported)

| Component | Reason |
|---|---|
| `scripts/train_dual_encoder.py` | Training script — not needed for inference |
| `scripts/train_joint_encoder.py` | Training script — not needed for inference |
| `scripts/training/` | Training runner scripts |
| `semdrift/models/dual_encoder.py` | Dual-encoder architecture — research showed joint-encoder is superior; may reconsider if multi-model support is added |

### 3. Evaluation-Only (will NOT be ported)

| Component | Reason |
|---|---|
| `scripts/run_tfidf_baseline.py` | Baseline evaluation script |
| `scripts/run_zero_shot_baseline.py` | Zero-shot baseline evaluation |
| `scripts/runners/` | Experiment runner scripts |
| `scripts/analysis/` | Result analysis and visualization |
| `scripts/score_review_priority.py` | Review scoring for research evaluation |
| `tests/test_comparator.py` | Tests for research comparator |
| `tests/test_embedder.py` | Tests for research embedder |
| `tests/test_tfidf_baseline.py` | Tests for research baseline |
| `tests/test_dataset_architecture.py` | Tests for research dataset design |
| `tests/test_dataset_provenance.py` | Tests for research data provenance |
| `tests/test_label_invariants.py` | Tests for research label correctness |
| `tests/test_tamper_rejection.py` | Tests for research dataset integrity |
| `tests/test_v2_updates.py` | Tests for research V2 updates |

### 4. Dataset / Research-Only (will NOT be ported)

| Component | Reason |
|---|---|
| `data/` | All datasets (V1 synthetic, V2 real-world) |
| `experiments/` | Experiment results, checkpoints, locked test sets |
| `notebooks/` | Jupyter notebooks for analysis |
| `documentation/` | Research documentation |
| `archive/` | Historical code archive |
| `BENCHMARK_RESULTS.md` | Research benchmark results |
| `Results - Thunder.md` | GPU benchmark results |
| `scripts/mine_real_drift.py` | Git commit mining for dataset creation |
| `scripts/curate_drift_dataset.py` | Dataset curation |
| `scripts/consolidate_datasets.py` | Dataset consolidation |
| `scripts/generate_filtered_dataset.py` | Dataset filtering |
| `scripts/filter_real_drift_candidates.py` | Drift candidate filtering |
| `scripts/contract_check_candidates.py` | Contract verification for datasets |
| `scripts/auto_process_mined_repos.py` | Automated repo processing for mining |
| `scripts/review_candidates.py` | Manual review tooling |
| `scripts/data_pipeline/` | Data pipeline scripts |
| `scripts/extract_pairs_java.py` | Java pair extraction (research) |
| `scripts/test_java_parser.py` | Java parser testing (research) |
| `semdrift/data/integrity.py` | Dataset integrity verification (SHA256) |
| `semdrift/data/labels.py` | Label definitions for datasets |

### 5. Potentially Reusable (requires refactoring)

| Component | Status | Notes |
|---|---|---|
| `semdrift/parser/universal_parser.py` | **Uncertain** | Multi-language parser — may be useful if language support expands beyond Python |
| `semdrift/parser/contracts.py` | **Uncertain** | Contract-based formatting — tied to dataset generation, but contract concepts may inform runtime detection |
| `semdrift/parser/contract_formatter.py` | **Uncertain** | Contract formatting — same uncertainty as above |
| `semdrift/parser/doc_extractor.py` | **Likely reusable** | Docstring extraction — needs review for overlap with `docstring_parser.py` |
| `semdrift/embedder/embed.py` | **Uncertain** | Embedding generation — the implementation may use the model module directly instead of a separate embedder stage |
| `semdrift/comparator/__init__.py` | **Empty stub** | Only contains an `__init__.py` — no logic to evaluate |
| `semdrift/pipeline.py` | **Architectural reference only** | Monolithic orchestrator — will NOT be copied; its responsibilities are distributed across the modular architecture |
| `config.yaml` | **Reference** | Research config — the implementation will have its own configuration approach |
| `tests/test_parser.py` | **Partially reusable** | Parser tests — some test cases may be adapted for the implementation's parser module |
| `tests/test_universal_parser.py` | **Uncertain** | Universal parser tests — depends on whether `universal_parser.py` is ported |
| `tests/test_p0_fixes.py` | **Uncertain** | Bug fix tests — depends on which fixes apply to the implementation |

---

## Runtime Dependencies

Dependencies that will be required when the implementation matures:

| Dependency | Phase | Purpose |
|---|---|---|
| `torch` | Phase 3 (Model) | PyTorch for model inference |
| `transformers` | Phase 3 (Model) | CodeBERT tokenizer and model loading |
| `rich` | Phase 4 (CLI) | Terminal UI formatting |
| `pyyaml` | TBD | Configuration file parsing (if YAML config is adopted) |

## Research-Only Dependencies (will NOT be added)

| Dependency | Reason |
|---|---|
| `scikit-learn` | Training/evaluation metrics only |
| `scipy` | Statistical tests (McNemar's test) only |
| `pydriller` | Git commit mining for dataset creation |
| `sentence-transformers` | Zero-shot baseline evaluation |
| `astroid` | Research uses stdlib `ast` in practice; `astroid` may not be needed |

---

## Architectural Risks

1. **`scripts/scan_repo.py` mixed concerns**: The research CLI combines argument parsing, terminal rendering, model loading, AST parsing, inference, and result formatting in a single file (~190 lines). The implementation must decompose this into separate modules to maintain the clean architecture.

2. **Model training code in architecture files**: `semdrift/models/joint_encoder.py` contains both the model architecture (needed for inference) and training-specific code (loss functions, training loops). Inference-only wrappers must be extracted carefully.

3. **Checkpoint path coupling**: The research `scan_repo.py` hardcodes checkpoint paths relative to the project root (`experiments/2026-09-07_clean_v2/checkpoints/`). The implementation must support configurable model paths.

4. **Parser/formatter coupling to dataset generation**: Some parser modules (`contracts.py`, `contract_formatter.py`) may be coupled to the dataset generation workflow rather than general-purpose runtime parsing. This needs careful evaluation before porting.

5. **Dual-encoder vs. joint-encoder decision**: The research benchmarks show the joint-encoder outperforms the dual-encoder on synthetic data but both fail on real-world drift. The implementation initially targets the joint-encoder, but this architectural decision may need revisiting.
