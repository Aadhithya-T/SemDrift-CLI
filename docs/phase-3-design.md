# SemDrift — Phase 3 Design: Model & Inference Layer

This document details the runtime model architecture, preprocessing pipeline, inference contract, and research compatibility for Phase 3 of `SemDrift-CLI`.

---

## 1. Overview & Architecture

Phase 3 introduces the runtime model layer that consumes `CodeDocumentPair` records from the Phase 2 AST parser and generates raw `ModelPrediction` outputs.

The architecture directly reproduces the research repository's primary contribution: **Model B Joint-Encoder**.

### Joint Code-Doc Self-Attention
Rather than encoding code and documentation independently (dual-encoder), the joint-encoder concatenates the docstring and function code into a single unified input sequence:
```
[CLS] docstring_tokens [SEP] [SEP] code_tokens [SEP]
```
Bidirectional self-attention in CodeBERT allows every token in the documentation and code to attend directly to one another across all 12 transformer layers.

```
       CodeDocumentPair
              │
              ▼
   [Preprocessing & Tokenization]
              │
              ▼
    [CLS] doc [SEP][SEP] code [SEP]
              │
              ▼
     CodeBERT Backbone (768d)
              │
              ▼
     [CLS] Token Representation
              │
              ▼
     Dropout (p=0.1)
              │
              ▼
     Linear Classifier (768 -> 2)
              │
              ▼
    Logits: [aligned, drifted]
              │
              ▼
       Softmax (dim=-1)
              │
              ▼
      ModelPrediction
```

---

## 2. Preprocessing & Tokenization Contract

### Docstring Summary Extraction
The model layer applies `extract_docstring_summary(docstring)` ported directly from the research repository:
- Strips interactive REPL prompts (`>>>`, `...`).
- Stops before structured section headers (`Parameters`, `Returns`, `Examples`, `See Also`, `Notes`, `Raises`, `Warnings`, `References`).
- Preserves the primary natural language summary sentence(s).

### Joint Sequence Formatting & Budgeting
1. **Special Tokens**: Uses RoBERTa sentence-pair format:
   - `[CLS]` (id 0) at start of sequence.
   - `[SEP] [SEP]` (ids 2, 2) between docstring and code.
   - `[SEP]` (id 2) at end of sequence.
2. **Docstring Budget**: Truncated to `doc_max_tokens` (default 96 tokens).
3. **Code Truncation Strategy**:
   - `remaining_budget = max_length - len(doc_ids) - 4` (accounting for 4 special tokens).
   - `"head_tail"` strategy:
     - `head_len = code_budget // 2`
     - `tail_len = code_budget - head_len`
     - Inserts a single `[MASK]` token (id 3) between the head (function signature, parameters, initial logic) and tail (return statements, termination logic).
   - Guarantees total sequence length does not exceed `max_length` (default 512).

---

## 3. Data Contracts

### `ModelConfig` (`@dataclass(frozen=True)`)
| Field | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | `"microsoft/codebert-base"` | HuggingFace backbone identifier. |
| `max_length` | `int` | `512` | Max total tokens in joint sequence. |
| `doc_max_tokens` | `int` | `96` | Token budget for docstrings. |
| `code_truncation` | `str` | `"head_tail"` | Code truncation strategy. |
| `pooling` | `str` | `"cls"` | `[CLS]` representation pooling. |
| `num_labels` | `int` | `2` | Classification output logits. |
| `dropout` | `float` | `0.1` | Classifier dropout. |
| `device` | `str` | `"cpu"` | Target PyTorch device (`cpu` or `cuda`). |
| `batch_size` | `int` | `16` | Default batch size for batch inference. |
| `clean_docstrings` | `bool` | `True` | Whether to extract docstring summary. |

### `ModelPrediction` (`@dataclass(frozen=True)`)
| Field | Type | Description |
|---|---|---|
| `file_path` | `str` | Source file path of the evaluated function. |
| `qualified_name` | `str` | Hierarchical deterministic function name. |
| `line_number` | `int` | 1-indexed start line of definition. |
| `drift_probability` | `Optional[float]` | Model probability of semantic drift in `[0.0, 1.0]`, or `None` if undocumented. |
| `prediction` | `str` | Predicted class: `"drifted"`, `"aligned"`, or `"undocumented"`. |
| `is_undocumented` | `bool` | `True` if `docstring is None`. |
| `raw_logits` | `Optional[Tuple[float, float]]` | Optional diagnostic unnormalized logits `(aligned_logit, drifted_logit)`. |

---

## 4. Missing Docstring Handling (`docstring is None`)

Functions without docstrings cannot be evaluated by the CodeBERT drift model because semantic drift is defined as divergence between code semantics and documentation semantics.

In `SemDriftModel.predict()`:
1. Functions with `docstring is None` are excluded from transformer batch tokenization and model execution.
2. A `ModelPrediction` is immediately constructed with:
   - `is_undocumented = True`
   - `drift_probability = None`
   - `prediction = "undocumented"`
   - `raw_logits = None`
3. Exact input ordering is strictly maintained in the output list.
4. Final policy decisions (e.g. flagging missing documentation, reporting warnings, or filtering) are deferred to Phase 4 (Detection) and Phase 5 (Reporting).

---

## 5. Research Compatibility & Verification

### Research Inference vs. Runtime Inference

| Component | Research Repo (`SemDrift`) | SemDrift-CLI Runtime (Phase 3) |
|---|---|---|
| **Architecture** | `JointEncoderModel` in `semdrift.models.joint_encoder` | Identical `JointEncoderModel` in `src.semdrift.model.architecture`. |
| **Token Layout** | `[CLS] doc [SEP][SEP] code [SEP]` | Identical `[CLS] doc [SEP][SEP] code [SEP]` in `prepare_joint_tokens`. |
| **Special Tokens** | `cls_tok=0`, `sep_tok=2`, `mask_tok=3` | Identical token mapping via CodeBERT tokenizer. |
| **Doc Preprocessing** | `extract_docstring_summary` | Identical logic and section boundary parsing. |
| **Truncation** | `head_tail` with `[MASK]` bridge | Identical `head_tail` implementation. |
| **Pooling** | `[CLS]` token representation at index 0 | Directly implemented `outputs.last_hidden_state[:, 0, :]`. |
| **Checkpoint Structure** | Raw `state_dict` with 201 tensor keys | Loaded via `model.load_state_dict(state_dict)`. |
| **Orchestration** | Mixed inside `scripts/scan_repo.py` CLI script | Clean `SemDriftModel.from_checkpoint()` and `predict()` API. |

### Checkpoint Compatibility Verification
Direct compatibility with the trained research checkpoint was verified on the actual checkpoint artifact:
- File: `experiments/2026-09-07_clean_v2/checkpoints/joint_encoder_checkpoint.pt`
- Encoder keys: 199 `encoder.*` keys matching `AutoModel.from_pretrained("microsoft/codebert-base")`
- Classifier weights: `classifier.weight` `torch.Size([2, 768])`, `classifier.bias` `torch.Size([2])`
- Loading result: `<All keys matched successfully>` (201/201 keys matched, 0 missing, 0 unexpected).

---

## 6. Offline Testing Design

Per architectural constraints, the automated test suite does NOT download CodeBERT weights from Hugging Face:
- Unit tests use `DummyBackbone(hidden_size=16)` and `MockTokenizer` to test the complete inference pipeline:
  - `state_dict` persistence and loading
  - Batching consistency across batch sizes
  - Ordering preservation between mixed documented and undocumented pairs
  - CPU device placement
  - Structured error raising on corrupted checkpoints or inference failures
- This ensures tests run in seconds without external network dependencies.

---

## 7. What is Deliberately Excluded

The following research and future components were deliberately excluded from Phase 3:
- **Training Code**: `FocalLoss`, optimizers, LR schedulers, training loops, dataset synthesis.
- **Evaluation Code**: Precision/recall/macro-F1 calculation, confusion matrices, ablation runners.
- **Detection Logic**: Configurable drift thresholds, severity levels, drift classification rules (belongs to Phase 4).
- **Reporting & CLI**: Rich progress bars, colored terminal dashboards, markdown/JSON export, CLI command parsers (belongs to Phase 5 & 6).
- **Model Binaries**: No `.pt`, `.bin`, or `.safetensors` files committed to Git.
