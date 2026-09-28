
"""
Preprocessing and tokenization utilities matching the SemDrift research model pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch


def extract_docstring_summary(docstring: str) -> str:
    """Extract clean natural language summary from docstrings.

    Identical to the research repository implementation.
    Strips REPL examples (>>>), parameter tables, return specs, etc.
    Keeps only the opening summary sentence(s).

    Parameters:
        docstring: Raw docstring text.

    Returns:
        Summary text string.
    """
    if not docstring:
        return ""

    lines = docstring.strip().split("\n")
    summary_lines = []

    for line in lines:
        stripped = line.strip()

        if (
            not stripped
            or stripped.startswith(">>>")
            or stripped.startswith("...")
            or stripped.startswith("Parameters")
            or stripped.startswith("Returns")
            or stripped.startswith("Examples")
            or stripped.startswith("See Also")
            or stripped.startswith("Notes")
            or stripped.startswith("Raises")
            or stripped.startswith("Warnings")
            or stripped.startswith("References")
        ):
            break

        summary_lines.append(stripped)

    cleaned = " ".join(summary_lines).strip()

    if len(cleaned) >= 10:
        return cleaned

    return lines[0].strip()


def _encode_without_length_warning(
    tokenizer: Any,
    text: str,
) -> List[int]:
    """Tokenize raw text without triggering Hugging Face model-length warnings.

    SemDrift performs its own truncation after tokenization. Therefore, this
    function deliberately obtains the complete token sequence before the
    custom SemDrift truncation logic is applied.

    For Hugging Face fast tokenizers, the underlying `tokenizers` backend is
    used directly. This bypasses the Transformers wrapper's pre-forward
    maximum-length warning.

    Lightweight/mock tokenizers used by the test suite continue to use the
    normal `.encode()` fallback.

    Parameters:
        tokenizer: HuggingFace or compatible tokenizer instance.
        text: Text to tokenize.

    Returns:
        List of integer token IDs.
    """

    # Hugging Face FastTokenizer:
    # Use the public backend_tokenizer rather than tokenizer.encode().
    backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)

    if backend_tokenizer is not None:
        encoding = backend_tokenizer.encode(
            text,
            add_special_tokens=False,
        )
        return list(encoding.ids)

    # Compatibility fallback for mocked/simple tokenizers.
    encoded = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    return list(encoded)


def prepare_joint_tokens(
    doc: str,
    code: str,
    tokenizer: Any,
    max_length: int = 512,
    doc_max_tokens: int = 96,
    truncation_strategy: str = "head_tail",
) -> List[int]:
    """Tokenize a (docstring, code) pair into a single sequence.

    Model B joint format:

        [CLS] doc_tokens [SEP] [SEP] code_tokens [SEP]

    Parameters:
        doc:
            Cleaned docstring text.

        code:
            Source code text.

        tokenizer:
            HuggingFace or compatible tokenizer instance.

        max_length:
            Maximum total token length.

        doc_max_tokens:
            Maximum token budget allocated to the docstring.

        truncation_strategy:
            Code truncation strategy:
            'head_tail', 'head', or 'tail'.

    Returns:
        List of integer token IDs.
    """

    # Tokenize the complete strings first.
    #
    # We intentionally do NOT pass truncation=True or max_length here.
    # SemDrift applies its own research-specific truncation logic below.
    doc_ids = _encode_without_length_warning(
        tokenizer,
        doc,
    )

    code_ids = _encode_without_length_warning(
        tokenizer,
        code,
    )

    # Explicit documentation-token budget.
    doc_ids = doc_ids[:doc_max_tokens]

    # Reserve four special tokens:
    #
    # [CLS] + [SEP] + [SEP] + [SEP]
    #
    # Remaining budget is allocated to code.
    remaining_budget = max_length - len(doc_ids) - 4

    # Resolve mask token ID.
    mask_token_id = getattr(
        tokenizer,
        "mask_token_id",
        None,
    )

    if mask_token_id is None:
        mask_token_id = getattr(
            tokenizer,
            "unk_token_id",
            0,
        )

    # Apply SemDrift's custom code truncation.
    if len(code_ids) > remaining_budget:

        if truncation_strategy == "head_tail":
            # Reserve one token for the mask marker.
            code_budget = remaining_budget - 1

            if code_budget > 0:
                head_len = code_budget // 2
                tail_len = code_budget - head_len

                code_ids = (
                    code_ids[:head_len]
                    + [mask_token_id]
                    + code_ids[-tail_len:]
                )
            else:
                code_ids = [mask_token_id]

        elif truncation_strategy == "tail":
            code_ids = code_ids[-remaining_budget:]

        else:
            # "head" or unknown strategy.
            code_ids = code_ids[:remaining_budget]

    # Resolve special-token IDs.
    cls_tok = getattr(
        tokenizer,
        "cls_token_id",
        None,
    )

    if cls_tok is None:
        cls_tok = 0

    sep_tok = getattr(
        tokenizer,
        "sep_token_id",
        None,
    )

    if sep_tok is None:
        sep_tok = 2

    # Joint encoder format:
    #
    # [CLS] doc [SEP] [SEP] code [SEP]
    combined_ids = (
        [cls_tok]
        + doc_ids
        + [sep_tok, sep_tok]
        + code_ids
        + [sep_tok]
    )

    # Final safety boundary.
    return combined_ids[:max_length]


def collate_joint_batch(
    token_id_sequences: Sequence[List[int]],
    tokenizer: Any,
) -> Dict[str, torch.Tensor]:
    """Pad token sequences and build model input tensors.

    Parameters:
        token_id_sequences:
            Sequence of token ID lists.

        tokenizer:
            Tokenizer providing pad_token_id.

    Returns:
        Dictionary containing:
            - input_ids
            - attention_mask
    """

    if not token_id_sequences:
        return {
            "input_ids": torch.empty(
                (0, 0),
                dtype=torch.long,
            ),
            "attention_mask": torch.empty(
                (0, 0),
                dtype=torch.long,
            ),
        }

    max_batch_len = max(
        len(ids)
        for ids in token_id_sequences
    )

    pad_token_id = getattr(
        tokenizer,
        "pad_token_id",
        None,
    )

    if pad_token_id is None:
        pad_token_id = 0

    padded_input_ids = []
    padded_attention_mask = []

    for ids in token_id_sequences:
        pad_len = max_batch_len - len(ids)

        padded_input_ids.append(
            ids + [pad_token_id] * pad_len
        )

        padded_attention_mask.append(
            [1] * len(ids)
            + [0] * pad_len
        )

    return {
        "input_ids": torch.tensor(
            padded_input_ids,
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            padded_attention_mask,
            dtype=torch.long,
        ),
    }
