"""
Unit tests for semdrift.model.preprocessing.
"""

import pytest

from semdrift.model.preprocessing import (
    collate_joint_batch,
    extract_docstring_summary,
    prepare_joint_tokens,
)


class MockTokenizer:
    """Deterministic offline mock tokenizer for unit testing tokenization logic."""

    def __init__(self):
        self.cls_token_id = 0
        self.pad_token_id = 1
        self.sep_token_id = 2
        self.mask_token_id = 3
        self.unk_token_id = 4

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # Maps each word or character to an integer ID deterministically
        if not text:
            return []
        words = text.split()
        return [10 + (hash(w) % 1000) for w in words]


class TestDocstringSummaryExtraction:
    """Verify extract_docstring_summary behavior matches research repo."""

    def test_empty_docstring(self):
        assert extract_docstring_summary("") == ""
        assert extract_docstring_summary("   ") == ""

    def test_single_line_docstring(self):
        doc = "Calculate Euclidean distance between two vectors."
        assert extract_docstring_summary(doc) == doc

    def test_strips_parameters_section(self):
        doc = """Calculate the sum of elements.

Parameters
----------
a : int
    First number.
b : int
    Second number.
"""
        assert extract_docstring_summary(doc) == "Calculate the sum of elements."

    def test_strips_returns_section(self):
        doc = """Fetch remote records from the database.

Returns:
    List of records.
"""
        assert extract_docstring_summary(doc) == "Fetch remote records from the database."

    def test_strips_repl_examples(self):
        doc = """Compute square root.

>>> sqrt(4)
2
"""
        assert extract_docstring_summary(doc) == "Compute square root."

    def test_multiline_summary(self):
        doc = """Compute the inverse fast Fourier transform.
Applies Cooley-Tukey FFT algorithm along axis.

Parameters:
    data: input array.
"""
        expected = "Compute the inverse fast Fourier transform. Applies Cooley-Tukey FFT algorithm along axis."
        assert extract_docstring_summary(doc) == expected


class TestJointTokenPreparation:
    """Verify prepare_joint_tokens builds [CLS] doc [SEP][SEP] code [SEP] correctly."""

    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer()

    def test_joint_structure_and_special_tokens(self, tokenizer):
        doc = "Summary line"
        code = "return x + 1"
        tokens = prepare_joint_tokens(
            doc=doc,
            code=code,
            tokenizer=tokenizer,
            max_length=512,
            doc_max_tokens=96,
            truncation_strategy="head_tail",
        )

        assert tokens[0] == tokenizer.cls_token_id  # [CLS]
        # [SEP] [SEP] sequence
        doc_tokens = tokenizer.encode(doc, add_special_tokens=False)
        assert tokens[1 : 1 + len(doc_tokens)] == doc_tokens
        sep_idx = 1 + len(doc_tokens)
        assert tokens[sep_idx] == tokenizer.sep_token_id
        assert tokens[sep_idx + 1] == tokenizer.sep_token_id
        assert tokens[-1] == tokenizer.sep_token_id  # trailing [SEP]

    def test_doc_max_tokens_budget(self, tokenizer):
        long_doc = "word " * 200
        code = "x = 1"
        tokens = prepare_joint_tokens(
            doc=long_doc,
            code=code,
            tokenizer=tokenizer,
            max_length=512,
            doc_max_tokens=20,
        )

        # After [CLS], there should be at most 20 doc tokens before [SEP] [SEP]
        sep1_idx = tokens.index(tokenizer.sep_token_id)
        doc_count = sep1_idx - 1
        assert doc_count == 20

    def test_head_tail_code_truncation(self, tokenizer):
        doc = "short doc"
        long_code = "word " * 100
        # Budget for code: max_length(30) - doc_len(2) - 4 special = 24
        tokens = prepare_joint_tokens(
            doc=doc,
            code=long_code,
            tokenizer=tokenizer,
            max_length=30,
            doc_max_tokens=10,
            truncation_strategy="head_tail",
        )
        assert len(tokens) <= 30
        assert tokenizer.mask_token_id in tokens

    def test_head_truncation(self, tokenizer):
        doc = "short doc"
        long_code = "word " * 100
        tokens = prepare_joint_tokens(
            doc=doc,
            code=long_code,
            tokenizer=tokenizer,
            max_length=30,
            doc_max_tokens=10,
            truncation_strategy="head",
        )
        assert len(tokens) <= 30
        assert tokenizer.mask_token_id not in tokens

    def test_tail_truncation(self, tokenizer):
        doc = "short doc"
        long_code = "word " * 100
        tokens = prepare_joint_tokens(
            doc=doc,
            code=long_code,
            tokenizer=tokenizer,
            max_length=30,
            doc_max_tokens=10,
            truncation_strategy="tail",
        )
        assert len(tokens) <= 30
        assert tokenizer.mask_token_id not in tokens


class TestCollateBatch:
    """Verify collate_joint_batch padding and attention mask generation."""

    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer()

    def test_padding_and_attention_mask(self, tokenizer):
        seq1 = [0, 10, 11, 2, 2, 20, 2]  # len 7
        seq2 = [0, 10, 2, 2, 20, 21, 22, 23, 2]  # len 9

        batch = collate_joint_batch([seq1, seq2], tokenizer)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        assert input_ids.shape == (2, 9)
        assert attention_mask.shape == (2, 9)

        # seq1 padded with 2 pad tokens
        assert input_ids[0, 7].item() == tokenizer.pad_token_id
        assert input_ids[0, 8].item() == tokenizer.pad_token_id
        assert attention_mask[0].tolist() == [1, 1, 1, 1, 1, 1, 1, 0, 0]
        # seq2 unpadded
        assert attention_mask[1].tolist() == [1] * 9

    def test_empty_batch(self, tokenizer):
        batch = collate_joint_batch([], tokenizer)
        assert batch["input_ids"].shape == (0, 0)
        assert batch["attention_mask"].shape == (0, 0)
