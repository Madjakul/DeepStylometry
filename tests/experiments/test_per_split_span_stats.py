# tests/experiments/test_per_split_span_stats.py
"""Unit tests for per-split span statistics helpers.

Tests exercise only the pure-Python helpers in ``deep_stylometry.utils.text_stats``
using mock tokenizers and stdlib.  No HuggingFace downloads, no network, no GPU.

Run::

    python -m pytest tests/experiments/test_per_split_span_stats.py -v
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pytest

from deep_stylometry.experiments.dataset_statistics import _compute_span_stats_for_texts
from deep_stylometry.utils.text_stats import (
    fertility,
    punct_density_from_ids,
    sentence_length_std,
    span_token_count,
)


# ---------------------------------------------------------------------------
# Mock tokenizer helpers
# ---------------------------------------------------------------------------

class _FixedTokenizer:
    """Tokenizer that returns a fixed list of token IDs regardless of text.

    Useful for testing helpers that take a tokenizer but whose correctness
    depends only on the token count or token-ID values, not on linguistic content.
    """

    def __init__(
        self,
        fixed_ids: List[int],
        fixed_ids_no_special: Optional[List[int]] = None,
    ) -> None:
        self._ids = fixed_ids
        self._no_special = fixed_ids_no_special if fixed_ids_no_special is not None else fixed_ids
        self.pad_token_id = 0

    def __call__(
        self,
        text: Any,
        add_special_tokens: bool = True,
        truncation: bool = False,
        padding: Any = False,
        max_length: int = 512,
        return_tensors: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        ids = self._ids if add_special_tokens else self._no_special
        # Handle list input (batched call).
        if isinstance(text, list):
            return {"input_ids": [list(ids)] * len(text)}
        return {"input_ids": list(ids)}


class _WordCountTokenizer:
    """Tokenizer that produces exactly 2 sub-tokens per whitespace word.

    fertility = n_subwords / n_words = 2.0 always (for non-empty texts).
    Prepends CLS (token_id=1) and appends SEP (token_id=2) when
    add_special_tokens=True.
    """

    cls_token_id = 1
    sep_token_id = 2
    pad_token_id = 0

    def __call__(
        self,
        text: Any,
        add_special_tokens: bool = True,
        truncation: bool = False,
        padding: Any = False,
        max_length: int = 512,
        return_tensors: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if isinstance(text, list):
            return {
                "input_ids": [self._encode_single(t, add_special_tokens) for t in text]
            }
        return {"input_ids": self._encode_single(text, add_special_tokens)}

    def _encode_single(self, text: str, add_special: bool) -> List[int]:
        words = text.split()
        # 2 sub-tokens per word using IDs 100, 101, 102, 103, …
        sub = []
        for i, _ in enumerate(words):
            sub.extend([100 + i * 2, 101 + i * 2])
        if add_special:
            return [self.cls_token_id] + sub + [self.sep_token_id]
        return sub


# ---------------------------------------------------------------------------
# Tests: fertility
# ---------------------------------------------------------------------------

class TestFertility:
    """fertility() must return correct sub-words-per-word ratios."""

    def test_two_subwords_per_word(self) -> None:
        """With _WordCountTokenizer, 4-word text → fertility = 2.0."""
        tok = _WordCountTokenizer()
        text = "This is a test"  # 4 words
        result = fertility(text, tok)
        # 4 words × 2 sub-tokens = 8 sub-tokens (no specials when add_special_tokens=False)
        assert result == pytest.approx(2.0), f"Expected 2.0, got {result}"

    def test_single_word(self) -> None:
        """Single-word text → fertility = 2.0 (1 word, 2 sub-tokens)."""
        tok = _WordCountTokenizer()
        result = fertility("hello", tok)
        assert result == pytest.approx(2.0)

    def test_empty_text_returns_none(self) -> None:
        """Empty text → None (avoids division by zero)."""
        tok = _WordCountTokenizer()
        assert fertility("", tok) is None
        assert fertility("   ", tok) is None

    def test_fixed_subword_count(self) -> None:
        """With a fixed 6-subtoken tokenizer and a 3-word text → fertility = 2.0."""
        # Tokenizer returns 6 sub-tokens (no specials) regardless of text.
        tok = _FixedTokenizer(fixed_ids=[1, 10, 11, 12, 13, 14, 15, 2], fixed_ids_no_special=[10, 11, 12, 13, 14, 15])
        text = "one two three"  # 3 words
        result = fertility(text, tok)
        assert result == pytest.approx(6 / 3)


# ---------------------------------------------------------------------------
# Tests: punct_density_from_ids
# ---------------------------------------------------------------------------

class TestPunctDensity:
    """punct_density_from_ids() must respect the punct_set and special_ids."""

    def test_empty_punct_set_returns_zero(self) -> None:
        """No punctuation tokens → density = 0.0 regardless of input."""
        token_ids = [1, 100, 101, 102, 2]
        special_ids = {1, 2}
        punct_set: Set[int] = set()
        result = punct_density_from_ids(token_ids, punct_set, special_ids)
        assert result == pytest.approx(0.0)

    def test_all_tokens_in_punct_set(self) -> None:
        """All non-special tokens are punctuation → density = 1.0."""
        token_ids = [1, 50, 51, 52, 2]
        special_ids = {1, 2}
        punct_set = {50, 51, 52}
        result = punct_density_from_ids(token_ids, punct_set, special_ids)
        assert result == pytest.approx(1.0)

    def test_half_tokens_in_punct_set(self) -> None:
        """Half non-special tokens are punctuation → density = 0.5."""
        token_ids = [1, 50, 100, 51, 101, 2]
        special_ids = {1, 2}
        punct_set = {50, 51}
        result = punct_density_from_ids(token_ids, punct_set, special_ids)
        # Non-special: 50, 100, 51, 101 → 4 tokens; 2 in punct_set
        assert result == pytest.approx(0.5)

    def test_only_special_tokens_returns_zero(self) -> None:
        """All tokens are special → density = 0.0 (denominator guard)."""
        token_ids = [1, 2, 3]
        special_ids = {1, 2, 3}
        punct_set = {1, 2, 3}
        result = punct_density_from_ids(token_ids, punct_set, special_ids)
        assert result == pytest.approx(0.0)

    def test_empty_token_list_returns_zero(self) -> None:
        """Empty token list → density = 0.0."""
        result = punct_density_from_ids([], {50}, {1, 2})
        assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests: sentence_length_std (sentence splitter)
# ---------------------------------------------------------------------------

class TestSentenceLengthStd:
    """sentence_length_std() must split sentences correctly and compute std."""

    def test_five_sentence_string(self) -> None:
        """Five clean sentences → is_singleton=False, std computed from 5 values."""
        text = (
            "The model achieves good results. "
            "We compare against three baselines. "
            "The first baseline is BM25. "
            "The second baseline is a fine-tuned BERT model. "
            "All baselines underperform our approach."
        )
        std, is_singleton = sentence_length_std(text)
        assert not is_singleton, "Expected 5 sentences, is_singleton should be False."
        # Manually: word counts = [5, 6, 7, 9, 6]
        # numpy std ddof=1 ≈ 1.517
        assert std > 0.0

    def test_et_al_false_positive(self) -> None:
        """The regex will incorrectly split at 'et al.' — this is documented behavior.

        The sentence 'Smith et al. (2021) found that X.' will be split into
        'Smith et al.' and '(2021) found that X.' by the regex.  This is a known
        limitation of the simple regex split; the absolute value is biased but the
        relative ordering across splits is preserved.
        """
        # Known miss: "et al." triggers a split.
        text = "Smith et al. (2021) found that X. Jones et al. (2022) confirmed this."
        std, is_singleton = sentence_length_std(text)
        # 4 'sentences' after split: "Smith et al.", "(2021) found that X",
        # "Jones et al.", "(2022) confirmed this."  (approximately)
        # We just assert it runs without error and produces a non-negative std.
        assert std >= 0.0

    def test_single_sentence_returns_zero_and_singleton_flag(self) -> None:
        """Single sentence → std=0.0 and is_singleton=True (no division by zero)."""
        text = "This is a single sentence with no terminal punctuation that triggers splits"
        std, is_singleton = sentence_length_std(text)
        assert is_singleton is True
        assert std == pytest.approx(0.0)

    def test_single_sentence_with_period_no_space_after(self) -> None:
        """Single sentence ending with period but no trailing whitespace → singleton."""
        text = "One sentence."
        std, is_singleton = sentence_length_std(text)
        assert is_singleton is True
        assert std == pytest.approx(0.0)

    def test_two_sentences_equal_length(self) -> None:
        """Two equal-length sentences → std = 0.0, is_singleton=False."""
        text = "Hello world. Hello world."
        std, is_singleton = sentence_length_std(text)
        assert not is_singleton
        assert std == pytest.approx(0.0)

    def test_two_sentences_different_length(self) -> None:
        """Two sentences of different length → std > 0."""
        text = "Short sentence. Much longer sentence with more words here."
        std, is_singleton = sentence_length_std(text)
        assert not is_singleton
        assert std > 0.0


# ---------------------------------------------------------------------------
# Tests: span_token_count
# ---------------------------------------------------------------------------

class TestSpanTokenCount:
    """span_token_count() must return correct pre- and post-truncation counts."""

    def test_no_truncation(self) -> None:
        """Text shorter than max_length → pre == post."""
        tok = _FixedTokenizer(fixed_ids=[1, 100, 101, 2])  # 4 tokens
        n_pre, n_post = span_token_count("short text", tok, max_length=512)
        assert n_pre == 4
        assert n_post == 4

    def test_truncation_applied(self) -> None:
        """Text with 10 tokens, max_length=5 → post = 5."""
        tok = _FixedTokenizer(fixed_ids=list(range(10)))
        n_pre, n_post = span_token_count("some text", tok, max_length=5)
        assert n_pre == 10
        assert n_post == 5

    def test_exact_boundary(self) -> None:
        """Exactly max_length tokens → pre == post == max_length."""
        tok = _FixedTokenizer(fixed_ids=list(range(512)))
        n_pre, n_post = span_token_count("text", tok, max_length=512)
        assert n_pre == 512
        assert n_post == 512


# ---------------------------------------------------------------------------
# Mock tokenizer with decode support (for _compute_span_stats_for_texts tests)
# ---------------------------------------------------------------------------

class _IdentityDecodeTokenizer:
    """Tokenizer that returns fixed IDs and decodes by returning a stored text.

    Simulates a tokenizer whose decode output equals the original input text
    (i.e. the post-truncation text is exactly the same as the raw text, which
    is the case when no truncation occurs).
    """

    cls_token_id = 1
    sep_token_id = 2
    pad_token_id = 0

    def __init__(self, fixed_ids: List[int], decode_text: str) -> None:
        self._ids = fixed_ids
        self._decode_text = decode_text

    def __call__(
        self,
        text: Any,
        add_special_tokens: bool = True,
        truncation: bool = False,
        padding: Any = False,
        max_length: int = 512,
        **kwargs,
    ) -> Dict[str, Any]:
        ids = list(self._ids)
        if truncation and len(ids) > max_length:
            ids = ids[:max_length]
        if isinstance(text, list):
            return {"input_ids": [list(ids)] * len(text)}
        return {"input_ids": list(ids)}

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        return self._decode_text


# ---------------------------------------------------------------------------
# Tests: sentences_per_span and tokens_per_sentence via _compute_span_stats_for_texts
# ---------------------------------------------------------------------------

class TestSentencesPerSpan:
    """sentences_per_span and tokens_per_sentence keys must appear in output."""

    def test_five_sentence_text(self) -> None:
        """5-sentence text: sentences_per_span=5, tokens_per_sentence=total/5."""
        five_sent_text = (
            "The first sentence ends here. "
            "The second sentence follows. "
            "Here comes the third sentence. "
            "A fourth sentence appears now. "
            "The fifth and final sentence."
        )
        # 50 token IDs: CLS + 48 content tokens + SEP
        fixed_ids = [1] + list(range(100, 148)) + [2]  # 50 tokens
        tok = _IdentityDecodeTokenizer(fixed_ids=fixed_ids, decode_text=five_sent_text)

        result = _compute_span_stats_for_texts(
            texts=[five_sent_text],
            tokenizer=tok,
            max_length=512,
            punct_set=set(),
            special_ids={1, 2},
            label="test_5sent",
        )

        assert result["sentences_per_span_mean"] == pytest.approx(5.0), (
            f"Expected 5.0 sentences, got {result['sentences_per_span_mean']}"
        )
        expected_tps = 50 / 5
        assert result["tokens_per_sentence_mean"] == pytest.approx(expected_tps), (
            f"Expected tokens_per_sentence={expected_tps}, "
            f"got {result['tokens_per_sentence_mean']}"
        )
        assert result["n_zero_sentence_samples"] == 0

    def test_single_sentence_text(self) -> None:
        """Single sentence: sentences_per_span=1, tokens_per_sentence=total, no div-by-zero."""
        single_sent_text = "One single sentence with no terminal punctuation"
        # 10 token IDs
        fixed_ids = [1] + list(range(100, 108)) + [2]  # 10 tokens
        tok = _IdentityDecodeTokenizer(fixed_ids=fixed_ids, decode_text=single_sent_text)

        result = _compute_span_stats_for_texts(
            texts=[single_sent_text],
            tokenizer=tok,
            max_length=512,
            punct_set=set(),
            special_ids={1, 2},
            label="test_1sent",
        )

        # The regex won't split a sentence with no terminal punctuation, so n_sents=1.
        assert result["sentences_per_span_mean"] == pytest.approx(1.0), (
            f"Expected 1.0, got {result['sentences_per_span_mean']}"
        )
        assert result["tokens_per_sentence_mean"] == pytest.approx(10.0), (
            f"Expected 10.0, got {result['tokens_per_sentence_mean']}"
        )
        assert result["n_zero_sentence_samples"] == 0
        # std over a single sample should be 0.0 (not NaN, not an error)
        assert result["sentences_per_span_std"] == pytest.approx(0.0)
