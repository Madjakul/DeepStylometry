#!/usr/bin/env python3
# deep_stylometry/utils/text_stats.py
"""Shared text-statistics helpers for dataset analysis and retrieval inspection.

All functions are importable without loading ``datasets`` or ``transformers``
at module level; heavy imports are deferred into function bodies.

Public API::

    from deep_stylometry.utils.text_stats import (
        _jaccard, _to_frozenset, _flatten_domain,
        build_punct_token_id_set,
        span_token_count, fertility,
        punct_density_from_ids, sentence_length_std,
    )
"""

from __future__ import annotations

import json
import logging
import re
import string
from typing import Any, FrozenSet, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Module-level flag so the JSON-decode warning fires only once per process run.
_JSON_WARN_ISSUED: bool = False


# ---------------------------------------------------------------------------
# Set-comparison helpers (moved from retrieval_inspection.py)
# ---------------------------------------------------------------------------

def _jaccard(a: Set, b: Set) -> float:
    """Jaccard similarity between two sets.

    Args:
        a: First set.
        b: Second set.

    Returns:
        Jaccard similarity in [0, 1]; 0.0 if both sets are empty.
    """
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _to_frozenset(raw: Any) -> FrozenSet[str]:
    """Convert a raw author-id field to a frozenset of strings.

    Handles the following formats encountered in HALvest-Contrastive:
    - ``None`` → empty frozenset
    - ``list`` / ``tuple`` → frozenset of ``str(elem)``
    - JSON-encoded list string (``"[\"id1\", \"id2\"]"``) → frozenset of elements
      (a warning is logged on the first occurrence)
    - bare string → singleton frozenset

    Args:
        raw: Value from ``pos_authorids`` / ``neg_authorids`` / ``query_authorids``.

    Returns:
        frozenset of author-id strings; empty frozenset if absent.
    """
    global _JSON_WARN_ISSUED
    if raw is None:
        return frozenset()
    if isinstance(raw, (list, tuple)):
        return frozenset(str(a) for a in raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                if not _JSON_WARN_ISSUED:
                    logger.warning(
                        "JSON-encoded author list encountered in _to_frozenset; "
                        "decoding automatically. Further occurrences silently handled."
                    )
                    _JSON_WARN_ISSUED = True
                return frozenset(str(a) for a in parsed)
        except (json.JSONDecodeError, ValueError):
            pass
        return frozenset([raw])
    return frozenset([str(raw)])


def _flatten_domain(val: Any) -> str:
    """Normalise a domain value that may be a str or a list of str.

    Args:
        val: Raw domain value from a HuggingFace row.

    Returns:
        A single string domain label, empty string if absent.
    """
    if isinstance(val, list):
        return val[0] if val else ""
    return str(val) if val is not None else ""


# ---------------------------------------------------------------------------
# Tokenisation-based helpers
# ---------------------------------------------------------------------------

def build_punct_token_id_set(tokenizer_name: str) -> Set[int]:
    """Build the punctuation token-ID set matching ``LateInteraction``'s skip_list logic.

    This mirrors the exact code path in ``LateInteraction.__init__``, decoding each
    vocab token and keeping those whose decoded form consists entirely of characters
    in ``string.punctuation | {" "}``.  The resulting set is what MaxSim scoring
    ignores when ``skip_list=True``, so using it as the metric definition ensures
    the statistics measure exactly the tokens the model skips.

    ``transformers`` is imported inside the function so this module remains
    importable on CI without transformers installed.

    Args:
        tokenizer_name: HuggingFace tokenizer identifier.

    Returns:
        Set of integer token IDs classified as punctuation / whitespace.
    """
    from tqdm import tqdm
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    punc_chars = set(string.punctuation) | {" "}
    punc_token_ids: Set[int] = set()
    for token_str, token_id in tqdm(
        tokenizer.get_vocab().items(),
        desc="Building punct token set",
        leave=False,
    ):
        decoded = tokenizer.convert_tokens_to_string([token_str])
        if decoded and all(c in punc_chars for c in decoded):
            punc_token_ids.add(token_id)
    logger.info("Punctuation token set: %d tokens.", len(punc_token_ids))
    return punc_token_ids


def span_token_count(
    text: str,
    tokenizer: Any,
    max_length: int,
) -> Tuple[int, int]:
    """Count tokens pre- and post-truncation for a single text.

    Args:
        text: Input text string.
        tokenizer: HuggingFace tokenizer (already loaded, passed in).
        max_length: Truncation ceiling.

    Returns:
        ``(n_pre_trunc, n_post_trunc)`` where ``n_post_trunc = min(n_pre, max_length)``.
    """
    pre = tokenizer(text, add_special_tokens=True, truncation=False)
    n_pre = len(pre["input_ids"])
    return n_pre, min(n_pre, max_length)


def fertility(text: str, tokenizer: Any) -> Optional[float]:
    """Sub-words per whitespace word, pre-truncation.

    Args:
        text: Input text.
        tokenizer: HuggingFace tokenizer.

    Returns:
        Fertility ratio (≥ 1.0 for ModernBERT on English academic text,
        expected range 1.3–1.5), or ``None`` if the text has no words.
    """
    words = text.split()
    if not words:
        return None
    sub_ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    return len(sub_ids) / len(words)


def punct_density_from_ids(
    token_ids: List[int],
    punct_set: Set[int],
    special_ids: Set[int],
) -> float:
    """Fraction of non-special tokens whose ID is in the punctuation set.

    Mirrors what ``LateInteraction`` masks during MaxSim scoring when
    ``skip_list=True``, so this metric measures exactly what the model ignores.

    Args:
        token_ids: Token IDs of the text (typically post-truncation).
        punct_set: Set of token IDs classified as punctuation / whitespace.
        special_ids: Set of special-token IDs excluded from the denominator.

    Returns:
        Punctuation density in [0, 1]; 0.0 if there are no non-special tokens.
    """
    non_special = [t for t in token_ids if t not in special_ids]
    if not non_special:
        return 0.0
    return sum(1 for t in non_special if t in punct_set) / len(non_special)


def sentence_length_std(text: str) -> Tuple[float, bool]:
    """Standard deviation of per-sentence word counts within a span.

    Sentences are split on ``(?<=[.!?])\\s+``.  This regex is deliberately
    imprecise (misses "et al.", "i.e.", "Fig. 1") but the bias is consistent
    across all splits, so relative orderings across base-k and PAN19 are
    preserved even if absolute values are slightly inflated.

    Args:
        text: Input text.

    Returns:
        ``(std, is_singleton)`` where ``is_singleton=True`` if fewer than two
        non-empty sentences were found (std is returned as 0.0 in that case).
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s for s in sentences if s.strip()]
    if len(sentences) < 2:
        return 0.0, True
    word_counts = [len(s.split()) for s in sentences]
    return float(np.std(word_counts, ddof=1)), False
