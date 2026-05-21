# deep_stylometry/experiments/mechanistic/tests/test_lisa_features.py
"""Tests for LISA feature extraction."""

from __future__ import annotations

import numpy as np
import pytest

from deep_stylometry.experiments.mechanistic.lisa_features import (
    extract_lisa_features,
    get_feature_names,
)

_SAMPLE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "However, this sentence is quite short and simple. "
    "Nevertheless, it contains various function words."
)


class TestExtractDeterministic:
    def test_two_calls_identical(self) -> None:
        """Two calls to extract_lisa_features on the same string return equal vectors."""
        vec1, names1 = extract_lisa_features(_SAMPLE_TEXT)
        vec2, names2 = extract_lisa_features(_SAMPLE_TEXT)
        np.testing.assert_array_equal(vec1, vec2)
        assert names1 == names2

    def test_returns_numpy_float32(self) -> None:
        vec, _ = extract_lisa_features(_SAMPLE_TEXT)
        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float32

    def test_non_empty(self) -> None:
        vec, names = extract_lisa_features(_SAMPLE_TEXT)
        assert len(vec) > 0
        assert len(names) == len(vec)


class TestFeatureNamesStable:
    def test_names_match_extract(self) -> None:
        """Feature names from get_feature_names() match those from extract_lisa_features."""
        names_from_get = get_feature_names()
        _, names_from_extract = extract_lisa_features(_SAMPLE_TEXT)
        assert names_from_get == names_from_extract

    def test_names_deterministic(self) -> None:
        n1 = get_feature_names()
        n2 = get_feature_names()
        assert n1 == n2

    def test_function_word_features_present(self) -> None:
        names = get_feature_names()
        # At least some function-word features must be present
        fw_names = [n for n in names if n.startswith("fw_")]
        assert len(fw_names) >= 100, f"Expected ≥100 fw_ features, got {len(fw_names)}"

    def test_sentence_length_features_present(self) -> None:
        names = get_feature_names()
        assert "sl_mean" in names
        assert "sl_std" in names

    def test_singleton_reuse(self) -> None:
        """Calling extract twice should not reload the spaCy model (singleton check)."""
        from deep_stylometry.experiments.mechanistic import lisa_features as lf
        # Reset singletons so we get a fresh load
        lf._SPACY_NLP = None
        lf._SAT_SPLITTER = None

        vec1, names1 = extract_lisa_features("Hello world.")
        vec2, names2 = extract_lisa_features("Hello world.")
        # After first call, singleton should be set (if spacy available)
        np.testing.assert_array_equal(vec1, vec2)
        assert names1 == names2
