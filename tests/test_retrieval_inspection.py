# tests/test_retrieval_inspection.py
"""Unit tests for helpers shared via deep_stylometry.utils.text_stats.

The old retrieval_inspection.py has been replaced (doc_to_meta and
find_true_positive_rank are removed; _jaccard is now in text_stats).
Comprehensive new tests live in tests/experiments/test_retrieval_inspection.py.

Run::

    python -m pytest tests/test_retrieval_inspection.py -v --tb=short
"""

from __future__ import annotations

from typing import FrozenSet

import numpy as np
import pytest

from deep_stylometry.utils.text_stats import _jaccard, _to_frozenset


class TestJaccard:
    """_jaccard must return correct similarity for various inputs."""

    def test_partial_overlap(self) -> None:
        """{'a','b'} ∩ {'b','c'} = {'b'}, union = {'a','b','c'} → 1/3."""
        a: FrozenSet[str] = frozenset({"a", "b"})
        b: FrozenSet[str] = frozenset({"b", "c"})
        result = _jaccard(a, b)
        assert abs(result - 1 / 3) < 1e-9, f"Expected 1/3, got {result}"

    def test_identical_sets(self) -> None:
        """Identical non-empty sets → Jaccard = 1.0."""
        a: FrozenSet[str] = frozenset({"a"})
        assert _jaccard(a, a) == pytest.approx(1.0)

    def test_disjoint_sets(self) -> None:
        """Disjoint sets → Jaccard = 0.0."""
        a: FrozenSet[str] = frozenset({"a"})
        b: FrozenSet[str] = frozenset({"b"})
        assert _jaccard(a, b) == pytest.approx(0.0)

    def test_both_empty(self) -> None:
        """Both empty → Jaccard = 0.0 (guard against division by zero)."""
        assert _jaccard(frozenset(), frozenset()) == pytest.approx(0.0)

    def test_one_empty(self) -> None:
        """One empty set → Jaccard = 0.0."""
        a: FrozenSet[str] = frozenset({"x"})
        assert _jaccard(a, frozenset()) == pytest.approx(0.0)
        assert _jaccard(frozenset(), a) == pytest.approx(0.0)

    def test_larger_sets(self) -> None:
        """{'a','b','c'} vs {'b','c','d'}: intersection=2, union=4 → 0.5."""
        a: FrozenSet[str] = frozenset({"a", "b", "c"})
        b: FrozenSet[str] = frozenset({"b", "c", "d"})
        assert _jaccard(a, b) == pytest.approx(0.5)


class TestToFrozenset:
    """_to_frozenset must handle all HALvest author-id formats."""

    def test_none_returns_empty(self) -> None:
        assert _to_frozenset(None) == frozenset()

    def test_list_of_strings(self) -> None:
        assert _to_frozenset(["a1", "a2"]) == frozenset({"a1", "a2"})

    def test_tuple_of_strings(self) -> None:
        assert _to_frozenset(("a1", "a2")) == frozenset({"a1", "a2"})

    def test_bare_string(self) -> None:
        assert _to_frozenset("a1") == frozenset({"a1"})

    def test_json_encoded_list(self) -> None:
        """JSON-encoded list string → correctly decoded frozenset."""
        raw = '["id1", "id2"]'
        result = _to_frozenset(raw)
        assert result == frozenset({"id1", "id2"})

    def test_list_with_integers(self) -> None:
        """Integer elements are converted to strings."""
        assert _to_frozenset([1, 2, 3]) == frozenset({"1", "2", "3"})
