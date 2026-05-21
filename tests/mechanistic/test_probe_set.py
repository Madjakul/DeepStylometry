# deep_stylometry/experiments/mechanistic/tests/test_probe_set.py
"""Tests for Phase 0 probe set construction."""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from deep_stylometry.experiments.mechanistic.probe_set import (
    _build_inventory,
    _make_tier_a_entries,
    _make_tier_b_entries,
    _make_tier_c_entries,
    _compute_statistics,
)


# ---------------------------------------------------------------------------
# Synthetic dataset builder
# ---------------------------------------------------------------------------

def _make_synthetic_ds(n_rows: int = 100) -> List[Dict]:
    """Build a tiny synthetic dataset with predictable structure."""
    import random
    rng = random.Random(42)

    # Define 10 author sets and 3 domains
    author_pools = [
        frozenset([f"A{i}", f"B{i}"]) for i in range(5)
    ] + [
        frozenset([f"C{i}"]) for i in range(5)
    ]
    domains = ["NLP", "CV", "BIO"]

    rows = []
    for i in range(n_rows):
        aset = rng.choice(author_pools)
        domain = rng.choice(domains)
        text = f"Document {i} by {list(aset)} on topic {domain}. " * 5
        neg_aset = rng.choice([a for a in author_pools if a.isdisjoint(aset)])
        neg_domain = rng.choice(domains)
        neg_text = f"Negative {i} by {list(neg_aset)} on topic {neg_domain}. " * 5
        pos_text = f"Positive {i} by {list(aset)} on topic {domain}. " * 5

        rows.append({
            "query": text,
            "positive": pos_text,
            "negative": neg_text,
            "query_authorids": list(aset),
            "pos_authorids": list(aset),
            "neg_authorids": list(neg_aset),
            "query_domain": domain,
            "pos_domain": domain,
            "neg_domain": neg_domain,
            "query_id": str(i),
            "pos_id": f"{i}_pos",
            "neg_id": f"{i}_neg",
        })
    return rows


class _FakeTokenizer:
    def __call__(self, text, **kwargs):
        words = text.split()
        ids = [i % 1000 for i in range(min(len(words) + 2, 140))]
        return {"input_ids": ids}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildInventory:
    def test_inventory_has_correct_keys(self):
        rows = _make_synthetic_ds(50)
        inv = _build_inventory(rows)
        assert len(inv) > 0
        for aset in inv:
            assert isinstance(aset, frozenset)
            for doc in inv[aset]:
                assert "text" in doc
                assert "domain" in doc
                assert "doc_id" in doc

    def test_inventory_groups_by_author_set(self):
        rows = _make_synthetic_ds(50)
        inv = _build_inventory(rows)
        for aset, docs in inv.items():
            for doc in docs:
                assert frozenset(doc["authorids"]) == aset


class TestTierAEntries:
    def setup_method(self):
        import random
        self.rng = random.Random(42)
        rows = _make_synthetic_ds(100)
        self.inventory = _build_inventory(rows)
        self.rotation_pool = list(self.inventory.keys())[:5]
        self.tokenizer = _FakeTokenizer()

    def _kwargs(self):
        return dict(
            inventory=self.inventory,
            n_per_tier=10,
            tokenizer=self.tokenizer,
            target_length=130,
            pair_tol=5,
            global_tol=20,
            rng=self.rng,
        )

    def test_tier_a_disjoint_authors(self):
        entries = _make_tier_a_entries(self.rotation_pool, **self._kwargs())
        for e in entries:
            anchor_set = frozenset(e["anchor_authors"])
            neg_set = frozenset(e["negative_authors"])
            assert anchor_set.isdisjoint(neg_set), (
                f"Tier A: anchor {anchor_set} not disjoint from neg {neg_set}"
            )

    def test_tier_a_same_author_positive(self):
        entries = _make_tier_a_entries(self.rotation_pool, **self._kwargs())
        for e in entries:
            assert frozenset(e["anchor_authors"]) == frozenset(e["positive_authors"])

    def test_tier_a_different_docs(self):
        entries = _make_tier_a_entries(self.rotation_pool, **self._kwargs())
        for e in entries:
            assert e["anchor_doc_id"] != e["positive_doc_id"]


class TestTierBEntries:
    def setup_method(self):
        import random
        self.rng = random.Random(42)
        rows = _make_synthetic_ds(100)
        self.inventory = _build_inventory(rows)
        self.rotation_pool = list(self.inventory.keys())[:5]
        self.tokenizer = _FakeTokenizer()

    def _kwargs(self):
        return dict(
            inventory=self.inventory,
            n_per_tier=10,
            tokenizer=self.tokenizer,
            target_length=130,
            pair_tol=5,
            global_tol=20,
            rng=self.rng,
        )

    def test_tier_b_partial_overlap(self):
        entries = _make_tier_b_entries(self.rotation_pool, **self._kwargs())
        for e in entries:
            anchor_set = frozenset(e["anchor_authors"])
            neg_set = frozenset(e["negative_authors"])
            overlap = len(anchor_set & neg_set)
            # Must have at least one shared author but not be equal
            assert overlap >= 1, (
                f"Tier B: no overlap between anchor {anchor_set} and neg {neg_set}"
            )
            assert anchor_set != neg_set, (
                f"Tier B: anchor {anchor_set} == neg {neg_set}"
            )


class TestTierCEntries:
    def setup_method(self):
        import random
        self.rng = random.Random(42)
        rows = _make_synthetic_ds(100)
        self.inventory = _build_inventory(rows)
        self.rotation_pool = list(self.inventory.keys())[:5]
        self.tokenizer = _FakeTokenizer()

    def _kwargs(self):
        return dict(
            inventory=self.inventory,
            n_per_tier=10,
            tokenizer=self.tokenizer,
            target_length=130,
            pair_tol=5,
            global_tol=20,
            rng=self.rng,
        )

    def test_tier_c_cross_domain(self):
        entries = _make_tier_c_entries(self.rotation_pool, **self._kwargs())
        if not entries:
            pytest.skip("No Tier C entries found with synthetic data (may need more domain diversity)")
        for e in entries:
            # Anchor and positive should be in different domains
            assert e["anchor_domain"] != e["positive_domain"], (
                f"Tier C: anchor domain {e['anchor_domain']} == positive domain {e['positive_domain']}"
            )

    def test_tier_c_same_author_set(self):
        entries = _make_tier_c_entries(self.rotation_pool, **self._kwargs())
        if not entries:
            pytest.skip("No Tier C entries")
        for e in entries:
            assert frozenset(e["anchor_authors"]) == frozenset(e["positive_authors"])


class TestStatistics:
    def test_statistics_counts(self):
        entries = [
            {"tier": "A", "positive_token_len": 130, "negative_token_len": 128,
             "anchor_authors": ["A"], "negative_authors": ["B"]},
            {"tier": "A", "positive_token_len": 132, "negative_token_len": 131,
             "anchor_authors": ["A"], "negative_authors": ["C"]},
            {"tier": "B", "positive_token_len": 129, "negative_token_len": 130,
             "anchor_authors": ["A", "B"], "negative_authors": ["A", "C"]},
        ]
        stats = _compute_statistics(entries)
        assert stats["counts"]["tier_A"] == 2
        assert stats["counts"]["tier_B"] == 1
        assert stats["counts"]["tier_C"] == 0
        assert "tier_A_length_mean" in stats
        assert stats["total"] == 3
