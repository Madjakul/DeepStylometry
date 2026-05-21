# tests/test_per_field_eval.py
"""Unit tests for per_field_eval helpers."""

import pytest
import torch


# ---------------------------------------------------------------------------
# group_queries_by_domain
# ---------------------------------------------------------------------------


class TestGroupByDomain:
    def test_basic_grouping(self):
        """Queries are grouped correctly by domain label."""
        from deep_stylometry.experiments.per_field_eval import group_queries_by_domain

        labels = ["cs", "bio", "cs", "phys", "bio", "cs"]
        groups = group_queries_by_domain(labels)

        assert set(groups["cs"]) == {0, 2, 5}
        assert set(groups["bio"]) == {1, 4}
        assert set(groups["phys"]) == {3}

    def test_single_domain(self):
        """All queries in one domain."""
        from deep_stylometry.experiments.per_field_eval import group_queries_by_domain

        labels = ["cs"] * 5
        groups = group_queries_by_domain(labels)
        assert groups == {"cs": [0, 1, 2, 3, 4]}

    def test_empty_labels(self):
        """Empty label list returns empty dict."""
        from deep_stylometry.experiments.per_field_eval import group_queries_by_domain

        groups = group_queries_by_domain([])
        assert groups == {}

    def test_ten_queries_two_domains(self):
        """10 queries split evenly across 2 domains."""
        from deep_stylometry.experiments.per_field_eval import group_queries_by_domain

        labels = ["A" if i % 2 == 0 else "B" for i in range(10)]
        groups = group_queries_by_domain(labels)
        assert len(groups["A"]) == 5
        assert len(groups["B"]) == 5
        assert all(i % 2 == 0 for i in groups["A"])
        assert all(i % 2 == 1 for i in groups["B"])


# ---------------------------------------------------------------------------
# compute_per_domain_metrics
# ---------------------------------------------------------------------------


class TestMetricsPerGroup:
    @pytest.fixture
    def tiny_scores(self):
        """4 queries × 8 documents (4 pos + 4 neg).

        Query i's true positive is document i (first 4 docs are positives).
        For simplicity, each query scores highest on its own positive.
        """
        # 4 queries, 8 docs
        # Score matrix: query i scores 1.0 on doc i, 0.1 elsewhere
        scores = torch.full((4, 8), 0.1)
        for i in range(4):
            scores[i, i] = 1.0  # positive is always top-ranked
        return scores

    def test_perfect_retrieval_accuracy(self, tiny_scores):
        """With perfect scores, accuracy@1 should be 1.0 for each domain."""
        from deep_stylometry.experiments.per_field_eval import (
            compute_per_domain_metrics,
        )

        labels = ["cs", "cs", "bio", "bio"]
        results = compute_per_domain_metrics(tiny_scores, labels, k_values=(1, 5))

        assert "cs" in results
        assert "bio" in results
        assert results["cs"]["accuracy"] == pytest.approx(1.0, abs=1e-4)
        assert results["bio"]["accuracy"] == pytest.approx(1.0, abs=1e-4)

    def test_query_count_per_domain(self, tiny_scores):
        """n_queries field is correct per domain."""
        from deep_stylometry.experiments.per_field_eval import (
            compute_per_domain_metrics,
        )

        labels = ["cs", "bio", "cs", "phys"]
        results = compute_per_domain_metrics(tiny_scores, labels, k_values=(1,))

        assert results["cs"]["n_queries"] == 2
        assert results["bio"]["n_queries"] == 1
        assert results["phys"]["n_queries"] == 1

    def test_all_domains_in_results(self, tiny_scores):
        """All unique domain labels appear as keys in results."""
        from deep_stylometry.experiments.per_field_eval import (
            compute_per_domain_metrics,
        )

        labels = ["x", "y", "x", "z"]
        results = compute_per_domain_metrics(tiny_scores, labels, k_values=(1,))

        assert set(results.keys()) == {"x", "y", "z"}

    def test_metric_keys_present(self, tiny_scores):
        """Expected metric keys are present for each domain."""
        from deep_stylometry.experiments.per_field_eval import (
            compute_per_domain_metrics,
        )

        labels = ["a", "a", "b", "b"]
        results = compute_per_domain_metrics(tiny_scores, labels, k_values=(5, 10))

        for domain in ["a", "b"]:
            assert "accuracy" in results[domain]
            assert "mrr@5" in results[domain]
            assert "ndcg@5" in results[domain]
            assert "recall@5" in results[domain]
            assert "mrr@10" in results[domain]

    def test_single_domain_all_queries(self, tiny_scores):
        """Works when all queries belong to the same domain."""
        from deep_stylometry.experiments.per_field_eval import (
            compute_per_domain_metrics,
        )

        labels = ["cs"] * 4
        results = compute_per_domain_metrics(tiny_scores, labels, k_values=(1,))
        assert "cs" in results
        assert results["cs"]["n_queries"] == 4

    def test_worst_case_retrieval(self):
        """When the true positive is always ranked last, accuracy is 0."""
        from deep_stylometry.experiments.per_field_eval import (
            compute_per_domain_metrics,
        )

        # 2 queries, 4 docs; positive is doc 0 and 1, but scored lowest
        scores = torch.tensor([
            [0.1, 0.5, 0.8, 0.9],  # q0 positive=d0, but d0 has lowest score
            [0.1, 0.4, 0.7, 0.9],  # q1 positive=d1, but d1 has second-lowest
        ])
        labels = ["cs", "cs"]
        results = compute_per_domain_metrics(scores, labels, k_values=(1,))
        assert results["cs"]["accuracy"] == pytest.approx(0.0, abs=1e-4)
