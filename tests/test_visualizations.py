# tests/test_visualizations.py
"""Unit tests for deep_stylometry.experiments.visualizations.

All tests run headless (matplotlib Agg backend) and write PDFs to tmp_path.
No network access, no GPU required.
"""

import os
from typing import Dict, Any, List

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entropy_dict(n: int = 5) -> Dict[str, float]:
    """Synthetic entropy_dict with *n* domains."""
    return {f"domain_{i}": float(i + 1) * 0.5 for i in range(n)}


def _jaccard_stats(qp: float = 0.4, qn: float = 0.1) -> Dict[str, float]:
    return {
        "mean_jaccard_qp": qp,
        "std_jaccard_qp":  0.05,
        "mean_jaccard_qn": qn,
        "std_jaccard_qn":  0.02,
    }


def _pan19_stats_summary() -> Dict[str, Any]:
    return {
        "unknown_text_word_length":   {"min": 50,  "max": 500,  "mean": 200.0},
        "candidate_text_word_length": {"min": 100, "max": 1000, "mean": 500.0},
    }


def _make_problem(pid: str, n_unknowns: int = 3, n_candidates: int = 2) -> List[Dict]:
    """Build a list of parsed-problem dicts (one entry per unknown)."""
    candidates = {f"candidate{j:05d}": f"known text {j} " * 20 for j in range(n_candidates)}
    return [
        {
            "problem_id":    pid,
            "unknown_file":  f"unknown{i:05d}.txt",
            "unknown_text":  f"unknown query {i} " * 15,
            "candidates":    candidates,
            "true_author":   "candidate00000",
        }
        for i in range(n_unknowns)
    ]


def _per_domain_results() -> Dict[str, Dict[str, float]]:
    return {
        "cs":   {"accuracy": 0.80, "mrr@5": 0.75, "mrr@10": 0.72, "ndcg@10": 0.70, "recall@10": 0.85, "n_queries": 30},
        "bio":  {"accuracy": 0.65, "mrr@5": 0.60, "mrr@10": 0.58, "ndcg@10": 0.55, "recall@10": 0.72, "n_queries": 20},
        "phys": {"accuracy": 0.72, "mrr@5": 0.68, "mrr@10": 0.66, "ndcg@10": 0.63, "recall@10": 0.78, "n_queries": 15},
    }


# ---------------------------------------------------------------------------
# plot_trigram_entropy
# ---------------------------------------------------------------------------


class TestPlotTrigramEntropy:
    def test_returns_pdf_path(self, tmp_path):
        from deep_stylometry.experiments.visualizations import plot_trigram_entropy

        path = plot_trigram_entropy(_entropy_dict(5), str(tmp_path))
        assert path.endswith(".pdf")
        assert os.path.isfile(path)

    def test_pdf_nonempty(self, tmp_path):
        from deep_stylometry.experiments.visualizations import plot_trigram_entropy

        path = plot_trigram_entropy(_entropy_dict(3), str(tmp_path))
        assert os.path.getsize(path) > 0

    def test_empty_dict_returns_empty_string(self, tmp_path):
        from deep_stylometry.experiments.visualizations import plot_trigram_entropy

        result = plot_trigram_entropy({}, str(tmp_path))
        assert result == ""

    def test_custom_filename(self, tmp_path):
        from deep_stylometry.experiments.visualizations import plot_trigram_entropy

        path = plot_trigram_entropy(_entropy_dict(2), str(tmp_path), filename="custom.pdf")
        assert path.endswith("custom.pdf")
        assert os.path.isfile(path)

    def test_single_domain(self, tmp_path):
        from deep_stylometry.experiments.visualizations import plot_trigram_entropy

        path = plot_trigram_entropy({"only_domain": 3.14}, str(tmp_path))
        assert os.path.isfile(path)

    def test_many_domains(self, tmp_path):
        """Large number of domains should not crash."""
        from deep_stylometry.experiments.visualizations import plot_trigram_entropy

        path = plot_trigram_entropy(_entropy_dict(40), str(tmp_path))
        assert os.path.isfile(path)


# ---------------------------------------------------------------------------
# plot_jaccard
# ---------------------------------------------------------------------------


class TestPlotJaccard:
    def test_returns_pdf_path(self, tmp_path):
        from deep_stylometry.experiments.visualizations import plot_jaccard

        path = plot_jaccard(_jaccard_stats(), str(tmp_path))
        assert path.endswith(".pdf")
        assert os.path.isfile(path)

    def test_pdf_nonempty(self, tmp_path):
        from deep_stylometry.experiments.visualizations import plot_jaccard

        path = plot_jaccard(_jaccard_stats(), str(tmp_path))
        assert os.path.getsize(path) > 0

    def test_zero_std_does_not_crash(self, tmp_path):
        """Zero standard deviation (constant similarity) should not crash."""
        from deep_stylometry.experiments.visualizations import plot_jaccard

        path = plot_jaccard(
            {"mean_jaccard_qp": 0.5, "std_jaccard_qp": 0.0,
             "mean_jaccard_qn": 0.2, "std_jaccard_qn": 0.0},
            str(tmp_path),
        )
        assert os.path.isfile(path)

    def test_missing_keys_fallback_to_zero(self, tmp_path):
        """Missing stats keys default to 0.0 without crashing."""
        from deep_stylometry.experiments.visualizations import plot_jaccard

        path = plot_jaccard({}, str(tmp_path))
        assert os.path.isfile(path)

    def test_custom_filename(self, tmp_path):
        from deep_stylometry.experiments.visualizations import plot_jaccard

        path = plot_jaccard(_jaccard_stats(), str(tmp_path), filename="jac.pdf")
        assert path.endswith("jac.pdf")


# ---------------------------------------------------------------------------
# plot_pan19_word_lengths
# ---------------------------------------------------------------------------


class TestPlotPan19WordLengths:
    def test_summary_path_returns_pdf(self, tmp_path):
        """No problems list → falls back to summary bar chart."""
        from deep_stylometry.experiments.visualizations import plot_pan19_word_lengths

        path = plot_pan19_word_lengths(_pan19_stats_summary(), output_dir=str(tmp_path))
        assert os.path.isfile(path)

    def test_histogram_path_returns_pdf(self, tmp_path):
        """With problems list → per-item histograms."""
        from deep_stylometry.experiments.visualizations import plot_pan19_word_lengths

        problems = _make_problem("p1", n_unknowns=10) + _make_problem("p2", n_unknowns=8)
        path = plot_pan19_word_lengths(
            _pan19_stats_summary(), problems=problems, output_dir=str(tmp_path)
        )
        assert os.path.isfile(path)

    def test_empty_problems_list_falls_back_to_summary(self, tmp_path):
        """Empty list (not None) should NOT crash — uses summary path."""
        from deep_stylometry.experiments.visualizations import plot_pan19_word_lengths

        path = plot_pan19_word_lengths(
            _pan19_stats_summary(), problems=[], output_dir=str(tmp_path)
        )
        assert os.path.isfile(path)

    def test_none_problems_uses_summary(self, tmp_path):
        from deep_stylometry.experiments.visualizations import plot_pan19_word_lengths

        path = plot_pan19_word_lengths(
            _pan19_stats_summary(), problems=None, output_dir=str(tmp_path)
        )
        assert os.path.isfile(path)

    def test_single_problem_no_crash(self, tmp_path):
        """A single problem/unknown should not crash _n_bins (Sturges edge case)."""
        from deep_stylometry.experiments.visualizations import plot_pan19_word_lengths

        problems = _make_problem("p1", n_unknowns=1)
        path = plot_pan19_word_lengths(
            _pan19_stats_summary(), problems=problems, output_dir=str(tmp_path)
        )
        assert os.path.isfile(path)

    def test_pdf_nonempty(self, tmp_path):
        from deep_stylometry.experiments.visualizations import plot_pan19_word_lengths

        path = plot_pan19_word_lengths(_pan19_stats_summary(), output_dir=str(tmp_path))
        assert os.path.getsize(path) > 0

    def test_output_dir_created(self, tmp_path):
        """Output dir is created if it doesn't exist."""
        from deep_stylometry.experiments.visualizations import plot_pan19_word_lengths

        nested = str(tmp_path / "a" / "b" / "c")
        path = plot_pan19_word_lengths(_pan19_stats_summary(), output_dir=nested)
        assert os.path.isfile(path)


# ---------------------------------------------------------------------------
# plot_per_domain_heatmap
# ---------------------------------------------------------------------------


class TestPlotPerDomainHeatmap:
    def test_returns_pdf(self, tmp_path):
        from deep_stylometry.experiments.visualizations import plot_per_domain_heatmap

        path = plot_per_domain_heatmap(_per_domain_results(), str(tmp_path))
        assert os.path.isfile(path)

    def test_empty_results_returns_empty_string(self, tmp_path):
        from deep_stylometry.experiments.visualizations import plot_per_domain_heatmap

        result = plot_per_domain_heatmap({}, str(tmp_path))
        assert result == ""

    def test_custom_metric_keys(self, tmp_path):
        from deep_stylometry.experiments.visualizations import plot_per_domain_heatmap

        path = plot_per_domain_heatmap(
            _per_domain_results(),
            str(tmp_path),
            metric_keys=["accuracy", "mrr@10"],
        )
        assert os.path.isfile(path)

    def test_single_domain(self, tmp_path):
        from deep_stylometry.experiments.visualizations import plot_per_domain_heatmap

        path = plot_per_domain_heatmap(
            {"only": {"accuracy": 0.9, "mrr@10": 0.8}}, str(tmp_path)
        )
        assert os.path.isfile(path)

    def test_single_metric(self, tmp_path):
        """One-metric heatmap should not crash."""
        from deep_stylometry.experiments.visualizations import plot_per_domain_heatmap

        results = {d: {"accuracy": v} for d, v in [("cs", 0.8), ("bio", 0.6)]}
        path = plot_per_domain_heatmap(results, str(tmp_path), metric_keys=["accuracy"])
        assert os.path.isfile(path)

    def test_identical_column_values_no_divide_by_zero(self, tmp_path):
        """All domains have the same metric value → no divide-by-zero in normalisation."""
        from deep_stylometry.experiments.visualizations import plot_per_domain_heatmap

        results = {d: {"accuracy": 0.75} for d in ["cs", "bio", "phys"]}
        path = plot_per_domain_heatmap(results, str(tmp_path))
        assert os.path.isfile(path)


# ---------------------------------------------------------------------------
# plot_per_domain_bars
# ---------------------------------------------------------------------------


class TestPlotPerDomainBars:
    def test_returns_pdf(self, tmp_path):
        from deep_stylometry.experiments.visualizations import plot_per_domain_bars

        path = plot_per_domain_bars(_per_domain_results(), str(tmp_path))
        assert os.path.isfile(path)

    def test_empty_results_returns_empty_string(self, tmp_path):
        from deep_stylometry.experiments.visualizations import plot_per_domain_bars

        result = plot_per_domain_bars({}, str(tmp_path))
        assert result == ""

    def test_missing_metrics_returns_empty_string(self, tmp_path):
        """If none of the requested metrics exist in results, returns ''."""
        from deep_stylometry.experiments.visualizations import plot_per_domain_bars

        results = {"cs": {"some_other_metric": 0.5}}
        result = plot_per_domain_bars(
            results, str(tmp_path), metrics=("accuracy", "mrr@10")
        )
        assert result == ""

    def test_custom_metrics(self, tmp_path):
        from deep_stylometry.experiments.visualizations import plot_per_domain_bars

        path = plot_per_domain_bars(
            _per_domain_results(), str(tmp_path), metrics=("accuracy",)
        )
        assert os.path.isfile(path)

    def test_single_domain(self, tmp_path):
        from deep_stylometry.experiments.visualizations import plot_per_domain_bars

        path = plot_per_domain_bars(
            {"cs": {"accuracy": 0.8, "mrr@10": 0.7, "ndcg@10": 0.65}},
            str(tmp_path),
        )
        assert os.path.isfile(path)

    def test_many_domains(self, tmp_path):
        """Many domains should not crash or produce a malformed figure."""
        from deep_stylometry.experiments.visualizations import plot_per_domain_bars

        results = {
            f"domain_{i}": {"accuracy": i / 20, "mrr@10": i / 25, "ndcg@10": i / 30}
            for i in range(1, 16)
        }
        path = plot_per_domain_bars(results, str(tmp_path))
        assert os.path.isfile(path)
