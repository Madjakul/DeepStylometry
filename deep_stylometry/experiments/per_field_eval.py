#!/usr/bin/env python3
# deep_stylometry/experiments/per_field_eval.py
"""Post-hoc per-domain evaluation analysis.

Loads HALvest-Contrastive test data and CSV logger results to compute
per-domain retrieval metrics. Run AFTER ``test.py`` completes.

Usage::

    python -m deep_stylometry.experiments.per_field_eval \\
        --subset base-2 \\
        --scores-csv /path/to/lightning_logs/version_0/metrics.csv \\
        --output per_field_results.json

Alternatively, provide ``--embeddings-h5`` to recompute scores from saved
embeddings rather than reading from the CSV logger.
"""

import argparse
import collections
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# Domain grouping


def group_queries_by_domain(
    domain_labels: List[str],
) -> Dict[str, List[int]]:
    """Group query indices by their domain label.

    Args:
        domain_labels: List of domain strings, one per query.

    Returns:
        Dict mapping domain name → list of query indices.
    """
    groups: Dict[str, List[int]] = collections.defaultdict(list)
    for idx, domain in enumerate(domain_labels):
        groups[domain].append(idx)
    return dict(groups)


# Per-domain metrics computation


def compute_per_domain_metrics(
    scores: torch.Tensor,
    domain_labels: List[str],
    k_values: Tuple[int, ...] = (5, 10, 20, 100),
) -> Dict[str, Dict[str, float]]:
    """Compute per-domain retrieval metrics from a score matrix.

    Assumes that query *i* has its true positive at document index *i*
    (hard positional qrels: rank the correct doc among 2N candidates where
    the positive is doc *i* and the negative is doc *i + N*).

    Args:
        scores: Score matrix of shape ``(n_queries, n_corpus)``.
        domain_labels: Domain label per query, length ``n_queries``.
        k_values: Cutoff values for MRR@k, nDCG@k, recall@k.

    Returns:
        Nested dict: domain → metric_name → value.
    """
    try:
        from ranx import Qrels, Run, evaluate
    except ImportError as exc:
        raise ImportError("ranx is required for metric computation.") from exc

    n_queries = scores.size(0)
    groups = group_queries_by_domain(domain_labels)
    results: Dict[str, Dict[str, float]] = {}

    for domain, indices in groups.items():
        hard = {
            f"q{i}": {f"d{i}": 1}
            for i in indices
        }
        hard_qrels = Qrels(hard)

        # Build ranx Run from score matrix
        run_dict: Dict[str, Dict[str, float]] = {}
        for i in indices:
            row_scores = scores[i]
            topk_scores, topk_indices = row_scores.topk(
                min(max(k_values), scores.size(1)), dim=0
            )
            run_dict[f"q{i}"] = {
                f"d{int(topk_indices[j])}": float(topk_scores[j])
                for j in range(len(topk_indices))
            }
        run = Run(run_dict)

        domain_metrics: Dict[str, float] = {}
        domain_metrics["accuracy"] = evaluate(hard_qrels, run, "precision@1")
        for k in k_values:
            domain_metrics[f"mrr@{k}"] = evaluate(hard_qrels, run, f"mrr@{k}")
            domain_metrics[f"ndcg@{k}"] = evaluate(hard_qrels, run, f"ndcg@{k}")
            domain_metrics[f"recall@{k}"] = evaluate(hard_qrels, run, f"recall@{k}")

        domain_metrics["n_queries"] = len(indices)
        results[domain] = domain_metrics
        logger.info("Domain '%s' (%d queries): %s", domain, len(indices), domain_metrics)

    return results


# Dataset loading helpers


def _load_halvest_domain_labels(
    subset: str = "base-2",
    halvest_name: str = "almanach/halvest-contrastive",
    cache_dir: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """Load query texts and domain labels from HALvest-Contrastive test split.

    IMPORTANT: column names are inspected at runtime.

    Args:
        subset: Subset name (e.g. ``"base-2"``).
        halvest_name: HuggingFace dataset identifier.
        cache_dir: Optional HuggingFace cache directory.

    Returns:
        Tuple of (query_texts, domain_labels), both of length n_queries.
    """
    import datasets as hf_datasets

    ds = hf_datasets.load_dataset(halvest_name, name=subset, cache_dir=cache_dir, split="test")
    logger.info("HALvest test columns: %s", ds.column_names)

    # Detect query column
    query_col = None
    for col in ("query", "anchor", "text", "sentence"):
        if col in ds.column_names:
            query_col = col
            break
    if query_col is None:
        raise ValueError(
            f"No query column found in {ds.column_names}. "
            "Expected one of: query, anchor, text, sentence."
        )

    # Detect domain column
    domain_col = None
    for col in ("query_domain", "domain", "field", "subset", "source", "journal"):
        if col in ds.column_names:
            domain_col = col
            break
    if domain_col is None:
        logger.warning(
            "No domain column found in %s. All queries will be assigned to "
            "'unknown' domain.",
            ds.column_names,
        )
        return list(ds[query_col]), ["unknown"] * len(ds)

    return list(ds[query_col]), list(ds[domain_col])


# Table rendering


def _render_table(results: Dict[str, Dict[str, float]]) -> str:
    """Render per-domain results as a simple text table.

    Args:
        results: Output of :func:`compute_per_domain_metrics`.

    Returns:
        Formatted string table.
    """
    if not results:
        return "(no results)"

    metrics = [k for k in next(iter(results.values())) if k != "n_queries"]
    header = f"{'Domain':<30}" + "".join(f"{m:>12}" for m in metrics) + f"{'N':>8}"
    sep = "-" * len(header)
    rows = [header, sep]
    for domain, vals in sorted(results.items()):
        row = f"{domain:<30}"
        for m in metrics:
            row += f"{vals.get(m, float('nan')):>12.4f}"
        row += f"{int(vals.get('n_queries', 0)):>8}"
        rows.append(row)
    return "\n".join(rows)


# Entry point


def main() -> None:
    """Parse arguments and run per-field evaluation."""
    parser = argparse.ArgumentParser(
        description="Per-domain retrieval metric analysis for HALvest-Contrastive."
    )
    parser.add_argument(
        "--subset",
        default="base-2",
        help="HALvest-Contrastive subset (e.g. base-2, base-4).",
    )
    parser.add_argument(
        "--halvest-cache",
        default=None,
        help="HuggingFace cache directory.",
    )
    parser.add_argument(
        "--scores-npy",
        default=None,
        help="Path to a .npy file containing the (n_queries, n_corpus) score matrix.",
    )
    parser.add_argument(
        "--output",
        default="per_field_results.json",
        help="Path to write the per-domain metrics JSON.",
    )
    args = parser.parse_args()

    logger.info("Loading domain labels for subset '%s' …", args.subset)
    try:
        _, domain_labels = _load_halvest_domain_labels(
            subset=args.subset,
            cache_dir=args.halvest_cache,
        )
    except Exception as exc:
        logger.error("Failed to load dataset: %s", exc)
        return

    if args.scores_npy:
        import numpy as np

        logger.info("Loading score matrix from %s …", args.scores_npy)
        scores_np = np.load(args.scores_npy)
        scores = torch.from_numpy(scores_np).float()
    else:
        logger.error(
            "No score matrix provided. Pass --scores-npy to a saved numpy array "
            "of shape (n_queries, n_corpus). Exiting."
        )
        return

    if scores.size(0) != len(domain_labels):
        logger.warning(
            "Score matrix has %d rows but %d domain labels. "
            "Truncating to the smaller size.",
            scores.size(0),
            len(domain_labels),
        )
        n = min(scores.size(0), len(domain_labels))
        scores = scores[:n]
        domain_labels = domain_labels[:n]

    logger.info("Computing per-domain metrics …")
    results = compute_per_domain_metrics(scores, domain_labels)

    print(_render_table(results))

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info("Wrote per-domain metrics to %s", args.output)

    # Write figures next to the JSON output
    out_dir = os.path.dirname(os.path.abspath(args.output))
    try:
        from deep_stylometry.experiments.visualizations import (
            plot_per_domain_bars,
            plot_per_domain_heatmap,
        )
        plot_per_domain_heatmap(results, out_dir)
        plot_per_domain_bars(results, out_dir)
    except Exception as exc:
        logger.warning("Visualization step failed: %s", exc)


if __name__ == "__main__":
    main()
