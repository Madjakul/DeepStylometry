# deep_stylometry/utils/retrieval_aggregation.py
"""Shared aggregation and reporting helpers for retrieval inspection.

# DECISION: Logic is extracted here to avoid duplication between
# deep_stylometry/experiments/retrieval_inspection.py (main GPU pipeline) and
# deep_stylometry/experiments/retrieval_inspection_reaggregate.py
# (post-hoc re-aggregation tool). Both scripts import from this module.

Note on JSON serialisation: float("inf") is not valid JSON per RFC 8259, and
Python's json module serialises it as "Infinity" (invalid JSON). We provide
_sanitize_for_json() which recursively converts non-finite floats to their
string representations ("inf", "-inf", "nan") before json.dump. The in-memory
summary dict always carries float("inf") so Python-side comparisons work
correctly; only the serialised file uses the string form.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _safe_mean(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _full_rank_bin(r: int) -> str:
    """Map a 1-indexed global TP rank to an 8-bin string label.

    The caller is responsible for ensuring r is a valid positive integer —
    this function does not accept None (see _aggregate_pair_records for the
    enforcement point).
    """
    if r == 1:
        return "1"
    if r <= 5:
        return "2-5"
    if r <= 10:
        return "6-10"
    if r <= 20:
        return "11-20"
    if r <= 50:
        return "21-50"
    if r <= 100:
        return "51-100"
    if r <= 500:
        return "101-500"
    return ">500"


def _rank_bin(rank: Optional[int], top_k: int) -> str:
    """Map a TP rank to a 4-bin bucket label for distractor-vs-easy comparisons.

    Args:
        rank: Global 1-indexed TP rank, or None if unknown.
        top_k: The K used during retrieval (boundary for "rank>K" bucket).

    Returns:
        One of ``"rank=1"``, ``"rank=2-5"``, ``"rank=6-20"``, ``"rank>K"``.
    """
    if rank is None or rank > top_k:
        return "rank>K"
    if rank == 1:
        return "rank=1"
    if rank <= 5:
        return "rank=2-5"
    return "rank=6-20"


def _bucket_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute mean of each measurement field over a list of pair records."""
    if not records:
        return {}
    fields = [
        "author_jaccard_with_query",
        "author_jaccard_with_true_positive",
        "same_domain_as_query",
        "same_domain_as_true_positive",
        "domain_jaccard_with_true_positive",
        "unigram_jaccard_with_query",
        "unigram_jaccard_with_true_positive",
        "any_author_overlap_with_query",
        "any_author_overlap_with_true_positive",
    ]
    out: Dict[str, Any] = {}
    for f in fields:
        vals = [r[f] for r in records if r.get(f) is not None]
        out[f"mean_{f}"] = _safe_mean([float(v) for v in vals])
    out["n_pairs"] = len(records)
    return out


def _ratio_block(
    key: str,
    distractor_stats: Dict[str, Any],
    random_baseline: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a self-describing ratio dict for one measurement key.

    Returns a dict with keys: distractor_mean, random_mean, n_random_pairs,
    ratio (float | None; may be float("inf")), note (str | None).

    A zero random baseline with positive distractor mean yields ratio=inf and
    an explanatory note rather than None, so the signal is not silently lost.
    """
    dval = distractor_stats.get(f"mean_{key}")
    bval = random_baseline.get(f"mean_{key}")
    n_random = random_baseline.get("n_pairs", 0)
    out: Dict[str, Any] = {
        "distractor_mean": dval,
        "random_mean": bval,
        "n_random_pairs": n_random,
    }
    if dval is None or bval is None:
        out["ratio"] = None
        out["note"] = "Insufficient data to compute ratio."
    elif bval == 0.0:
        if dval > 0:
            out["ratio"] = float("inf")
            out["note"] = (
                f"Random baseline is exactly zero across {n_random} pairs; "
                f"distractors exhibit this property at rate {dval:.4f} while "
                f"random pairs never do. Ratio is reported as +inf; the "
                f"underlying signal is that the model finds this property "
                f"essentially never present under chance pairing."
            )
        else:
            out["ratio"] = None
            out["note"] = "Both distractor and random baselines are zero."
    else:
        out["ratio"] = float(dval) / float(bval)
        out["note"] = None
    return out


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively replace non-finite floats with strings for JSON serialisation.

    json.dump produces "Infinity" for float("inf"), which is not valid JSON.
    This converts them to string "inf"/"-inf"/"nan" so the file is valid JSON.
    The in-memory summary dict is NOT modified (Python comparisons still work).
    """
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float):
        if obj == float("inf"):
            return "inf"
        if obj == float("-inf"):
            return "-inf"
        if obj != obj:  # NaN
            return "nan"
    return obj


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate_pair_records(
    pair_records: List[Dict[str, Any]],
    random_pairs: List[Dict[str, Any]],
    top_k: int,
    *,
    stale_ranks: bool = False,
) -> Dict[str, Any]:
    """Aggregate per-pair records into bucketed statistics and ratio blocks.

    Args:
        pair_records: All ranked (query, candidate) pair records. Each must
            contain ``seed``, ``query_ds_idx``, ``true_positive_rank``, and
            the measurement fields consumed by ``_bucket_stats``.
        random_pairs: Pair records for random-sampled (query, candidate) pairs
            (no ranking required; used only for the random baseline).
        top_k: K used during retrieval (threshold for the 4-bin bucketing).
        stale_ranks: If True, accept ``true_positive_rank=None`` from old
            pairs.jsonl and map those queries to a ``">top_k"`` bucket with a
            stderr warning. The bins 21-50, 51-100, 101-500, >500 cannot be
            recovered from stale data. If False (default), raises AssertionError
            on any None rank.

    Returns:
        Summary dict with ``n_queries``, ``rank_distribution``,
        ``bucketed_stats``, ``random_baseline``, ``headline_ratios``,
        and ``overall_metrics``.
    """
    rank_bins = ["1", "2-5", "6-10", "11-20", "21-50", "51-100", "101-500", ">500"]
    if stale_ranks:
        rank_bins = rank_bins + [">top_k"]
    rank_counts: Dict[str, int] = {b: 0 for b in rank_bins}

    stale_count = 0
    seen_queries: Set[Tuple[int, int]] = set()

    for rec in pair_records:
        key = (rec["seed"], rec["query_ds_idx"])
        if key not in seen_queries:
            seen_queries.add(key)
            tp_rank = rec["true_positive_rank"]
            if tp_rank is None:
                if stale_ranks:
                    rank_counts[">top_k"] += 1
                    stale_count += 1
                else:
                    raise AssertionError(
                        f"true_positive_rank is None for query {key}. "
                        "This should not occur after the global ranking fix in "
                        "retrieval_inspection.py. If you are re-aggregating an "
                        "old pairs.jsonl, use stale_ranks=True."
                    )
            else:
                rank_counts[_full_rank_bin(tp_rank)] += 1

    if stale_count > 0:
        print(
            f"WARNING: {stale_count}/{len(seen_queries)} queries have "
            f"true_positive_rank=None (outside top-{top_k} in the original run). "
            "These bins cannot be recovered from the existing pairs.jsonl; "
            "rerun retrieval_inspection.py to populate the 21-500 rank bins.",
            file=sys.stderr,
        )

    n_queries = len(seen_queries)
    rank_pct = {b: 100.0 * rank_counts[b] / max(n_queries, 1) for b in rank_bins}

    # 4-bin bucketed stats (rank=1, rank=2-5, rank=6-20, rank>K).
    bucket_labels = ["rank=1", "rank=2-5", "rank=6-20", "rank>K"]
    bucket_pairs: Dict[str, List[Dict[str, Any]]] = {b: [] for b in bucket_labels}
    for rec in pair_records:
        tp_rank = rec["true_positive_rank"]
        bucket = _rank_bin(tp_rank, top_k)
        bucket_pairs[bucket].append(rec)

    bucketed_stats = {b: _bucket_stats(bucket_pairs[b]) for b in bucket_labels}

    random_baseline = _bucket_stats(random_pairs)
    random_baseline["n_pairs"] = len(random_pairs)

    # Headline ratios: rank=2-20 distractors vs. random baseline.
    distractor_records = bucket_pairs["rank=2-5"] + bucket_pairs["rank=6-20"]
    distractor_stats = _bucket_stats(distractor_records)

    headline_ratios = {
        "author_jaccard_with_query": _ratio_block(
            "author_jaccard_with_query", distractor_stats, random_baseline
        ),
        "same_domain_as_query": _ratio_block(
            "same_domain_as_query", distractor_stats, random_baseline
        ),
        "unigram_jaccard_with_query": _ratio_block(
            "unigram_jaccard_with_query", distractor_stats, random_baseline
        ),
    }

    rank1_pct = rank_pct.get("1", 0.0)
    top5_recall = (
        sum(rank_counts.get(b, 0) for b in ["1", "2-5"]) / max(n_queries, 1) * 100.0
    )

    return {
        "n_queries": n_queries,
        "rank_distribution": {"counts": rank_counts, "percentages": rank_pct},
        "bucketed_stats": bucketed_stats,
        "random_baseline": random_baseline,
        "headline_ratios": headline_ratios,
        "overall_metrics": {
            "rank1_accuracy_pct": rank1_pct,
            "top5_recall_pct": top5_recall,
        },
    }


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def _write_report(
    path: Path,
    summary: Dict[str, Any],
    subset: str,
    n_seeds: int,
    n_queries: int,
) -> None:
    """Write a human-readable summary of retrieval inspection results.

    Args:
        path: Output file path.
        summary: Output of :func:`_aggregate_pair_records`.
        subset: Subset name (informational).
        n_seeds: Number of seeds used (informational).
        n_queries: Queries per seed (informational).
    """
    om = summary.get("overall_metrics", {})
    rd = summary.get("rank_distribution", {})
    hr = summary.get("headline_ratios", {})
    n_q = summary.get("n_queries", 0)

    lines = [
        "=" * 70,
        f"Retrieval Inspection Report — subset={subset}",
        f"  {n_seeds} seeds × {n_queries} queries = {n_seeds * n_queries} total",
        f"  Actual ranked queries: {n_q}",
        "=" * 70,
        "",
        "=== Rank Distribution (full-pool global ranks) ===",
    ]
    for b, cnt in rd.get("counts", {}).items():
        pct = rd.get("percentages", {}).get(b, 0.0)
        if b == ">top_k":
            label = ">top_k  [stale: rank unknown, rerun to populate 21-500 bins]"
        else:
            label = b
        lines.append(f"  {label:>58}: {cnt:>5}  ({pct:.1f}%)")

    lines += [
        "",
        "=== Overall Metrics ===",
        f"  Rank-1 accuracy:  {om.get('rank1_accuracy_pct', 0.0):.1f}%",
        f"  Top-5 recall:     {om.get('top5_recall_pct', 0.0):.1f}%",
        "",
        "=== Headline Ratios vs. Random Baseline (rank=2-20 distractors) ===",
    ]

    ratio_labels = {
        "author_jaccard_with_query": "Author-Jaccard",
        "same_domain_as_query": "Domain-match",
        "unigram_jaccard_with_query": "Unigram-Jaccard",
    }
    for key, label in ratio_labels.items():
        block = hr.get(key, {})
        d_mean = block.get("distractor_mean")
        r_mean = block.get("random_mean")
        ratio = block.get("ratio")
        note = block.get("note")
        n_rand = block.get("n_random_pairs", 0)

        if ratio is None:
            ratio_str = "N/A"
        elif ratio == float("inf"):
            ratio_str = "+inf (random baseline is zero)"
        else:
            ratio_str = f"{ratio:.4f}×"

        lines.append(f"  {label}:")
        lines.append(f"    distractor_mean = {d_mean}")
        lines.append(f"    random_mean     = {r_mean}  (n={n_rand})")
        lines.append(f"    ratio           = {ratio_str}")
        if note:
            lines.append(f"    NOTE: {note}")
        lines.append("")

    lines += [
        "=== Diagnostic Sanity ===",
        f"  Rank-1 accuracy: {om.get('rank1_accuracy_pct', 0.0):.1f}%",
        "  If rank-1 accuracy is much lower than test.py reports for this checkpoint,",
        "  the pool may be misconstructed — verify pool size and tp_pool_idx mapping.",
        "",
    ]

    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    logger.info("Wrote %s", path)
    print(text)
