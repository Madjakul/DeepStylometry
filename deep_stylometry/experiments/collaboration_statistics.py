#!/usr/bin/env python3
# deep_stylometry/experiments/collaboration_statistics.py
"""Collaboration Rate & Author-Set Statistics for HALvest-Contrastive.

Computes, per config and per domain:
  a) Author-set size distribution (mean, median, std, min, max, histogram).
  b) Single-author rate (% of triplets with exactly one author).
  c) Author-set stability / recurrence (unique sets, repeat rate, frequency
     distribution).
  d) Cross-triplet author overlap (Jaccard between pos and neg author-sets).

Outputs:
  - ``collaboration_stats.json``
  - ``collaboration_tables.tex``
  - ``author_set_size_distribution.{pdf,png}``
  - ``collaboration_recurrence.{pdf,png}``

Run::

    python -m deep_stylometry.experiments.collaboration_statistics \\
        --cache-dir ~/.cache/huggingface/hub \\
        --output-dir ./stats \\
        --configs base-2 base-4 base-6 base-8 base-10 \\
        --split train
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import statistics
from typing import Any, Counter, Dict, FrozenSet, List, Optional, Set, Tuple

import datasets as hf_datasets
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASET_NAME = "almanach/halvest-contrastive"
ALL_CONFIGS = ["base-2", "base-4", "base-6", "base-8", "base-10"]

# Candidate column names to search for domain information.
_DOMAIN_CANDIDATES = ("query_domain", "domain", "field", "source", "subset")


# Column resolution helpers


def _resolve_col(col_names: List[str], candidates: Tuple[str, ...]) -> Optional[str]:
    """Return the first candidate column that exists in ``col_names``.

    Args:
        col_names: Available column names in the dataset.
        candidates: Ordered candidate column names.

    Returns:
        The matching column name, or ``None`` if none found.
    """
    for c in candidates:
        if c in col_names:
            return c
    return None


def _flatten_domain(val: Any) -> str:
    """Normalise a domain value that may be a list or a scalar.

    Args:
        val: Raw domain value from the HuggingFace row.

    Returns:
        A single string domain label.
    """
    if isinstance(val, list):
        return val[0] if val else ""
    return str(val) if val is not None else ""


# Jaccard utility


def _jaccard(set_a: FrozenSet, set_b: FrozenSet) -> float:
    """Jaccard similarity between two frozensets.

    Args:
        set_a: First set.
        set_b: Second set.

    Returns:
        Jaccard similarity in [0, 1]; 0.0 if both sets are empty.
    """
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


# Per-config statistics


def _authorids_to_frozenset(raw: Any) -> Optional[FrozenSet[str]]:
    """Convert a raw author-id value to a frozenset of strings.

    Args:
        raw: Value from the ``pos_authorids`` column (list, str, or None).

    Returns:
        ``frozenset`` of author-id strings, or ``None`` if the value is
        empty/null.
    """
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        if not raw:
            return None
        return frozenset(str(a) for a in raw)
    return frozenset([str(raw)])


def _hist_bins(sizes: List[int]) -> Dict[str, int]:
    """Count author-set sizes into bins 1, 2, ..., 7+.

    Args:
        sizes: List of author-set sizes (positive integers).

    Returns:
        Dict mapping bin label (``"1"``, ``"2"``, ..., ``"7+"``) to count.
    """
    bins: Dict[str, int] = {str(i): 0 for i in range(1, 8)}
    bins["7+"] = 0
    for s in sizes:
        key = str(s) if s <= 7 else "7+"
        bins[key] = bins.get(key, 0) + 1
    return bins


def _desc_stats(values: List[float]) -> Dict[str, float]:
    """Compute descriptive statistics for a numeric list.

    Args:
        values: Numeric values (must be non-empty).

    Returns:
        Dict with ``mean``, ``median``, ``std``, ``min``, ``max``.
    """
    if not values:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "std": float(statistics.pstdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def compute_config_stats(
    config_name: str,
    split: str,
    cache_dir: Optional[str],
) -> Dict[str, Any]:
    """Compute all statistics for one HALvest-Contrastive config.

    Args:
        config_name: Config identifier (e.g. ``"base-4"``).
        split: HuggingFace split name (e.g. ``"train"``).
        cache_dir: Optional HuggingFace cache directory.

    Returns:
        Dict of computed statistics, structured by domain and globally.
    """
    logger.info("Loading %s / %s …", config_name, split)
    ds: hf_datasets.Dataset = hf_datasets.load_dataset(
        DATASET_NAME, name=config_name, split=split, cache_dir=cache_dir
    )

    cols = ds.column_names
    logger.info("  Columns: %s", cols)

    # --- Column safety checks -----------------------------------------------
    assert "pos_authorids" in cols, (
        f"Expected 'pos_authorids' in {config_name} columns, got: {cols}"
    )

    domain_col = _resolve_col(cols, _DOMAIN_CANDIDATES)
    if domain_col is None:
        logger.warning("  No domain column found in %s; treating as single domain.", config_name)

    neg_auth_col = "neg_authorids" if "neg_authorids" in cols else None

    # --- Build per-row records -----------------------------------------------
    n_dropped = 0
    rows: List[Dict] = []
    for row in ds:
        pos_auth = _authorids_to_frozenset(row["pos_authorids"])
        if pos_auth is None:
            n_dropped += 1
            continue

        domain = _flatten_domain(row[domain_col]) if domain_col else "unknown"

        neg_auth: Optional[FrozenSet[str]] = None
        if neg_auth_col:
            neg_auth = _authorids_to_frozenset(row.get(neg_auth_col))

        rows.append({"pos_auth": pos_auth, "domain": domain, "neg_auth": neg_auth})

    logger.info(
        "  %d rows loaded, %d dropped (empty/null pos_authorids)",
        len(rows),
        n_dropped,
    )

    if not rows:
        return {"n_rows": 0, "n_dropped": n_dropped}

    # --- Global and per-domain aggregation -----------------------------------
    domains: List[str] = sorted(set(r["domain"] for r in rows))
    if not domains:
        domains = ["unknown"]

    def _stats_for_rows(subset: List[Dict]) -> Dict[str, Any]:
        """Compute all statistics for a list of row dicts."""
        # a) Author-set size distribution.
        sizes = [len(r["pos_auth"]) for r in subset]
        size_stats = _desc_stats([float(s) for s in sizes])
        size_hist = _hist_bins(sizes)
        size_hist_pct = {k: 100.0 * v / len(sizes) for k, v in size_hist.items()}

        # b) Single-author rate.
        n_single = sum(1 for s in sizes if s == 1)
        single_author_rate = 100.0 * n_single / len(sizes)

        # c) Author-set recurrence.
        auth_set_counter: Counter[FrozenSet[str]] = collections.Counter(
            r["pos_auth"] for r in subset
        )
        n_unique_sets = len(auth_set_counter)
        n_sets_ge2 = sum(1 for c in auth_set_counter.values() if c >= 2)
        n_sets_ge5 = sum(1 for c in auth_set_counter.values() if c >= 5)
        pct_ge2 = 100.0 * n_sets_ge2 / max(n_unique_sets, 1)
        pct_ge5 = 100.0 * n_sets_ge5 / max(n_unique_sets, 1)

        # Frequency distribution of author-set appearances.
        freq_counter: Counter[int] = collections.Counter(auth_set_counter.values())
        freq_dist: Dict[str, int] = {str(k): v for k, v in sorted(freq_counter.items())}

        # d) Cross-triplet author overlap (Jaccard pos ↔ neg).
        jac_values = [
            _jaccard(r["pos_auth"], r["neg_auth"])
            for r in subset
            if r["neg_auth"] is not None
        ]
        jaccard_stats = _desc_stats(jac_values) if jac_values else {}
        if jac_values:
            jaccard_stats["n_pairs"] = len(jac_values)

        return {
            "n_triplets": len(subset),
            "author_set_size": size_stats,
            "author_set_size_histogram": size_hist,
            "author_set_size_histogram_pct": size_hist_pct,
            "single_author_rate_pct": single_author_rate,
            "n_unique_author_sets": n_unique_sets,
            "pct_author_sets_ge2_docs": pct_ge2,
            "pct_author_sets_ge5_docs": pct_ge5,
            "author_set_freq_distribution": freq_dist,
            "neg_pos_jaccard": jaccard_stats,
        }

    result: Dict[str, Any] = {
        "config": config_name,
        "split": split,
        "n_rows": len(rows),
        "n_dropped": n_dropped,
        "global": _stats_for_rows(rows),
        "by_domain": {},
    }

    for domain in domains:
        subset = [r for r in rows if r["domain"] == domain]
        if not subset:
            continue
        result["by_domain"][domain] = _stats_for_rows(subset)

    return result


# LaTeX output


def _render_latex(all_stats: Dict[str, Any]) -> str:
    """Render LaTeX booktabs tables summarising per-domain statistics.

    Args:
        all_stats: Dict keyed by config name, each value being the output of
            :func:`compute_config_stats`.

    Returns:
        LaTeX string containing two ``table`` environments.
    """
    # Collect all domain names across all configs.
    all_domains: Set[str] = set()
    for cfg_stats in all_stats.values():
        all_domains.update(cfg_stats.get("by_domain", {}).keys())
    all_domains_sorted = sorted(all_domains)

    lines: List[str] = []

    # Table 1: Author-set size stats per domain (averaged across configs).
    lines += [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Author-set size statistics and single-author rate per domain"
        r" (HALvest-Contrastive training splits, averaged across configs)}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Domain & Mean & Median & Std & Min & Max & Single-author \% \\",
        r"\midrule",
    ]

    for domain in all_domains_sorted:
        means, medians, stds, mins, maxs, singles = [], [], [], [], [], []
        for cfg_stats in all_stats.values():
            d = cfg_stats.get("by_domain", {}).get(domain)
            if d is None:
                continue
            s = d["author_set_size"]
            means.append(s["mean"])
            medians.append(s["median"])
            stds.append(s["std"])
            mins.append(s["min"])
            maxs.append(s["max"])
            singles.append(d["single_author_rate_pct"])

        def _avg(vals: List[float]) -> str:
            return f"{sum(vals)/len(vals):.2f}" if vals else "--"

        lines.append(
            f"{domain} & {_avg(means)} & {_avg(medians)} & {_avg(stds)} "
            f"& {_avg(mins)} & {_avg(maxs)} & {_avg(singles)} \\\\"
        )

    # Global row.
    g_means, g_medians, g_stds, g_mins, g_maxs, g_singles = [], [], [], [], [], []
    for cfg_stats in all_stats.values():
        g = cfg_stats.get("global", {})
        if not g:
            continue
        s = g.get("author_set_size", {})
        g_means.append(s.get("mean", 0.0))
        g_medians.append(s.get("median", 0.0))
        g_stds.append(s.get("std", 0.0))
        g_mins.append(s.get("min", 0.0))
        g_maxs.append(s.get("max", 0.0))
        g_singles.append(g.get("single_author_rate_pct", 0.0))

    lines += [
        r"\midrule",
        f"\\textbf{{Global}} & {_avg(g_means)} & {_avg(g_medians)} & {_avg(g_stds)} "
        f"& {_avg(g_mins)} & {_avg(g_maxs)} & {_avg(g_singles)} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]

    # Table 2: Author-set recurrence per config.
    lines += [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Author-set recurrence statistics per HALvest-Contrastive config}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Config & \#~Triplets & \#~Unique sets & $\geq$2 docs (\%) & $\geq$5 docs (\%) \\",
        r"\midrule",
    ]

    for cfg_name, cfg_stats in sorted(all_stats.items()):
        g = cfg_stats.get("global", {})
        lines.append(
            f"{cfg_name} & {cfg_stats.get('n_rows', 0):,} "
            f"& {g.get('n_unique_author_sets', 0):,} "
            f"& {g.get('pct_author_sets_ge2_docs', 0.0):.1f} "
            f"& {g.get('pct_author_sets_ge5_docs', 0.0):.1f} \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# Plotting


def _plot_size_distribution(
    all_stats: Dict[str, Any],
    output_dir: str,
) -> None:
    """Grouped bar chart of author-set size bins per domain (global stats).

    Args:
        all_stats: Full statistics dict keyed by config name.
        output_dir: Directory for output files.
    """
    # Aggregate histogram across all configs, globally.
    bin_labels = [str(i) for i in range(1, 8)] + ["7+"]
    agg_counts: Dict[str, Dict[str, float]] = {}

    for cfg_name, cfg_stats in sorted(all_stats.items()):
        hist = cfg_stats.get("global", {}).get("author_set_size_histogram", {})
        if not hist:
            continue
        agg_counts[cfg_name] = {b: hist.get(b, 0) for b in bin_labels}

    if not agg_counts:
        logger.warning("No histogram data available for size distribution plot.")
        return

    configs = sorted(agg_counts.keys())
    n_configs = len(configs)
    n_bins = len(bin_labels)
    x = np.arange(n_bins)
    width = 0.8 / max(n_configs, 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, cfg in enumerate(configs):
        counts = [agg_counts[cfg].get(b, 0) for b in bin_labels]
        ax.bar(x + i * width - 0.4 + width / 2, counts, width, label=cfg)

    ax.set_xlabel("Author-set size")
    ax.set_ylabel("Number of triplets")
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.legend(title="Config", fontsize=8)
    plt.tight_layout()

    for ext in ("pdf", "png"):
        path = os.path.join(output_dir, f"author_set_size_distribution.{ext}")
        fig.savefig(path, dpi=150)
        logger.info("Saved %s", path)
    plt.close(fig)


def _plot_recurrence(
    all_stats: Dict[str, Any],
    output_dir: str,
) -> None:
    """Histogram of author-set document-frequency distribution.

    Shows how many unique author-sets appear exactly k times, aggregated
    across all configs.

    Args:
        all_stats: Full statistics dict keyed by config name.
        output_dir: Directory for output files.
    """
    agg_freq: Counter[int] = collections.Counter()
    for cfg_stats in all_stats.values():
        freq_dist = cfg_stats.get("global", {}).get("author_set_freq_distribution", {})
        for k_str, v in freq_dist.items():
            agg_freq[int(k_str)] += v

    if not agg_freq:
        logger.warning("No frequency data for recurrence plot.")
        return

    # Cap display at frequency 20 for readability; bucket everything higher.
    cap = 20
    x_vals: List[int] = []
    y_vals: List[int] = []
    overflow = 0
    for k in sorted(agg_freq.keys()):
        if k <= cap:
            x_vals.append(k)
            y_vals.append(agg_freq[k])
        else:
            overflow += agg_freq[k]

    if overflow:
        x_vals.append(cap + 1)
        y_vals.append(overflow)

    tick_labels = [str(x) for x in x_vals]
    if overflow:
        tick_labels[-1] = f">{cap}"

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(x_vals)), y_vals, color="#4393c3", edgecolor="white")
    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Number of documents per unique author-set")
    ax.set_ylabel("Number of unique author-sets")
    plt.tight_layout()

    for ext in ("pdf", "png"):
        path = os.path.join(output_dir, f"collaboration_recurrence.{ext}")
        fig.savefig(path, dpi=150)
        logger.info("Saved %s", path)
    plt.close(fig)


# Entry point


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Collaboration rate & author-set statistics for HALvest-Contrastive"
    )
    p.add_argument("--cache-dir", default=None, help="HuggingFace cache directory")
    p.add_argument("--output-dir", default="./stats")
    p.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="HALvest configs to process (default: all available)",
    )
    p.add_argument("--split", default="train", help="HuggingFace split (default: train)")
    return p


def main() -> None:
    """Parse arguments and run collaboration statistics computation."""
    args = _build_arg_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.configs is None:
        try:
            args.configs = hf_datasets.get_dataset_config_names(DATASET_NAME)
            logger.info("Auto-detected configs: %s", args.configs)
        except Exception as exc:
            logger.warning("Could not auto-detect configs: %s; using defaults.", exc)
            args.configs = ALL_CONFIGS

    all_stats: Dict[str, Any] = {}

    for cfg_name in args.configs:
        try:
            stats = compute_config_stats(
                config_name=cfg_name,
                split=args.split,
                cache_dir=args.cache_dir,
            )
            all_stats[cfg_name] = stats
        except Exception as exc:
            logger.error("Failed on config '%s': %s", cfg_name, exc)
            all_stats[cfg_name] = {"error": str(exc)}

    # --- JSON output ---------------------------------------------------------
    json_path = os.path.join(args.output_dir, "collaboration_stats.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(all_stats, fh, indent=2, default=str)
    logger.info("Saved statistics → %s", json_path)

    # --- LaTeX output --------------------------------------------------------
    tex = _render_latex(all_stats)
    tex_path = os.path.join(args.output_dir, "collaboration_tables.tex")
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write(tex)
    logger.info("Saved LaTeX tables → %s", tex_path)

    # --- Figures -------------------------------------------------------------
    _plot_size_distribution(all_stats, args.output_dir)
    _plot_recurrence(all_stats, args.output_dir)

    # Print summary to stdout.
    for cfg_name, stats in all_stats.items():
        g = stats.get("global", {})
        print(
            f"{cfg_name}: {stats.get('n_rows', 0):,} triplets, "
            f"mean author-set size={g.get('author_set_size', {}).get('mean', 0):.2f}, "
            f"single-author={g.get('single_author_rate_pct', 0):.1f}%, "
            f"unique sets={g.get('n_unique_author_sets', 0):,}"
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
