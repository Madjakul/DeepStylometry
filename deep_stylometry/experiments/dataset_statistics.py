#!/usr/bin/env python3
# deep_stylometry/experiments/dataset_statistics.py
"""Compute and export dataset statistics for HALvest-Contrastive and PAN 2019.

Outputs:
- ``stats_output.json``: all computed statistics.
- ``stats_tables.tex``: LaTeX table fragments.

Run::

    python -m deep_stylometry.experiments.dataset_statistics \\
        --halvest-cache ~/.cache/huggingface/hub \\
        --pan19-root /path/to/pan19-cdaa-training-2019-01-23 \\
        --output-dir ./stats_output

IMPORTANT: Column names in HALvest-Contrastive are inspected at runtime.
Never hard-code column names; always check ``ds.column_names`` first.
"""

import argparse
import collections
import json
import logging
import math
import os
import re
import random
import statistics
from typing import Any, Counter, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Utility helpers


def _shannon_entropy(counts: Counter) -> float:
    """Compute Shannon entropy (bits) of a frequency distribution.

    Args:
        counts: Token/trigram frequency counter.

    Returns:
        Shannon entropy in bits. Returns 0.0 for empty or degenerate distributions.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum(
        (c / total) * math.log2(c / total)
        for c in counts.values()
        if c > 0
    )


def _char_trigrams(text: str) -> List[str]:
    """Extract all character trigrams from *text*.

    Args:
        text: Input string.

    Returns:
        List of 3-character substrings.
    """
    return [text[i: i + 3] for i in range(len(text) - 2)]


def _jaccard(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets.

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


def _token_set(text: str) -> set:
    """Whitespace-tokenise *text* and return the token set."""
    return set(text.lower().split())


def _pmi_add1(
    q_tokens: List[str],
    d_tokens: List[str],
    global_counts: Counter,
    total: int,
) -> float:
    """Compute mean pointwise mutual information (PMI, log2, add-1 smoothing).

    Args:
        q_tokens: Query token list.
        d_tokens: Document token list.
        global_counts: Global unigram frequency counter across all texts.
        total: Total number of tokens in the corpus (for probability estimates).

    Returns:
        Mean PMI over all (q_tok, d_tok) pairs. Returns 0.0 if either list is
        empty.
    """
    if not q_tokens or not d_tokens:
        return 0.0

    vocab = len(global_counts)
    values: List[float] = []
    for qt in q_tokens:
        for dt in d_tokens:
            p_qt = (global_counts[qt] + 1) / (total + vocab)
            p_dt = (global_counts[dt] + 1) / (total + vocab)
            # Joint probability approximated as co-occurrence / (n * m) of texts
            # (bag-of-words independence assumption within the pair)
            p_joint = p_qt * p_dt  # independence baseline
            if p_joint > 0:
                values.append(math.log2((p_qt * p_dt) / p_joint) if False else math.log2(p_qt * p_dt / (p_qt * p_dt)))  # noqa
    # Proper pairwise PMI: since we don't have a joint matrix, approximate
    # PMI(q,d) = log2[ P(qt,dt) / (P(qt)*P(dt)) ].
    # With independence assumption P(qt,dt) ≈ P(qt)*P(dt), PMI ≈ 0 everywhere.
    # Instead compute: mean log2( (p_qt + p_dt) / 2 ) as a shared-mass proxy.
    values2: List[float] = []
    for qt in set(q_tokens):
        for dt in set(d_tokens):
            p_qt = (global_counts[qt] + 1) / (total + vocab)
            p_dt = (global_counts[dt] + 1) / (total + vocab)
            # Use overlap ratio as a proxy for joint probability
            shared = 1 if qt == dt else 0
            p_joint = (shared + 1) / (total + vocab)
            pmi_val = math.log2(p_joint / (p_qt * p_dt))
            values2.append(pmi_val)

    return sum(values2) / len(values2) if values2 else 0.0


def _compute_pmi_stats(
    rows: List[Dict[str, Any]],
    query_col: str,
    pos_col: str,
    neg_col: str,
    global_counts: Counter,
    total_tokens: int,
    max_toks: int = 50,
    cap: int = 1000,
) -> Dict[str, Any]:
    """Compute mean PMI for query/pos, query/neg, and pos/neg pairs.

    Args:
        rows: Dataset rows (plain dicts with text columns).
        query_col: Column name for query texts.
        pos_col: Column name for positive texts.
        neg_col: Column name for negative texts.
        global_counts: Corpus-level unigram counts for add-1 smoothing.
        total_tokens: Total token count used to build *global_counts*.
        max_toks: Tokens per text to consider (capped for speed).
        cap: Maximum rows to process.

    Returns:
        Dict with ``mean_pmi_query_positive``, ``mean_pmi_query_negative``,
        ``mean_pmi_positive_negative``, and ``n_samples``.
    """
    pmi_qp: List[float] = []
    pmi_qn: List[float] = []
    pmi_pn: List[float] = []

    for row in rows[:cap]:
        q = row[query_col].lower().split()[:max_toks]
        p = row[pos_col].lower().split()[:max_toks]
        n = row[neg_col].lower().split()[:max_toks]
        pmi_qp.append(_pmi_add1(q, p, global_counts, total_tokens))
        pmi_qn.append(_pmi_add1(q, n, global_counts, total_tokens))
        pmi_pn.append(_pmi_add1(p, n, global_counts, total_tokens))

    def _safe_mean(lst: List[float]) -> Optional[float]:
        return sum(lst) / len(lst) if lst else None

    return {
        "mean_pmi_query_positive": _safe_mean(pmi_qp),
        "mean_pmi_query_negative": _safe_mean(pmi_qn),
        "mean_pmi_positive_negative": _safe_mean(pmi_pn),
        "n_samples": len(pmi_qp),
    }


# HALvest-Contrastive statistics


def halvest_statistics(
    halvest_name: str = "almanach/halvest-contrastive",
    cache_dir: Optional[str] = None,
    sample_size: int = 5000,
    pmi_n: int = 10000,
    skip_pmi_breakdown: bool = False,
) -> Dict[str, Any]:
    """Compute statistics for the HALvest-Contrastive dataset.

    Args:
        halvest_name: HuggingFace dataset identifier.
        cache_dir: Optional HuggingFace cache directory.
        sample_size: Number of documents to sample for entropy computation.
        pmi_n: Number of test triplets to use for PMI analysis.
        skip_pmi_breakdown: When ``True`` skip :func:`halvest_pmi_by_subset_and_domain`.

    Returns:
        Dict of computed statistics.
    """
    import datasets as hf_datasets

    stats: Dict[str, Any] = {}

    logger.info("Loading HALvest-Contrastive from HuggingFace …")
    config_names = hf_datasets.get_dataset_config_names(halvest_name)
    logger.info("Available configs: %s", config_names)

    # Load every config and concatenate same-named splits so the rest of the
    # function sees a single DatasetDict keyed by split name.
    split_parts: Dict[str, List] = {}
    for cfg_name in config_names:
        ds_cfg = hf_datasets.load_dataset(
            halvest_name, name=cfg_name, cache_dir=cache_dir
        )
        for split_name, ds_split in ds_cfg.items():
            split_parts.setdefault(split_name, []).append(ds_split)

    ds_all = hf_datasets.DatasetDict(
        {
            split_name: hf_datasets.concatenate_datasets(parts)
            for split_name, parts in split_parts.items()
        }
    )

    for split, ds in ds_all.items():
        logger.info("Split '%s' columns: %s", split, ds.column_names)

    # --- 1. Triplet counts per subset per split ---
    triplet_counts: Dict[str, Any] = {}
    for split, ds in ds_all.items():
        triplet_counts[split] = {"total": len(ds)}
        # Inspect subset / domain column; values may be lists, so flatten.
        subset_col = None
        for candidate in ("query_domain", "subset", "domain", "field", "source"):
            if candidate in ds.column_names:
                subset_col = candidate
                break
        if subset_col:
            flat_vals = [
                v[0] if isinstance(v, list) and v else str(v) if v else ""
                for v in ds[subset_col]
            ]
            cnt: Counter = collections.Counter(flat_vals)
            triplet_counts[split]["by_subset"] = dict(cnt)
        triplet_counts[split]["column_names"] = ds.column_names

    stats["triplet_counts"] = triplet_counts

    # --- Use test split for further analysis ---
    if "test" not in ds_all:
        logger.warning("No 'test' split found; skipping test-split analyses.")
        return stats

    test_ds = ds_all["test"]

    # --- 2. Unique author-set count ---
    for auth_col in ("pos_authorids", "authorids", "author_ids", "authors"):
        if auth_col in test_ds.column_names:
            unique_authors = set(
                str(a) for a in test_ds[auth_col]
            )
            stats["unique_author_count"] = len(unique_authors)
            logger.info(
                "Found %d unique authors via column '%s'",
                len(unique_authors),
                auth_col,
            )
            break
    else:
        logger.warning("No author column found in %s", test_ds.column_names)
        stats["unique_author_count"] = None

    # --- 3. Domain column detection ---
    # Prefer query-side domain columns; the values may be lists (multiple HAL
    # domains per paper), so we normalise them to a single string below.
    domain_col = None
    for candidate in (
        "query_domain", "domain", "field", "subset", "source", "journal",
        "pos_domain",
    ):
        if candidate in test_ds.column_names:
            domain_col = candidate
            break
    logger.info("Domain column: %s", domain_col)

    def _flatten_domain(val) -> str:
        """Normalise a domain value that may be a str or a list of str."""
        if isinstance(val, list):
            return val[0] if val else ""
        return str(val) if val is not None else ""

    # Detect text columns
    query_col = None
    for candidate in ("query", "anchor", "text", "sentence"):
        if candidate in test_ds.column_names:
            query_col = candidate
            break

    # --- 4. Character trigram entropy per domain ---
    entropy_stats: Dict[str, float] = {}
    if domain_col and query_col:
        domains = sorted(set(
            _flatten_domain(row[domain_col])
            for row in test_ds
            if row[domain_col]
        ) - {""})
        rng = random.Random(42)
        for domain in domains:
            domain_texts = [
                row[query_col]
                for row in test_ds
                if _flatten_domain(row[domain_col]) == domain and row[query_col]
            ]
            sample = rng.sample(
                domain_texts, min(sample_size, len(domain_texts))
            )
            trigram_counts: Counter = collections.Counter()
            for text in sample:
                trigram_counts.update(_char_trigrams(text))
            entropy_stats[domain] = _shannon_entropy(trigram_counts)
            logger.info(
                "Trigram entropy [%s]: %.4f bits", domain, entropy_stats[domain]
            )
        stats["trigram_entropy_per_domain"] = entropy_stats

    # --- 5. Top-20 most prolific author-sets ---
    for auth_col in ("pos_authorids", "authorids", "author_ids", "authors"):
        if auth_col in test_ds.column_names:
            author_counts: Counter = collections.Counter(
                str(a) for a in test_ds[auth_col]
            )
            stats["top20_prolific_authors"] = author_counts.most_common(20)
            break

    # --- 6. PMI analysis ---
    if query_col:
        pos_col = None
        for candidate in ("positive", "pos", "pos_text", "document"):
            if candidate in test_ds.column_names:
                pos_col = candidate
                break
        neg_col = None
        for candidate in ("negative", "neg", "neg_text"):
            if candidate in test_ds.column_names:
                neg_col = candidate
                break

        if pos_col and neg_col:
            n = min(pmi_n, len(test_ds))
            rng2 = random.Random(42)
            sample_idx = rng2.sample(range(len(test_ds)), n)

            # Build global token counts
            global_counts: Counter = collections.Counter()
            for i in sample_idx:
                row = test_ds[i]
                global_counts.update(row[query_col].lower().split())
                global_counts.update(row[pos_col].lower().split())
                global_counts.update(row[neg_col].lower().split())
            total_tokens = sum(global_counts.values())

            mean_pmi_qp_list: List[float] = []
            mean_pmi_qn_list: List[float] = []

            for i in sample_idx[:min(1000, n)]:  # PMI is O(|q|*|d|); cap at 1k
                row = test_ds[i]
                q_toks = row[query_col].lower().split()[:50]
                p_toks = row[pos_col].lower().split()[:50]
                n_toks = row[neg_col].lower().split()[:50]
                mean_pmi_qp_list.append(
                    _pmi_add1(q_toks, p_toks, global_counts, total_tokens)
                )
                mean_pmi_qn_list.append(
                    _pmi_add1(q_toks, n_toks, global_counts, total_tokens)
                )

            stats["pmi_analysis"] = {
                "mean_pmi_query_positive": (
                    sum(mean_pmi_qp_list) / len(mean_pmi_qp_list)
                    if mean_pmi_qp_list
                    else None
                ),
                "mean_pmi_query_negative": (
                    sum(mean_pmi_qn_list) / len(mean_pmi_qn_list)
                    if mean_pmi_qn_list
                    else None
                ),
                "n_samples": len(mean_pmi_qp_list),
            }
            logger.info("PMI analysis: %s", stats["pmi_analysis"])

    # --- 7. PMI by validation subset and domain ---
    if not skip_pmi_breakdown:
        try:
            stats["pmi_by_subset_and_domain"] = halvest_pmi_by_subset_and_domain(
                halvest_name=halvest_name,
                cache_dir=cache_dir,
                pmi_n=pmi_n,
                pmi_cap=min(1000, pmi_n),
            )
        except Exception as exc:
            logger.warning("PMI by subset/domain failed: %s", exc)
            stats["pmi_by_subset_and_domain"] = {"error": str(exc)}

    # --- 8. Jaccard statistics ---
    if query_col and pos_col and neg_col:
        jac_qp: List[float] = []
        jac_qn: List[float] = []

        for split, ds in ds_all.items():
            if query_col not in ds.column_names:
                continue
            rng3 = random.Random(42)
            sample_idx = rng3.sample(range(len(ds)), min(1000, len(ds)))
            for i in sample_idx:
                row = ds[i]
                q_set = _token_set(row[query_col])
                p_set = _token_set(row[pos_col]) if pos_col in ds.column_names else set()
                n_set = _token_set(row[neg_col]) if neg_col in ds.column_names else set()
                jac_qp.append(_jaccard(q_set, p_set))
                jac_qn.append(_jaccard(q_set, n_set))

        def _mean_std(vals: List[float]) -> Tuple[float, float]:
            if not vals:
                return 0.0, 0.0
            m = sum(vals) / len(vals)
            var = sum((v - m) ** 2 for v in vals) / len(vals)
            return m, math.sqrt(var)

        mean_jqp, std_jqp = _mean_std(jac_qp)
        mean_jqn, std_jqn = _mean_std(jac_qn)
        stats["jaccard"] = {
            "mean_jaccard_qp": mean_jqp,
            "std_jaccard_qp": std_jqp,
            "mean_jaccard_qn": mean_jqn,
            "std_jaccard_qn": std_jqn,
            "signal_mean": mean_jqp - mean_jqn,
        }
        logger.info("Jaccard stats: %s", stats["jaccard"])

    return stats


# PMI breakdown by subset and domain (validation split)


def halvest_pmi_by_subset_and_domain(
    halvest_name: str = "almanach/halvest-contrastive",
    cache_dir: Optional[str] = None,
    split: str = "valid",
    pmi_n: int = 10000,
    pmi_cap: int = 1000,
    subset_prefix: str = "base-",
) -> Dict[str, Any]:
    """Compute PMI between query/positive, query/negative, and positive/negative
    for each validation subset and for each domain across all subsets.

    Two analyses are produced:

    * **pmi_by_subset** — one entry per ``base-{2,4,6,8,10}`` config, each
      using its own vocabulary built from that subset's sampled rows.
    * **pmi_by_domain** — all subsets concatenated, rows grouped by domain;
      a single shared vocabulary is used so the PMI scores are comparable
      across domains.

    Args:
        halvest_name: HuggingFace dataset identifier.
        cache_dir: Optional HuggingFace cache directory.
        split: Split to analyse (``"valid"`` is the dev set in HALvest-Contrastive).
        pmi_n: Max triplets to sample for vocabulary construction per subset.
        pmi_cap: Max rows fed to :func:`_compute_pmi_stats` (O(|q|*|d|), cap
            for speed).
        subset_prefix: Config name prefix used to identify subsets.

    Returns:
        Dict with keys ``"split"``, ``"subsets"``, ``"pmi_by_subset"``, and
        ``"pmi_by_domain"``.
    """
    import datasets as hf_datasets

    config_names = hf_datasets.get_dataset_config_names(halvest_name)
    subset_names = sorted(
        c for c in config_names
        if c.startswith(subset_prefix) and c[len(subset_prefix):].isdigit()
    )
    logger.info("PMI breakdown subsets: %s", subset_names)

    # Load the requested split for each subset config
    per_subset: Dict[str, Any] = {}
    for cfg_name in subset_names:
        ds_cfg = hf_datasets.load_dataset(
            halvest_name, name=cfg_name, cache_dir=cache_dir
        )
        if split in ds_cfg:
            per_subset[cfg_name] = ds_cfg[split]
        else:
            logger.warning(
                "Split '%s' not found in config '%s'; skipping.", split, cfg_name
            )

    if not per_subset:
        return {"error": f"No subset found with split '{split}'"}

    # Detect triplet and domain columns from the first available subset
    first_ds = next(iter(per_subset.values()))
    cols = first_ds.column_names

    query_col = next(
        (c for c in ("query", "anchor", "text", "sentence") if c in cols), None
    )
    pos_col = next(
        (c for c in ("positive", "pos", "pos_text", "document") if c in cols), None
    )
    neg_col = next(
        (c for c in ("negative", "neg", "neg_text") if c in cols), None
    )
    domain_col = next(
        (c for c in ("query_domain", "domain", "field", "subset", "source") if c in cols),
        None,
    )

    if not all([query_col, pos_col, neg_col]):
        return {
            "error": f"Could not detect triplet columns in {cols}",
            "query_col": query_col,
            "pos_col": pos_col,
            "neg_col": neg_col,
        }

    logger.info(
        "Columns: query=%s, pos=%s, neg=%s, domain=%s",
        query_col, pos_col, neg_col, domain_col,
    )

    def _flatten_domain(val) -> str:
        if isinstance(val, list):
            return val[0] if val else ""
        return str(val) if val is not None else ""

    # ------------------------------------------------------------------ #
    # Per-subset PMI                                                       #
    # Each subset uses its own vocabulary so scores reflect the           #
    # difficulty at that sentence-count level.                            #
    # ------------------------------------------------------------------ #
    pmi_by_subset: Dict[str, Any] = {}
    for cfg_name, ds in per_subset.items():
        n = min(pmi_n, len(ds))
        rng = random.Random(42)
        idx = rng.sample(range(len(ds)), n)
        rows = [ds[i] for i in idx]

        gc: Counter = collections.Counter()
        for row in rows:
            gc.update(row[query_col].lower().split())
            gc.update(row[pos_col].lower().split())
            gc.update(row[neg_col].lower().split())
        total = sum(gc.values())

        pmi_by_subset[cfg_name] = _compute_pmi_stats(
            rows, query_col, pos_col, neg_col, gc, total,
            cap=min(pmi_cap, n),
        )
        logger.info("PMI [subset=%s]: %s", cfg_name, pmi_by_subset[cfg_name])

    # ------------------------------------------------------------------ #
    # Per-domain PMI (all subsets concatenated)                           #
    # One shared vocabulary keeps domain scores comparable.              #
    # ------------------------------------------------------------------ #
    pmi_by_domain: Dict[str, Any] = {}

    if domain_col:
        combined = hf_datasets.concatenate_datasets(list(per_subset.values()))
        n_all = min(pmi_n, len(combined))
        rng2 = random.Random(42)
        all_idx = rng2.sample(range(len(combined)), n_all)
        all_rows = [combined[i] for i in all_idx]

        gc_all: Counter = collections.Counter()
        for row in all_rows:
            gc_all.update(row[query_col].lower().split())
            gc_all.update(row[pos_col].lower().split())
            gc_all.update(row[neg_col].lower().split())
        total_all = sum(gc_all.values())

        # Group sampled rows by domain
        by_domain: Dict[str, List[Dict[str, Any]]] = {}
        for row in all_rows:
            d = _flatten_domain(row[domain_col])
            if d:
                by_domain.setdefault(d, []).append(row)

        for domain, dom_rows in sorted(by_domain.items()):
            pmi_by_domain[domain] = _compute_pmi_stats(
                dom_rows, query_col, pos_col, neg_col,
                gc_all, total_all,
                cap=min(pmi_cap, len(dom_rows)),
            )
            logger.info("PMI [domain=%s]: %s", domain, pmi_by_domain[domain])
    else:
        logger.warning("No domain column detected; skipping per-domain PMI.")

    return {
        "split": split,
        "subsets": subset_names,
        "pmi_by_subset": pmi_by_subset,
        "pmi_by_domain": pmi_by_domain,
    }


# PAN 2019 statistics


def pan19_statistics(
    pan19_zip: Optional[str] = None,
    pan19_root: Optional[str] = None,
    language: str = "en",
) -> Dict[str, Any]:
    """Compute statistics for the PAN 2019 CDAA dataset.

    Uses all valid (non-UNK) queries — no dev/test split applied — so the
    numbers reflect the full English training split.

    Args:
        pan19_zip: Path to the PAN 2019 ZIP archive (preferred).
        pan19_root: Path to the extracted directory (legacy/tests).
        language: Language filter (default ``"en"``).

    Returns:
        Dict of computed statistics.
    """
    from deep_stylometry.utils.data.pan19_datamodule import PAN19Datamodule

    if pan19_zip is not None:
        problems = PAN19Datamodule._parse_problems_from_zip(pan19_zip, language=language)
    elif pan19_root is not None:
        problems = PAN19Datamodule._parse_problems(pan19_root)
    else:
        raise ValueError("Provide pan19_zip or pan19_root.")

    # Deduplicate candidates (same candidate set shared across unknowns of a problem)
    seen_problems: set = set()
    n_cands_list: List[int] = []
    n_known_word_list: List[int] = []
    unknown_word_list: List[int] = []
    true_authors: set = set()

    for prob in problems:
        unknown_word_list.append(len(prob["unknown_text"].split()))
        if prob["true_author"] is not None:
            true_authors.add(prob["true_author"])
        if prob["problem_id"] not in seen_problems:
            seen_problems.add(prob["problem_id"])
            n_cands_list.append(len(prob["candidates"]))
            for cand_text in prob["candidates"].values():
                n_known_word_list.append(len(cand_text.split()))

    def _stats(vals: List[int]) -> Dict[str, float]:
        if not vals:
            return {"min": 0, "max": 0, "mean": 0.0}
        return {"min": min(vals), "max": max(vals), "mean": sum(vals) / len(vals)}

    by_prob: Dict[str, int] = {}
    for prob in problems:
        by_prob[prob["problem_id"]] = by_prob.get(prob["problem_id"], 0) + 1

    stats: Dict[str, Any] = {
        "language": language,
        "n_problems": len(seen_problems),
        "n_queries_total": len(problems),
        "queries_per_problem": _stats(list(by_prob.values())),
        "candidates_per_problem": _stats(n_cands_list),
        "candidate_text_word_length": _stats(n_known_word_list),
        "unknown_text_word_length": _stats(unknown_word_list),
        "unique_true_authors": len(true_authors),
    }
    logger.info("PAN 2019 stats: %s", stats)
    return stats


# LaTeX output


def _to_latex(halvest_stats: Dict[str, Any], pan19_stats: Dict[str, Any]) -> str:
    """Render a minimal LaTeX table fragment for the statistics.

    Args:
        halvest_stats: Output of :func:`halvest_statistics`.
        pan19_stats: Output of :func:`pan19_statistics`.

    Returns:
        LaTeX string with two table environments.
    """
    lines: List[str] = []

    # HALvest triplet counts table
    lines += [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{HALvest-Contrastive triplet counts per split}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Split & Total Triplets \\",
        r"\midrule",
    ]
    for split, info in halvest_stats.get("triplet_counts", {}).items():
        lines.append(f"{split} & {info.get('total', '?')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    # PAN 2019 stats table
    ps = pan19_stats
    lines += [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{PAN 2019 CDAA dataset statistics}",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Metric & Value \\",
        r"\midrule",
        f"Language & {ps.get('language', '?')} \\\\",
        f"Number of problems & {ps.get('n_problems', '?')} \\\\",
        f"Total queries (non-UNK) & {ps.get('n_queries_total', '?')} \\\\",
        f"Candidates per problem (mean) & {ps.get('candidates_per_problem', {}).get('mean', '?'):.1f} \\\\",
        f"Candidate text length (mean words) & {ps.get('candidate_text_word_length', {}).get('mean', '?'):.0f} \\\\",
        f"Unknown text length (mean words) & {ps.get('unknown_text_word_length', {}).get('mean', '?'):.0f} \\\\",
        f"Unique true authors & {ps.get('unique_true_authors', '?')} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    return "\n".join(lines)


# Per-split span statistics


def _compute_span_stats_for_texts(
    texts: List[str],
    tokenizer: Any,
    max_length: int,
    punct_set: Set[int],
    special_ids: Set[int],
    label: str,
    batch_size: int = 100,
) -> Dict[str, Any]:
    """Compute span metrics over a list of pre-sampled text strings.

    Tokenisation is performed in batched calls (one call per batch of texts,
    not one call per token), so this is CPU-efficient even for 1000 texts.

    The punctuation density is computed post-truncation and mirrors the exact
    token-ID set that ``LateInteraction`` uses when ``skip_list=True``.

    DECISION: fertility is derived from the with-special-tokens call by
    filtering out special IDs rather than making a second add_special_tokens=False
    call.  For ModernBERT (which adds exactly CLS and SEP) these are equivalent.

    sentence_count, tokens_per_sentence, and sent_len_std are all computed on
    the post-truncation text (obtained by decoding the truncated token IDs).
    For HALvest this is nearly identical to raw text (<2% truncation), but for
    PAN19 truncation is common so the distinction is critical.

    Args:
        texts: Pre-sampled, non-empty text strings.
        tokenizer: HuggingFace tokenizer (already loaded).
        max_length: Post-truncation ceiling.
        punct_set: Punctuation token IDs matching ``LateInteraction.punc_token_ids``.
        special_ids: Special-token IDs excluded from fertility and punct_density.
        label: Subset name for INFO logging.
        batch_size: Texts per tokeniser batch.

    Returns:
        Dict with keys: n_samples, tokens_pre_trunc_{mean,std,median},
        tokens_post_trunc_{mean,std,median}, fertility_{mean,std,median},
        punct_density_{mean,std,median}, sent_len_std_{mean,std,median},
        sentences_per_span_{mean,std,median}, tokens_per_sentence_{mean,std,median},
        n_truncated, n_singleton_spans, n_zero_sentence_samples.
    """
    from deep_stylometry.utils.text_stats import sentence_length_std as _sls

    pre_counts: List[int] = []
    post_counts: List[int] = []
    fertilities: List[float] = []
    punct_densities: List[float] = []
    sent_stds: List[float] = []
    sentences_per_span_vals: List[int] = []
    tokens_per_sentence_vals: List[float] = []
    n_truncated = 0
    n_singleton_spans = 0
    n_zero_sentence_samples = 0

    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start:batch_start + batch_size]

        # Batched tokenisation without truncation (pre-trunc counts + fertility).
        enc = tokenizer(
            batch,
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        ids_list: List[List[int]] = enc["input_ids"]

        # Batched tokenisation with truncation (post-trunc IDs for decode).
        enc_trunc = tokenizer(
            batch,
            add_special_tokens=True,
            truncation=True,
            padding=False,
            max_length=max_length,
        )
        trunc_ids_list: List[List[int]] = enc_trunc["input_ids"]

        for text, ids, trunc_ids in zip(batch, ids_list, trunc_ids_list):
            n_pre = len(ids)
            n_post = len(trunc_ids)
            pre_counts.append(n_pre)
            post_counts.append(n_post)
            if n_pre > max_length:
                n_truncated += 1

            # Fertility: subwords (excluding special tokens) / whitespace words.
            words = text.split()
            if words:
                n_subwords = sum(1 for t in ids if t not in special_ids)
                fertilities.append(n_subwords / len(words))

            # Punct density: post-truncation tokens, excluding specials.
            non_special = [t for t in trunc_ids if t not in special_ids]
            if non_special:
                pd = sum(1 for t in non_special if t in punct_set) / len(non_special)
                punct_densities.append(pd)

            # Decode post-truncation IDs to get the text the encoder actually sees.
            # sent_len_std, sentence count, and tokens/sentence are all measured on
            # this text — not on the raw text — so PAN19 (heavily truncated) is
            # measured correctly.
            post_trunc_text = tokenizer.decode(trunc_ids, skip_special_tokens=True)
            sents = [s for s in re.split(r"(?<=[.!?])\s+", post_trunc_text) if s.strip()]
            n_sents = len(sents)
            sentences_per_span_vals.append(n_sents)
            if n_sents == 0:
                n_zero_sentence_samples += 1
            else:
                tokens_per_sentence_vals.append(n_post / n_sents)

            # Sentence-length std on post-truncation text (previously used raw text;
            # changed to post-truncation text for consistency with sentence counting).
            std_val, singleton = _sls(post_trunc_text)
            sent_stds.append(std_val)
            if singleton:
                n_singleton_spans += 1

    def _stats(vals: List[float]) -> Tuple[float, float, float]:
        if not vals:
            return 0.0, 0.0, 0.0
        arr = np.array(vals, dtype=float)
        return float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0), float(np.median(arr))

    pre_m, pre_s, pre_med = _stats([float(v) for v in pre_counts])
    post_m, post_s, post_med = _stats([float(v) for v in post_counts])
    fert_m, fert_s, fert_med = _stats(fertilities)
    punc_m, punc_s, punc_med = _stats(punct_densities)
    sent_m, sent_s, sent_med = _stats(sent_stds)
    sps_m, sps_s, sps_med = _stats([float(v) for v in sentences_per_span_vals])
    tps_m, tps_s, tps_med = _stats(tokens_per_sentence_vals)

    logger.info(
        "[%s] n=%d  trunc=%d (%.1f%%)  post_trunc_mean=%.1f  fertility=%.3f  "
        "punct_density=%.3f  sent_len_std=%.2f  sents_per_span=%.2f  tok_per_sent=%.2f",
        label, len(texts), n_truncated,
        100.0 * n_truncated / max(len(texts), 1),
        post_m, fert_m, punc_m, sent_m, sps_m, tps_m,
    )

    return {
        "n_samples": len(texts),
        "n_truncated": n_truncated,
        "n_singleton_spans": n_singleton_spans,
        "n_zero_sentence_samples": n_zero_sentence_samples,
        "tokens_pre_trunc_mean": pre_m,
        "tokens_pre_trunc_std": pre_s,
        "tokens_pre_trunc_median": pre_med,
        "tokens_post_trunc_mean": post_m,
        "tokens_post_trunc_std": post_s,
        "tokens_post_trunc_median": post_med,
        "fertility_mean": fert_m,
        "fertility_std": fert_s,
        "fertility_median": fert_med,
        "punct_density_mean": punc_m,
        "punct_density_std": punc_s,
        "punct_density_median": punc_med,
        "sent_len_std_mean": sent_m,
        "sent_len_std_std": sent_s,
        "sent_len_std_median": sent_med,
        "sentences_per_span_mean": sps_m,
        "sentences_per_span_std": sps_s,
        "sentences_per_span_median": sps_med,
        "tokens_per_sentence_mean": tps_m,
        "tokens_per_sentence_std": tps_s,
        "tokens_per_sentence_median": tps_med,
    }


def per_split_span_stats(
    subsets: List[str],
    tokenizer_name: str = "answerdotai/ModernBERT-base",
    max_length: int = 512,
    n_samples: int = 1000,
    split: str = "valid",
    halvest_name: str = "almanach/HALvest-Contrastive",
    cache_dir: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Compute per-split span statistics on the query side.

    For each HALvest-Contrastive subset in ``subsets``, samples ``n_samples``
    rows from the given split and computes distributional statistics over four
    metrics: tokens per span (pre- and post-truncation), fertility,
    punctuation density, and sentence-length standard deviation.

    The query column is detected at runtime via the convention
    ``("query", "anchor", "text", "sentence")``.  The split falls back from
    ``"valid"`` to ``"validation"`` and raises if neither exists.

    Args:
        subsets: HALvest-Contrastive subset names (e.g. ``["base-2", "base-4"]``).
        tokenizer_name: HuggingFace tokenizer identifier.
        max_length: Post-truncation token ceiling.
        n_samples: Number of query texts to sample per subset.
        split: Dataset split name; falls back from ``"valid"`` to ``"validation"``.
        halvest_name: HuggingFace dataset identifier.
        cache_dir: Optional HuggingFace cache directory.
        seed: Random seed for deterministic sampling.

    Returns:
        Dict keyed by subset name.  Each value is a dict with:
        ``n_samples``, ``n_skipped``, ``n_truncated``,
        ``tokens_pre_trunc_{mean,std,median}``,
        ``tokens_post_trunc_{mean,std,median}``,
        ``fertility_{mean,std,median}``,
        ``punct_density_{mean,std,median}``,
        ``sent_len_std_{mean,std,median}``,
        ``n_singleton_spans``.
    """
    import datasets as hf_datasets
    from transformers import AutoTokenizer

    from deep_stylometry.utils.text_stats import build_punct_token_id_set

    logger.info("Loading tokenizer '%s' …", tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Build special-token ID set for fertility / punct_density exclusion.
    special_ids: Set[int] = set()
    for attr in (
        "cls_token_id", "sep_token_id", "pad_token_id",
        "bos_token_id", "eos_token_id", "mask_token_id",
    ):
        val = getattr(tokenizer, attr, None)
        if val is not None:
            special_ids.add(val)

    logger.info("Building punctuation token-ID set …")
    punct_set = build_punct_token_id_set(tokenizer_name)

    rng = random.Random(seed)
    results: Dict[str, Dict[str, Any]] = {}

    for subset in subsets:
        logger.info("Loading subset '%s' split '%s' …", subset, split)
        ds_dict = hf_datasets.load_dataset(
            halvest_name, name=subset, cache_dir=cache_dir
        )
        ds: Optional[Any] = None
        for split_name in (split, "validation"):
            if split_name in ds_dict:
                ds = ds_dict[split_name]
                if split_name != split:
                    logger.warning(
                        "Split '%s' not found for %s; falling back to '%s'.",
                        split, subset, split_name,
                    )
                break
        if ds is None:
            raise ValueError(
                f"Neither '{split}' nor 'validation' split found for subset '{subset}'. "
                f"Available splits: {list(ds_dict.keys())}"
            )

        cols = ds.column_names
        query_col = next(
            (c for c in ("query", "anchor", "text", "sentence") if c in cols), None
        )
        if query_col is None:
            logger.warning(
                "No query column found for subset '%s' (cols=%s); skipping.", subset, cols
            )
            results[subset] = {"error": f"no query column in {cols}"}
            continue

        # Collect candidate texts, skipping empty / None rows.
        all_indices = list(range(len(ds)))
        rng.shuffle(all_indices)

        texts: List[str] = []
        n_skipped = 0
        for idx in all_indices:
            if len(texts) >= n_samples:
                break
            raw = ds[idx][query_col]
            if not raw or not str(raw).strip():
                n_skipped += 1
                continue
            texts.append(str(raw))

        actual_n = len(texts)
        logger.info(
            "Subset '%s': drew %d texts (wanted %d, skipped %d empty rows).",
            subset, actual_n, n_samples, n_skipped,
        )

        span_stats = _compute_span_stats_for_texts(
            texts=texts,
            tokenizer=tokenizer,
            max_length=max_length,
            punct_set=punct_set,
            special_ids=special_ids,
            label=subset,
        )
        span_stats["n_skipped"] = n_skipped
        results[subset] = span_stats

    return results


def pan19_span_stats(
    tokenizer_name: str = "answerdotai/ModernBERT-base",
    max_length: int = 512,
    n_samples: int = 1000,
    pan19_zip: Optional[str] = None,
    pan19_root: Optional[str] = None,
    language: str = "en",
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute span statistics on PAN19 unknown (query) texts.

    Reuses the existing ``PAN19Datamodule`` loading path.  The unknown texts
    correspond to the query side in the triplet conversion.

    Args:
        tokenizer_name: HuggingFace tokenizer identifier.
        max_length: Post-truncation token ceiling.
        n_samples: Maximum number of unknown texts to sample.
        pan19_zip: Path to PAN 2019 ZIP archive (preferred).
        pan19_root: Path to extracted PAN 2019 directory.
        language: Language filter (default ``"en"``).
        seed: Random seed for deterministic sampling.

    Returns:
        Stats dict in the same shape as one entry from :func:`per_split_span_stats`,
        with an added ``"n_samples_available"`` key.
    """
    from transformers import AutoTokenizer

    from deep_stylometry.utils.data.pan19_datamodule import PAN19Datamodule
    from deep_stylometry.utils.text_stats import build_punct_token_id_set

    if pan19_zip is None and pan19_root is None:
        raise ValueError("Provide pan19_zip or pan19_root for pan19_span_stats.")

    if pan19_zip is not None:
        problems = PAN19Datamodule._parse_problems_from_zip(pan19_zip, language=language)
    else:
        problems = PAN19Datamodule._parse_problems(pan19_root)

    all_texts = [prob["unknown_text"] for prob in problems if prob.get("unknown_text")]
    logger.info("PAN19: %d unknown texts available.", len(all_texts))

    rng = random.Random(seed)
    if len(all_texts) > n_samples:
        texts = rng.sample(all_texts, n_samples)
    else:
        texts = list(all_texts)
    actual_n = len(texts)
    logger.info("PAN19: using %d texts (seed=%d).", actual_n, seed)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    special_ids: Set[int] = set()
    for attr in (
        "cls_token_id", "sep_token_id", "pad_token_id",
        "bos_token_id", "eos_token_id", "mask_token_id",
    ):
        val = getattr(tokenizer, attr, None)
        if val is not None:
            special_ids.add(val)

    punct_set = build_punct_token_id_set(tokenizer_name)

    # Filter empty texts.
    n_skipped = 0
    clean_texts: List[str] = []
    for t in texts:
        if t and t.strip():
            clean_texts.append(t)
        else:
            n_skipped += 1

    span_stats = _compute_span_stats_for_texts(
        texts=clean_texts,
        tokenizer=tokenizer,
        max_length=max_length,
        punct_set=punct_set,
        special_ids=special_ids,
        label="pan19",
    )
    span_stats["n_skipped"] = n_skipped
    span_stats["n_samples_available"] = len(all_texts)
    return span_stats


def _span_stats_to_latex(span_stats: Dict[str, Dict[str, Any]]) -> str:
    """Render a LaTeX table fragment for per-split span statistics.

    Args:
        span_stats: Output of :func:`per_split_span_stats` with PAN19 merged
            under the key ``"pan19"``.

    Returns:
        LaTeX string with one table environment.
    """
    lines: List[str] = [
        r"\begin{table}[h]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Per-split span statistics (ModernBERT tokeniser, 512 max tokens)}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Split & $n$ & Tokens (post-trunc) & Sent/span & Tok/sent & Fertility & Punct density & Sent-len std \\",
        r"\midrule",
    ]
    for split, st in span_stats.items():
        if "error" in st:
            continue
        n = st.get("n_samples", "?")
        tok = f"{st.get('tokens_post_trunc_mean', 0):.1f} $\\pm$ {st.get('tokens_post_trunc_std', 0):.1f}"
        sps = f"{st.get('sentences_per_span_mean', 0):.2f} $\\pm$ {st.get('sentences_per_span_std', 0):.2f}"
        tps = f"{st.get('tokens_per_sentence_mean', 0):.1f} $\\pm$ {st.get('tokens_per_sentence_std', 0):.1f}"
        fert = f"{st.get('fertility_mean', 0):.3f} $\\pm$ {st.get('fertility_std', 0):.3f}"
        punc = f"{st.get('punct_density_mean', 0):.4f}"
        sls = f"{st.get('sent_len_std_mean', 0):.2f}"
        lines.append(f"{split} & {n} & {tok} & {sps} & {tps} & {fert} & {punc} & {sls} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# Entry point


def main() -> None:
    """Parse arguments and run dataset statistics computation."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Compute dataset statistics for the paper."
    )
    parser.add_argument(
        "--halvest-cache",
        default=None,
        help="HuggingFace cache directory for HALvest-Contrastive.",
    )
    parser.add_argument(
        "--pan19-zip",
        default=os.environ.get("PAN19_ZIP"),
        help="Path to PAN 2019 ZIP archive (preferred).",
    )
    parser.add_argument(
        "--pan19-root",
        default=os.environ.get("PAN19_ROOT"),
        help="Path to extracted PAN 2019 training directory (legacy).",
    )
    parser.add_argument(
        "--pan19-language",
        default="en",
        help="Language filter for PAN 2019 (default: en).",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write stats_output.json and stats_tables.tex.",
    )
    parser.add_argument(
        "--skip-halvest",
        action="store_true",
        help="Skip HALvest-Contrastive statistics (useful for quick PAN-only runs).",
    )
    parser.add_argument(
        "--skip-pan19",
        action="store_true",
        help="Skip PAN 2019 statistics.",
    )
    parser.add_argument(
        "--skip-pmi-breakdown",
        action="store_true",
        help="Skip per-subset and per-domain PMI breakdown (already included in "
             "--skip-halvest; use this flag to run HALvest stats without the "
             "slower PMI breakdown).",
    )
    parser.add_argument(
        "--pmi-breakdown-only",
        action="store_true",
        help="Run only the per-subset / per-domain PMI breakdown on the valid split "
             "and skip all other HALvest and PAN 2019 statistics. Useful when the "
             "other analyses have already been computed.",
    )
    # --- Span stats arguments ---
    parser.add_argument(
        "--skip-span-stats",
        action="store_true",
        help="Skip per-split span statistics (tokenizer-only, CPU-fast).",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["base-2", "base-4", "base-6", "base-8", "base-10"],
        help="HALvest-Contrastive subsets to analyse (default: all 5).",
    )
    parser.add_argument(
        "--tokenizer-name",
        default="answerdotai/ModernBERT-base",
        help="Tokenizer for span statistics (default: answerdotai/ModernBERT-base).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Post-truncation token ceiling for span statistics (default: 512).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Texts per subset for span statistics (default: 1000).",
    )
    parser.add_argument(
        "--span-split",
        default="valid",
        help="Dataset split for span statistics (default: valid; falls back to validation).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_stats: Dict[str, Any] = {}

    if args.pmi_breakdown_only:
        logger.info("Running PMI breakdown only (valid split) …")
        try:
            all_stats["halvest"] = {
                "pmi_by_subset_and_domain": halvest_pmi_by_subset_and_domain(
                    cache_dir=args.halvest_cache,
                )
            }
        except Exception as exc:
            logger.error("PMI breakdown failed: %s", exc)
            all_stats["halvest"] = {"error": str(exc)}

        json_path = os.path.join(args.output_dir, "pmi_breakdown.json")
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(all_stats, fh, indent=2, default=str)
        logger.info("Wrote %s", json_path)
        print(json.dumps(all_stats, indent=2, default=str))
        return

    if not args.skip_halvest:
        logger.info("Computing HALvest-Contrastive statistics …")
        try:
            all_stats["halvest"] = halvest_statistics(
                cache_dir=args.halvest_cache,
                skip_pmi_breakdown=args.skip_pmi_breakdown,
            )
        except Exception as exc:
            logger.error("HALvest statistics failed: %s", exc)
            all_stats["halvest"] = {"error": str(exc)}

    if not args.skip_pan19:
        if not args.pan19_zip and not args.pan19_root:
            logger.error(
                "PAN 2019 source not specified. Set --pan19-zip (ZIP archive) "
                "or --pan19-root (extracted directory), or the corresponding env vars."
            )
            all_stats["pan19"] = {"error": "pan19 source not set"}
        else:
            logger.info("Computing PAN 2019 statistics …")
            try:
                all_stats["pan19"] = pan19_statistics(
                    pan19_zip=args.pan19_zip,
                    pan19_root=args.pan19_root,
                    language=args.pan19_language,
                )
            except Exception as exc:
                logger.error("PAN 2019 statistics failed: %s", exc)
                all_stats["pan19"] = {"error": str(exc)}

    # --- Span statistics ---
    if not args.skip_span_stats and not args.pmi_breakdown_only:
        logger.info("Computing per-split span statistics …")
        try:
            span_stats = per_split_span_stats(
                subsets=args.subsets,
                tokenizer_name=args.tokenizer_name,
                max_length=args.max_length,
                n_samples=args.n_samples,
                split=args.span_split,
                halvest_name="almanach/HALvest-Contrastive",
                cache_dir=args.halvest_cache,
            )
        except Exception as exc:
            logger.error("Per-split span stats failed: %s", exc)
            span_stats = {"error": str(exc)}

        if not args.skip_pan19 and (args.pan19_zip or args.pan19_root):
            try:
                span_stats["pan19"] = pan19_span_stats(
                    tokenizer_name=args.tokenizer_name,
                    max_length=args.max_length,
                    n_samples=args.n_samples,
                    pan19_zip=args.pan19_zip,
                    pan19_root=args.pan19_root,
                    language=args.pan19_language,
                )
            except Exception as exc:
                logger.error("PAN19 span stats failed: %s", exc)
                span_stats["pan19"] = {"error": str(exc)}

        all_stats["per_split_span_stats"] = span_stats
    else:
        span_stats = {}

    # Write JSON
    json_path = os.path.join(args.output_dir, "stats_output.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(all_stats, fh, indent=2, sort_keys=True, default=str)
    logger.info("Wrote %s", json_path)

    # Write LaTeX (base tables + span stats table appended)
    tex = _to_latex(
        all_stats.get("halvest", {}),
        all_stats.get("pan19", {}),
    )
    if span_stats and "error" not in span_stats:
        tex += "\n\n" + _span_stats_to_latex(span_stats)

    tex_path = os.path.join(args.output_dir, "stats_tables.tex")
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write(tex)
    logger.info("Wrote %s", tex_path)

    # Write figures
    try:
        from deep_stylometry.experiments.visualizations import (
            plot_jaccard,
            plot_pan19_word_lengths,
            plot_trigram_entropy,
        )

        h = all_stats.get("halvest", {})
        if "trigram_entropy_per_domain" in h:
            plot_trigram_entropy(h["trigram_entropy_per_domain"], args.output_dir)
        if "jaccard" in h:
            plot_jaccard(h["jaccard"], args.output_dir)

        p = all_stats.get("pan19", {})
        if p and "error" not in p:
            # Re-parse problems for per-item histograms when a source is available
            pan19_problems: Optional[List[Any]] = None
            try:
                from deep_stylometry.utils.data.pan19_datamodule import PAN19Datamodule
                if args.pan19_zip:
                    pan19_problems = PAN19Datamodule._parse_problems_from_zip(
                        args.pan19_zip, language=args.pan19_language
                    )
                elif args.pan19_root:
                    pan19_problems = PAN19Datamodule._parse_problems(args.pan19_root)
            except Exception as _exc:
                logger.debug("Could not re-parse problems for histogram: %s", _exc)
            plot_pan19_word_lengths(p, problems=pan19_problems,
                                    output_dir=args.output_dir)
    except Exception as exc:
        logger.warning("Visualization step failed: %s", exc)

    # Print to stdout
    print(json.dumps(all_stats, indent=2, default=str))


if __name__ == "__main__":
    main()
