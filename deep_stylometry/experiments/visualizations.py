#!/usr/bin/env python3
# deep_stylometry/experiments/visualizations.py
"""Paper-quality visualization helpers for dataset statistics and per-domain evaluation.

All figures are exported as PDF (vector) with layout tuned for double-column
academic papers (IEEE / ACL style).

  Single-column width  ≈ 3.35 in
  Full text width      ≈ 6.85 in

Each public function returns the absolute path to the written PDF.

Usage::

    from deep_stylometry.experiments.visualizations import (
        plot_trigram_entropy,
        plot_jaccard,
        plot_pan19_word_lengths,
        plot_per_domain_heatmap,
        plot_per_domain_bars,
    )
"""

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for HPC / headless
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Layout constants

_COL_W  = 3.35   # single-column width (inches)
_FULL_W = 6.85   # full text width — spans both columns

# Font sizes calibrated for 9/10 pt body text (IEEE / ACL):
# labels/ticks at 7 pt, axis labels at 8 pt, panel titles at 9 pt.
_FONT_SM = 7
_FONT_MD = 8
_FONT_LG = 9

# Wong (2011) colorblind-safe palette
_PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # amber
    "#009E73",  # green
    "#D55E00",  # vermilion
    "#CC79A7",  # purple-pink
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
]


# Internal helpers


@contextmanager
def _paper_rc():
    """Apply paper-quality rcParams for the duration of the with-block."""
    rc = {
        "font.family":        "serif",
        "font.size":          _FONT_SM,
        "axes.labelsize":     _FONT_MD,
        "axes.titlesize":     _FONT_LG,
        "xtick.labelsize":    _FONT_SM,
        "ytick.labelsize":    _FONT_SM,
        "legend.fontsize":    _FONT_SM,
        # Embed fonts as TrueType — editable in Illustrator / Inkscape
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        # Minimal chrome
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.3,
        "grid.linewidth":     0.5,
        "lines.linewidth":    1.0,
    }
    with plt.rc_context(rc):
        yield


def _savefig(fig: plt.Figure, path: str) -> None:
    """Save *fig* to *path* as PDF and close the figure."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info("Saved figure → %s", path)


# 1. HALvest — trigram entropy per domain


def plot_trigram_entropy(
    entropy_dict: Dict[str, float],
    output_dir: str,
    filename: str = "fig_halvest_entropy.pdf",
) -> str:
    """Horizontal bar chart of character-trigram Shannon entropy per HALvest domain.

    Bars are sorted by ascending entropy so the figure reads bottom-to-top from
    lowest to highest stylistic diversity.

    Args:
        entropy_dict: Mapping ``domain → entropy (bits)`` from
                      :func:`halvest_statistics`.
        output_dir: Directory to write the PDF.
        filename: Output filename (default ``fig_halvest_entropy.pdf``).

    Returns:
        Absolute path to the written PDF, or ``""`` if *entropy_dict* is empty.
    """
    if not entropy_dict:
        return ""

    domains = sorted(entropy_dict, key=lambda d: entropy_dict[d])
    values  = [entropy_dict[d] for d in domains]

    with _paper_rc():
        fig_h = max(1.8, 0.28 * len(domains) + 0.5)
        fig, ax = plt.subplots(figsize=(_COL_W, fig_h))

        y    = np.arange(len(domains))
        bars = ax.barh(y, values, color=_PALETTE[0], edgecolor="none", height=0.6)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + max(values) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                va="center", ha="left",
                fontsize=_FONT_SM - 1,
            )

        ax.set_yticks(y)
        ax.set_yticklabels(domains)
        ax.set_xlabel("Char-trigram entropy (bits)")
        ax.set_xlim(0, max(values) * 1.18)
        ax.grid(axis="x", alpha=0.3)
        ax.grid(axis="y", alpha=0.0)

    path = os.path.join(output_dir, filename)
    _savefig(fig, path)
    return path


# 2. HALvest — Jaccard similarity (Q-P vs Q-N)


def plot_jaccard(
    jaccard_stats: Dict[str, float],
    output_dir: str,
    filename: str = "fig_halvest_jaccard.pdf",
) -> str:
    """Paired bar chart: mean Jaccard similarity for query-positive vs. query-negative.

    Error bars show ±1 standard deviation.  A large gap between the two bars
    signals strong lexical separation between same-author and cross-author text pairs.

    Args:
        jaccard_stats: Dict with keys ``mean_jaccard_qp``, ``std_jaccard_qp``,
                       ``mean_jaccard_qn``, ``std_jaccard_qn``
                       (output of :func:`halvest_statistics`).
        output_dir: Directory to write the PDF.
        filename: Output filename.

    Returns:
        Absolute path to the written PDF.
    """
    mean_qp = jaccard_stats.get("mean_jaccard_qp", 0.0)
    std_qp  = jaccard_stats.get("std_jaccard_qp",  0.0)
    mean_qn = jaccard_stats.get("mean_jaccard_qn", 0.0)
    std_qn  = jaccard_stats.get("std_jaccard_qn",  0.0)

    with _paper_rc():
        fig, ax = plt.subplots(figsize=(_COL_W, 2.0))

        labels  = ["Query–Positive", "Query–Negative"]
        means   = [mean_qp, mean_qn]
        stds    = [std_qp,  std_qn]
        colors  = [_PALETTE[0], _PALETTE[3]]  # blue, vermilion

        x    = np.array([0.0, 0.55])
        bars = ax.bar(
            x, means, width=0.38, yerr=stds, capsize=3,
            color=colors, edgecolor="none",
            error_kw=dict(elinewidth=0.9, ecolor="#444444"),
        )

        max_top = max(m + s for m, s in zip(means, stds))
        for bar, val in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_top * 0.04,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=_FONT_SM,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Jaccard similarity")
        ax.set_ylim(0, max_top * 1.25)
        ax.grid(axis="x", alpha=0.0)

    path = os.path.join(output_dir, filename)
    _savefig(fig, path)
    return path


# 3. PAN 2019 — word-length distributions


def plot_pan19_word_lengths(
    pan19_stats: Dict[str, Any],
    problems: Optional[List[Dict]] = None,
    output_dir: str = ".",
    filename: str = "fig_pan19_wordlengths.pdf",
) -> str:
    """Side-by-side word-length distribution for PAN 2019 unknown and candidate texts.

    When the raw *problems* list is provided (from
    :func:`PAN19Datamodule._parse_problems_from_zip`), the function draws
    proper histograms.  Otherwise it falls back to a summary bar chart using
    the aggregate statistics from :func:`pan19_statistics`.

    Args:
        pan19_stats: Output of :func:`pan19_statistics`.
        problems: Optional raw problem list; enables per-item histograms.
        output_dir: Directory to write the PDF.
        filename: Output filename.

    Returns:
        Absolute path to the written PDF.
    """
    with _paper_rc():
        if problems:
            # Per-item histograms
            unk_lens  = [len(p["unknown_text"].split()) for p in problems]
            cand_lens: List[int] = []
            seen: set = set()
            for p in problems:
                if p["problem_id"] not in seen:
                    seen.add(p["problem_id"])
                    for text in p["candidates"].values():
                        cand_lens.append(len(text.split()))

            fig, (ax_u, ax_c) = plt.subplots(
                1, 2, figsize=(_FULL_W, 2.0), sharey=False
            )

            def _n_bins(vals: List[int]) -> int:
                if not vals:
                    return 8
                return max(8, int(np.ceil(np.log2(len(vals)) + 1)))

            ax_u.hist(
                unk_lens, bins=_n_bins(unk_lens),
                color=_PALETTE[0], edgecolor="white", linewidth=0.3,
            )
            ax_u.axvline(
                float(np.mean(unk_lens)), color=_PALETTE[3],
                linestyle="--", linewidth=0.9,
                label=f"Mean\u2009=\u2009{np.mean(unk_lens):.0f}",
            )
            ax_u.set_xlabel("Word count")
            ax_u.set_ylabel("Unknown texts")
            ax_u.legend(frameon=False)

            ax_c.hist(
                cand_lens, bins=_n_bins(cand_lens),
                color=_PALETTE[1], edgecolor="white", linewidth=0.3,
            )
            ax_c.axvline(
                float(np.mean(cand_lens)), color=_PALETTE[3],
                linestyle="--", linewidth=0.9,
                label=f"Mean\u2009=\u2009{np.mean(cand_lens):.0f}",
            )
            ax_c.set_xlabel("Word count")
            ax_c.set_ylabel("Candidate texts")
            ax_c.legend(frameon=False)

            fig.tight_layout(pad=0.4, w_pad=0.6)

        else:
            # Summary bar chart from aggregate stats
            unk_s  = pan19_stats.get("unknown_text_word_length", {})
            cand_s = pan19_stats.get("candidate_text_word_length", {})

            fig, ax = plt.subplots(figsize=(_COL_W, 2.0))

            categories = ["Unknown text", "Candidate text"]
            means = [unk_s.get("mean", 0.0), cand_s.get("mean", 0.0)]
            mins_ = [unk_s.get("min",  0.0), cand_s.get("min",  0.0)]
            maxs_ = [unk_s.get("max",  0.0), cand_s.get("max",  0.0)]
            x     = np.array([0.0, 0.55])

            bars = ax.bar(x, means, width=0.38,
                          color=[_PALETTE[0], _PALETTE[1]], edgecolor="none")

            # Min/max range markers
            for xi, lo, hi in zip(x, mins_, maxs_):
                ax.plot([xi, xi], [lo, hi], color="#555", linewidth=0.8)
                for y_end in (lo, hi):
                    ax.plot([xi - 0.06, xi + 0.06], [y_end, y_end],
                            color="#555", linewidth=0.8)

            for bar, val in zip(bars, means):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.04,
                    f"{val:.0f}",
                    ha="center", va="bottom", fontsize=_FONT_SM,
                )

            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.set_ylabel("Word count")
            ax.grid(axis="x", alpha=0.0)

    path = os.path.join(output_dir, filename)
    _savefig(fig, path)
    return path


# 4. Per-domain retrieval — heatmap


def plot_per_domain_heatmap(
    results: Dict[str, Dict[str, float]],
    output_dir: str,
    metric_keys: Optional[List[str]] = None,
    filename: str = "fig_per_domain_heatmap.pdf",
) -> str:
    """Heatmap of per-domain retrieval metrics (domains × metrics).

    Cell colours are column-normalised to [0, 1] so that relative differences
    across domains are easy to read at a glance; actual numeric values are
    annotated inside each cell.

    Args:
        results: Output of :func:`compute_per_domain_metrics`.
        output_dir: Directory to write the PDF.
        metric_keys: Ordered list of metric names to display.  Defaults to
                     ``[accuracy, mrr@5, mrr@10, ndcg@10, recall@10]`` when
                     those keys exist, otherwise all non-count metrics.
        filename: Output filename.

    Returns:
        Absolute path to the written PDF, or ``""`` if *results* is empty.
    """
    if not results:
        return ""

    if metric_keys is None:
        preferred = ["accuracy", "mrr@5", "mrr@10", "ndcg@10", "recall@10"]
        first     = next(iter(results.values()))
        metric_keys = [m for m in preferred if m in first]
        if not metric_keys:
            metric_keys = [k for k in first if k != "n_queries"]

    domains = sorted(results.keys())
    n_d, n_m = len(domains), len(metric_keys)

    matrix = np.array(
        [[results[d].get(m, 0.0) for m in metric_keys] for d in domains]
    )  # (n_d, n_m)

    # Column-wise normalisation for colour only
    col_min  = matrix.min(axis=0, keepdims=True)
    col_max  = matrix.max(axis=0, keepdims=True)
    col_rng  = np.where(col_max - col_min > 1e-9, col_max - col_min, 1.0)
    normed   = (matrix - col_min) / col_rng

    with _paper_rc():
        cell_h  = 0.30
        cell_w  = 0.78
        fig_h   = max(1.6, n_d * cell_h + 0.7)
        fig_w   = min(_FULL_W, max(_COL_W, n_m * cell_w + 1.4))

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(normed, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)

        for i in range(n_d):
            for j in range(n_m):
                text_col = "white" if normed[i, j] > 0.55 else "#1a1a1a"
                ax.text(j, i, f"{matrix[i, j]:.3f}",
                        ha="center", va="center",
                        fontsize=_FONT_SM - 1, color=text_col)

        ax.set_xticks(range(n_m))
        ax.set_xticklabels(metric_keys, rotation=30, ha="right")
        ax.set_yticks(range(n_d))
        ax.set_yticklabels(domains)
        ax.tick_params(axis="both", which="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.ax.tick_params(labelsize=_FONT_SM - 1)
        cbar.set_label("Col.-normalised score", fontsize=_FONT_SM - 1)
        cbar.set_ticks([0, 0.5, 1.0])

    path = os.path.join(output_dir, filename)
    _savefig(fig, path)
    return path


# 5. Per-domain retrieval — grouped bar chart


def plot_per_domain_bars(
    results: Dict[str, Dict[str, float]],
    output_dir: str,
    metrics: Tuple[str, ...] = ("accuracy", "mrr@10", "ndcg@10"),
    filename: str = "fig_per_domain_bars.pdf",
) -> str:
    """Grouped bar chart of selected retrieval metrics per domain.

    Domains are on the x-axis, one bar group per domain; each group contains
    one bar per selected metric.  Suitable for comparing a small number of
    metrics across domains at a glance.

    Args:
        results: Output of :func:`compute_per_domain_metrics`.
        output_dir: Directory to write the PDF.
        metrics: Metrics to include as grouped bars.
        filename: Output filename.

    Returns:
        Absolute path to the written PDF, or ``""`` if *results* is empty.
    """
    if not results:
        return ""

    # Filter to metrics actually present
    first   = next(iter(results.values()))
    metrics = tuple(m for m in metrics if m in first)
    if not metrics:
        return ""

    domains = sorted(results.keys())
    n_d, n_m = len(domains), len(metrics)

    data = np.array(
        [[results[d].get(m, 0.0) for m in metrics] for d in domains]
    )  # (n_d, n_m)

    with _paper_rc():
        bar_w   = min(0.22, 0.7 / n_m)
        grp_w   = n_m * bar_w + 0.15
        fig_w   = min(_FULL_W, max(_COL_W, n_d * grp_w + 0.9))
        fig, ax = plt.subplots(figsize=(fig_w, 2.2))

        x       = np.arange(n_d, dtype=float)
        offsets = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * bar_w

        for j, (metric, offset) in enumerate(zip(metrics, offsets)):
            ax.bar(
                x + offset, data[:, j], width=bar_w,
                label=metric,
                color=_PALETTE[j % len(_PALETTE)],
                edgecolor="none",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(domains, rotation=30, ha="right")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.08)
        ax.legend(
            frameon=False, ncol=n_m, fontsize=_FONT_SM,
            loc="upper right", handlelength=1.0, columnspacing=0.7,
        )
        ax.grid(axis="x", alpha=0.0)

    path = os.path.join(output_dir, filename)
    _savefig(fig, path)
    return path
