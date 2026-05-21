# deep_stylometry/experiments/mechanistic/plotting.py
"""All figure generation for the mechanistic study."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_COLORS = {
    "mean": "#1f77b4",
    "layerwise": "#9467bd",
    "li": "#ff7f0e",
    "pli_ngram2": "#2ca02c",
    "e5": "#d62728",
}

_MODEL_LABELS = {
    "mean": "Mean Pooling",
    "layerwise": "Layerwise Attention",
    "li": "Late Interaction",
    "pli_ngram2": "PLI n-gram 2",
    "e5": "Multilingual-E5",
}


def _get_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ---------------------------------------------------------------------------
# Phase 0: length distribution
# ---------------------------------------------------------------------------

def plot_probe_set_lengths(root: Path) -> None:
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    probe_path = root / "phase0_probe_set" / "probe_set.json"
    if not probe_path.exists():
        logger.warning("Probe set not found; skipping length plot.")
        return

    with open(probe_path) as f:
        entries = json.load(f)

    tiers = ["A", "B", "C"]
    fig, axes = plt.subplots(1, len(tiers), figsize=(12, 4), sharey=True)

    for ax, tier in zip(axes, tiers):
        t_entries = [e for e in entries if e["tier"] == tier]
        if not t_entries:
            ax.set_title(f"Tier {tier} (empty)")
            continue
        pos_lens = [e["positive_token_len"] for e in t_entries]
        neg_lens = [e["negative_token_len"] for e in t_entries]
        ax.hist(pos_lens, alpha=0.7, label="positive", bins=15, color="steelblue")
        ax.hist(neg_lens, alpha=0.7, label="negative", bins=15, color="coral")
        ax.axvline(130, color="black", linestyle="--", linewidth=1, label="target=130")
        ax.set_title(f"Tier {tier} (n={len(t_entries)})")
        ax.set_xlabel("Token length")
        ax.legend(fontsize=7)

    axes[0].set_ylabel("Count")
    fig.suptitle("Probe Set Token Length Distribution")
    fig.tight_layout()
    out = root / "figures" / "phase0_length_distribution.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Phase 1: LISA R² heatmaps
# ---------------------------------------------------------------------------

def plot_probe_heatmap(
    root: Path,
    model_id: str,
    step: str = "final",
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from deep_stylometry.experiments.mechanistic.io_utils import phase1_probe_path

    probe_path = phase1_probe_path(root, model_id, step, "probes.npz")
    if not probe_path.exists():
        logger.warning("Probes not found at %s", probe_path)
        return

    data = np.load(probe_path, allow_pickle=True)
    r2 = data["r2_per_layer_per_feature"]         # (n_layers, n_features)
    feat_names = list(data["feature_names"])

    # Aggregate into categories for readability
    categories = {
        "func_words": [i for i, n in enumerate(feat_names) if n.startswith("fw_")],
        "sent_len": [i for i, n in enumerate(feat_names) if n.startswith("sl_")],
        "punct": [i for i, n in enumerate(feat_names) if n.startswith("punct_")],
        "cap": [i for i, n in enumerate(feat_names) if n.startswith("cap_")],
        "ttr": [i for i, n in enumerate(feat_names) if n.startswith("ttr_")],
        "word_len": [i for i, n in enumerate(feat_names) if n.startswith("wl_")],
        "hedging": [i for i, n in enumerate(feat_names) if n == "hedging"],
        "citations": [i for i, n in enumerate(feat_names) if n == "citations"],
        "pos_bigrams": [i for i, n in enumerate(feat_names) if n.startswith("posbg_")],
        "discourse": [i for i, n in enumerate(feat_names) if n == "discourse"],
        "dep_depth": [i for i, n in enumerate(feat_names) if n == "dep_depth"],
    }

    # Per-category mean R²
    cat_names = []
    cat_r2 = []
    for cat, idxs in categories.items():
        if idxs:
            cat_names.append(cat)
            cat_r2.append(r2[:, idxs].mean(axis=1))  # (n_layers,)

    if not cat_names:
        logger.warning("No feature categories found.")
        return

    grid = np.stack(cat_r2, axis=1)  # (n_layers, n_cats)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        grid.T,
        ax=ax,
        xticklabels=[str(i) for i in range(grid.shape[0])],
        yticklabels=cat_names,
        cmap="YlOrRd",
        vmin=0.0,
        vmax=max(0.5, float(grid.max())),
        annot=False,
    )
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Feature category")
    ax.set_title(f"LISA probe R² — {_MODEL_LABELS.get(model_id, model_id)} ({step})")

    out = root / "figures" / f"phase1_probe_heatmap_{model_id}_{step}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_cross_model_probe_comparison(root: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from deep_stylometry.experiments.mechanistic.io_utils import phase1_probe_path

    model_ids = ["layerwise", "li", "pli_ngram2"]
    fig, axes = plt.subplots(1, len(model_ids), figsize=(20, 5), sharey=True)

    for ax, model_id in zip(axes, model_ids):
        probe_path = phase1_probe_path(root, model_id, "final", "probes.npz")
        if not probe_path.exists():
            ax.set_title(f"{model_id} (missing)")
            continue
        data = np.load(probe_path, allow_pickle=True)
        r2 = data["r2_per_layer_per_feature"]
        max_r2_per_layer = r2.max(axis=1)  # (n_layers,)
        ax.plot(range(len(max_r2_per_layer)), max_r2_per_layer,
                color=_MODEL_COLORS.get(model_id, "black"), linewidth=2)
        ax.set_title(_MODEL_LABELS.get(model_id, model_id))
        ax.set_xlabel("Layer index")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)

    axes[0].set_ylabel("Max R² across features")
    fig.suptitle("LISA probe R² by layer (final checkpoint)")
    fig.tight_layout()
    out = root / "figures" / "phase1_cross_model_comparison.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Phase 2: recovery curves
# ---------------------------------------------------------------------------

def plot_recovery_curve(
    root: Path,
    model_id: str,
    step: str = "final",
    tier: str = "A",
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from deep_stylometry.experiments.mechanistic.io_utils import phase2_path

    rec_path = phase2_path(root, model_id, step, "recovery.npz")
    if not rec_path.exists():
        logger.warning("Recovery not found at %s", rec_path)
        return

    data = np.load(rec_path, allow_pickle=True)
    recovery = data["recovery"]         # (n_triplets, n_layers)
    tier_labels = list(data["tier_labels"])

    tier_idx = [i for i, t in enumerate(tier_labels) if t == tier]
    if not tier_idx:
        logger.warning("No tier %s entries in %s", tier, rec_path)
        return

    rec_tier = recovery[tier_idx]
    mean_rec = rec_tier.mean(axis=0)
    std_rec = rec_tier.std(axis=0)
    layers = np.arange(len(mean_rec))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, mean_rec, color=_MODEL_COLORS.get(model_id, "black"), linewidth=2,
            label=_MODEL_LABELS.get(model_id, model_id))
    ax.fill_between(layers, mean_rec - std_rec, mean_rec + std_rec,
                    color=_MODEL_COLORS.get(model_id, "black"), alpha=0.2)
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, label="50% recovery")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Patch layer index")
    ax.set_ylabel("Recovery (%)")
    ax.set_title(f"Recovery curve — {_MODEL_LABELS.get(model_id, model_id)} Tier {tier} ({step})")
    ax.legend()

    out = root / "figures" / f"phase2_recovery_{model_id}_tier{tier}_{step}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_all_models_recovery(root: Path, step: str = "final", tier: str = "A") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from deep_stylometry.experiments.mechanistic.io_utils import phase2_path

    fig, ax = plt.subplots(figsize=(10, 6))

    for model_id in ["layerwise", "li", "pli_ngram2", "e5"]:
        rec_path = phase2_path(root, model_id, step, "recovery.npz")
        if not rec_path.exists():
            continue
        data = np.load(rec_path, allow_pickle=True)
        recovery = data["recovery"]
        tl = list(data["tier_labels"])
        t_idx = [i for i, t in enumerate(tl) if t == tier]
        if not t_idx:
            continue
        rec_tier = recovery[t_idx]
        mean_rec = rec_tier.mean(axis=0)
        std_rec = rec_tier.std(axis=0)
        layers = np.arange(len(mean_rec))
        color = _MODEL_COLORS.get(model_id, "black")
        ax.plot(layers, mean_rec, color=color, linewidth=2,
                label=_MODEL_LABELS.get(model_id, model_id))
        ax.fill_between(layers, mean_rec - std_rec, mean_rec + std_rec,
                        color=color, alpha=0.15)

    ax.axhline(50, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Patch layer index")
    ax.set_ylabel("Recovery (%)")
    ax.set_title(f"Recovery curves — all models, Tier {tier}")
    ax.legend()

    out = root / "figures" / f"phase2_recovery_all_models_tier{tier}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


def _rank_recovery_for_tier(
    recovery: np.ndarray,
    clean_scores: np.ndarray,
    corrupt_scores: np.ndarray,
    tier_labels: list,
    tier: str,
):
    """Return (frac_per_layer, n_correct) for triplets in `tier`."""
    tier_idx = [i for i, t in enumerate(tier_labels) if t == tier]
    if not tier_idx:
        return None, 0
    rec = recovery[tier_idx]
    cs = clean_scores[tier_idx]
    cc = corrupt_scores[tier_idx]
    gap = cs - cc
    correct_mask = gap > 1e-8
    rank_recovered = (rec > 0) & correct_mask[:, None]
    n_correct = int(correct_mask.sum())
    if n_correct > 0:
        frac = rank_recovered[correct_mask].astype(float).mean(axis=0)
    else:
        frac = np.zeros(rec.shape[1])
    return frac, n_correct


def plot_rank_recovery_curve(
    root: Path,
    model_id: str,
    step: str = "final",
    tier: str = "A",
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from deep_stylometry.experiments.mechanistic.io_utils import phase2_path

    rec_path = phase2_path(root, model_id, step, "recovery.npz")
    if not rec_path.exists():
        logger.warning("Recovery not found at %s", rec_path)
        return

    data = np.load(rec_path, allow_pickle=True)
    frac, n_correct = _rank_recovery_for_tier(
        data["recovery"], data["clean_scores"], data["corrupt_scores"],
        list(data["tier_labels"]), tier,
    )
    if frac is None:
        logger.warning("No tier %s entries in %s", tier, rec_path)
        return

    layers = np.arange(len(frac))
    color = _MODEL_COLORS.get(model_id, "black")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, frac, color=color, linewidth=2,
            label=_MODEL_LABELS.get(model_id, model_id))
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance (0.5)")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Patch layer index")
    ax.set_ylabel("Fraction rank-recovered")
    ax.set_title(
        f"Rank recovery — {_MODEL_LABELS.get(model_id, model_id)} Tier {tier} ({step})"
        f"\n(n_correct={n_correct})"
    )
    ax.legend()

    out = root / "figures" / f"phase2_rank_recovery_{model_id}_tier{tier}_{step}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_all_models_rank_recovery(
    root: Path, step: str = "final", tier: str = "A"
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from deep_stylometry.experiments.mechanistic.io_utils import phase2_path

    fig, ax = plt.subplots(figsize=(10, 6))

    for model_id in ["layerwise", "li", "pli_ngram2", "e5"]:
        rec_path = phase2_path(root, model_id, step, "recovery.npz")
        if not rec_path.exists():
            continue
        data = np.load(rec_path, allow_pickle=True)
        frac, n_correct = _rank_recovery_for_tier(
            data["recovery"], data["clean_scores"], data["corrupt_scores"],
            list(data["tier_labels"]), tier,
        )
        if frac is None:
            continue
        color = _MODEL_COLORS.get(model_id, "black")
        ax.plot(np.arange(len(frac)), frac, color=color, linewidth=2,
                label=f"{_MODEL_LABELS.get(model_id, model_id)} (n={n_correct})")

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Patch layer index")
    ax.set_ylabel("Fraction rank-recovered")
    ax.set_title(f"Rank recovery — all models, Tier {tier}")
    ax.legend()

    out = root / "figures" / f"phase2_rank_recovery_all_models_tier{tier}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_rank_recovery_tier_comparison(root: Path, model_id: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from deep_stylometry.experiments.mechanistic.io_utils import phase2_path

    rec_path = phase2_path(root, model_id, "final", "recovery.npz")
    if not rec_path.exists():
        logger.warning("Recovery not found at %s", rec_path)
        return

    data = np.load(rec_path, allow_pickle=True)
    recovery = data["recovery"]
    clean_scores = data["clean_scores"]
    corrupt_scores = data["corrupt_scores"]
    tier_labels = list(data["tier_labels"])

    tiers = ["A", "B", "C"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    color = _MODEL_COLORS.get(model_id, "black")

    for ax, tier in zip(axes, tiers):
        frac, n_correct = _rank_recovery_for_tier(
            recovery, clean_scores, corrupt_scores, tier_labels, tier,
        )
        if frac is None:
            ax.set_title(f"Tier {tier} (empty)")
            continue
        layers = np.arange(len(frac))
        ax.plot(layers, frac, color=color, linewidth=2)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"Tier {tier} (n_correct={n_correct})")
        ax.set_xlabel("Layer index")

    axes[0].set_ylabel("Fraction rank-recovered")
    fig.suptitle(f"Rank recovery by tier — {_MODEL_LABELS.get(model_id, model_id)}")
    fig.tight_layout()
    out = root / "figures" / f"phase2_rank_tier_comparison_{model_id}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_score_sensitivity(
    root: Path, step: str = "final", tier: str = "A"
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from deep_stylometry.experiments.mechanistic.io_utils import phase2_path

    fig, ax = plt.subplots(figsize=(10, 6))

    for model_id in ["layerwise", "li", "pli_ngram2", "e5"]:
        rec_path = phase2_path(root, model_id, step, "recovery.npz")
        if not rec_path.exists():
            continue
        data = np.load(rec_path, allow_pickle=True)
        recovery = data["recovery"]
        clean_scores = data["clean_scores"]
        corrupt_scores = data["corrupt_scores"]
        tier_labels = list(data["tier_labels"])

        tier_idx = [i for i, t in enumerate(tier_labels) if t == tier]
        if not tier_idx:
            continue

        rec = recovery[tier_idx]
        cs = clean_scores[tier_idx]
        cc = corrupt_scores[tier_idx]
        gap = cs - cc
        patched_scores = cc[:, None] + (rec / 100.0) * gap[:, None]
        raw_delta = np.abs(patched_scores - cc[:, None])
        mean_delta = raw_delta.mean(axis=0)

        color = _MODEL_COLORS.get(model_id, "black")
        ax.plot(np.arange(len(mean_delta)), mean_delta, color=color, linewidth=2,
                label=_MODEL_LABELS.get(model_id, model_id))

    ax.set_xlabel("Patch layer index")
    ax.set_ylabel("Mean |score change|")
    ax.set_title(f"Score sensitivity — all models, Tier {tier}")
    ax.legend()

    out = root / "figures" / f"phase2_score_sensitivity_tier{tier}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Phase 3: dynamics panel
# ---------------------------------------------------------------------------

def plot_dynamics_panel(root: Path, model_id: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from deep_stylometry.experiments.mechanistic.io_utils import phase2_path, _step_label

    # Collect all checkpoint recovery curves
    steps = [0, 500, 1500, 3000, 5000, 10000, 20000, "final"]
    curves = {}
    for step in steps:
        rec_path = phase2_path(root, model_id, step, "recovery.npz")
        if rec_path.exists():
            data = np.load(rec_path, allow_pickle=True)
            recovery = data["recovery"]
            tl = list(data["tier_labels"])
            t_idx = [i for i, t in enumerate(tl) if t == "A"]
            if t_idx:
                curves[step] = recovery[t_idx].mean(axis=0)

    if not curves:
        logger.warning("No recovery curves for %s", model_id)
        return

    n_plots = len(curves)
    fig, axes = plt.subplots(1, n_plots, figsize=(3 * n_plots, 4), sharey=True)
    if n_plots == 1:
        axes = [axes]

    for ax, (step, curve) in zip(axes, curves.items()):
        ax.plot(np.arange(len(curve)), curve,
                color=_MODEL_COLORS.get(model_id, "black"), linewidth=1.5)
        ax.axhline(50, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(f"step={step}", fontsize=8)
        ax.set_xlabel("Layer", fontsize=7)

    axes[0].set_ylabel("Mean recovery (%)")
    fig.suptitle(f"Training dynamics — {_MODEL_LABELS.get(model_id, model_id)}", fontsize=10)
    fig.tight_layout()
    out = root / "figures" / f"phase3_dynamics_{model_id}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Phase 4: distractor analysis
# ---------------------------------------------------------------------------

def plot_distractor_recovery(root: Path, model_id: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from deep_stylometry.experiments.mechanistic.io_utils import phase4_path, phase2_path

    causal_path = phase4_path(root, model_id, "causal.npz")
    if not causal_path.exists():
        logger.warning("Causal data not found for %s", model_id)
        return

    data = np.load(causal_path, allow_pickle=True)
    rec_b = data["recovery_b"]
    rec_c = data["recovery_c"]
    rec_a = data.get("tier_a_recovery", None)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    labels_data = [("Tier A", rec_a, "#1f77b4"), ("Tier B", rec_b, "#ff7f0e"), ("Tier C", rec_c, "#2ca02c")]

    for ax, (label, rec, color) in zip(axes, labels_data):
        if rec is None or len(rec) == 0:
            ax.set_title(f"{label} (empty)")
            continue
        mean_r = rec.mean(axis=0)
        std_r = rec.std(axis=0)
        layers = np.arange(len(mean_r))
        ax.plot(layers, mean_r, color=color, linewidth=2)
        ax.fill_between(layers, mean_r - std_r, mean_r + std_r, color=color, alpha=0.2)
        ax.axhline(50, color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"{label} (n={len(rec)})")
        ax.set_xlabel("Layer index")

    axes[0].set_ylabel("Recovery (%)")
    fig.suptitle(f"Tier comparison — {_MODEL_LABELS.get(model_id, model_id)}")
    fig.tight_layout()
    out = root / "figures" / f"phase4_tier_comparison_{model_id}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mechanistic study figures.")
    parser.add_argument("--config", default="configs/mechanistic.yml")
    parser.add_argument("--phase", choices=["0", "1", "2", "3", "4", "all"], default="all")
    parser.add_argument("--models", nargs="+", default=["layerwise", "li", "pli_ngram2"])
    args = parser.parse_args()

    from deep_stylometry.experiments.mechanistic.config import MechanisticConfig
    from deep_stylometry.experiments.mechanistic.io_utils import output_root

    cfg = MechanisticConfig.from_yaml(args.config)
    root = output_root(cfg)

    phases = [args.phase] if args.phase != "all" else ["0", "1", "2", "3", "4"]

    if "0" in phases:
        plot_probe_set_lengths(root)

    if "1" in phases:
        for m in args.models:
            plot_probe_heatmap(root, m)
        plot_cross_model_probe_comparison(root)

    if "2" in phases:
        for m in args.models:
            for tier in ["A", "B", "C"]:
                plot_recovery_curve(root, m, tier=tier)
        for tier in ["A", "B", "C"]:
            plot_all_models_recovery(root, tier=tier)
        for m in args.models:
            for tier in ["A", "B", "C"]:
                plot_rank_recovery_curve(root, m, tier=tier)
            plot_rank_recovery_tier_comparison(root, m)
        for tier in ["A", "B", "C"]:
            plot_all_models_rank_recovery(root, tier=tier)
        for tier in ["A", "B", "C"]:
            plot_score_sensitivity(root, tier=tier)

    if "3" in phases:
        for m in args.models:
            plot_dynamics_panel(root, m)

    if "4" in phases:
        for m in args.models:
            plot_distractor_recovery(root, m)


if __name__ == "__main__":
    main()
