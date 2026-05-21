# deep_stylometry/experiments/mechanistic/geometry.py
"""Optional appendix: author/topic geometry projections."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_pca_projections(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    n_components: int = 2,
) -> np.ndarray:
    """PCA of (N, H) hidden states. Returns (N, n_components)."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    return pca.fit_transform(hidden_states)


def compute_umap_projections(
    hidden_states: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """UMAP projection of (N, H) -> (N, 2). Skips gracefully if umap not installed."""
    try:
        import umap
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=seed,
        )
        return reducer.fit_transform(hidden_states)
    except ImportError:
        logger.warning("umap-learn not installed; falling back to PCA.")
        return compute_pca_projections(hidden_states, np.zeros(len(hidden_states)), n_components)


def compute_author_topic_separation(
    hidden_states: np.ndarray,
    author_labels: np.ndarray,
    topic_labels: np.ndarray,
) -> Dict[str, float]:
    """Compute silhouette scores for author vs topic separation."""
    try:
        from sklearn.metrics import silhouette_score
        author_sil = float(silhouette_score(hidden_states, author_labels))
        topic_sil = float(silhouette_score(hidden_states, topic_labels))
        return {"author_silhouette": author_sil, "topic_silhouette": topic_sil}
    except Exception as e:
        logger.warning("Silhouette score failed: %s", e)
        return {}


def run_geometry_analysis(
    cfg: "MechanisticConfig",  # noqa: F821
    model_id: str,
    layer_idx: int = -1,
) -> None:
    """Run geometry analysis for a model at a specific layer."""
    import json
    import pandas as pd
    from pathlib import Path
    from deep_stylometry.experiments.mechanistic.io_utils import (
        output_root,
        activation_cache_path,
        phase0_path,
        figures_path,
    )

    root = output_root(cfg)

    # Load probe-set activations
    cache_path = activation_cache_path(root, model_id, "final", "probe_train")
    if not cache_path.exists():
        logger.warning("Activation cache not found for geometry analysis.")
        return

    data = np.load(cache_path, allow_pickle=True)
    hidden = data["hidden_states"]  # (N, L, H)
    passage_ids = list(data["passage_ids"])

    # Use last layer by default, or specified layer
    if layer_idx < 0:
        layer_idx = hidden.shape[1] + layer_idx
    h = hidden[:, layer_idx, :]  # (N, H)

    # Load LISA features for author/domain labels
    feat_path = root / "phase1_lisa" / "features.parquet"
    if not feat_path.exists():
        logger.warning("LISA features not found; skipping geometry analysis.")
        return

    df = pd.read_parquet(feat_path)
    domain_labels = np.array([
        df[df["doc_id"] == pid]["domain"].values[0]
        if pid in df["doc_id"].values else "unknown"
        for pid in passage_ids
    ])

    projections = compute_pca_projections(h, domain_labels)

    out_path = figures_path(root, f"geometry_{model_id}_layer{layer_idx}.npz")
    np.savez_compressed(
        out_path,
        projections=projections,
        domain_labels=domain_labels,
        layer_idx=layer_idx,
        model_id=model_id,
    )
    logger.info("Saved geometry projections to %s", out_path)
