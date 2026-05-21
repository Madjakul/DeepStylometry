# deep_stylometry/experiments/mechanistic/linear_probes.py
"""Phase 1b: train and evaluate linear probes for LISA features."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ridge regression probe
# ---------------------------------------------------------------------------

def fit_ridge_probes(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_eval: np.ndarray,
    Y_eval: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """Fit per-feature ridge probes and return R² on eval set.

    Args:
        X_train: (N_train, H) hidden-state vectors.
        Y_train: (N_train, D) LISA feature targets.
        X_eval:  (N_eval, H)
        Y_eval:  (N_eval, D)

    Returns:
        r2: (D,) R² score per feature dimension.
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    D = Y_train.shape[1]

    clf = Ridge(alpha=alpha, fit_intercept=True)
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_eval)

    r2_scores = np.zeros(D, dtype=np.float32)
    for d in range(D):
        if np.all(Y_train[:, d] == Y_train[0, d]):
            # Constant training target — R² is meaningless
            r2_scores[d] = 0.0
        else:
            r2_scores[d] = float(r2_score(Y_eval[:, d], Y_pred[:, d]))

    return r2_scores


def run_layer_probes(
    train_hidden: np.ndarray,
    eval_hidden: np.ndarray,
    Y_train: np.ndarray,
    Y_eval: np.ndarray,
    feature_names: List[str],
    alpha: float = 1.0,
) -> np.ndarray:
    """Fit probes at every layer and return (n_layers, n_features) R² array.

    Args:
        train_hidden: (N_train, n_layers, H)
        eval_hidden:  (N_eval, n_layers, H)
        Y_train:      (N_train, D)
        Y_eval:       (N_eval, D)
    """
    import warnings
    from numpy.linalg import LinAlgError  # noqa: F401
    warnings.filterwarnings("ignore", message=".*ill-conditioned.*")

    n_layers = train_hidden.shape[1]
    n_features = len(feature_names)
    r2_grid = np.zeros((n_layers, n_features), dtype=np.float32)

    for ell in range(n_layers):
        logger.debug("  Layer %d/%d", ell + 1, n_layers)
        r2 = fit_ridge_probes(
            train_hidden[:, ell, :],
            Y_train,
            eval_hidden[:, ell, :],
            Y_eval,
            alpha=alpha,
        )
        r2_grid[ell] = r2

    return r2_grid


# ---------------------------------------------------------------------------
# Phase 1b entry point
# ---------------------------------------------------------------------------

def run_phase1b(
    cfg: "MechanisticConfig",  # noqa: F821
    model_id: str,
    step: Union[int, str],
    resume: bool = False,
) -> np.ndarray:
    """Run probe training for a single (model_id, step) pair.

    Returns r2_grid of shape (n_layers, n_features).
    """
    import warnings
    from sklearn.exceptions import ConvergenceWarning  # noqa: F401
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

    import pandas as pd
    from deep_stylometry.experiments.mechanistic.io_utils import (
        output_root,
        output_exists,
        phase1_probe_path,
        phase1_lisa_path,
        activation_cache_path,
    )

    root = output_root(cfg)
    probe_path = phase1_probe_path(root, model_id, step, "probes.npz")

    if resume and output_exists(probe_path):
        logger.info("Phase 1b: loading cached probes from %s", probe_path)
        data = np.load(probe_path, allow_pickle=True)
        return data["r2_per_layer_per_feature"]

    # Load LISA features
    feat_path = phase1_lisa_path(root, "features.parquet")
    if not feat_path.exists():
        raise FileNotFoundError(
            f"LISA features not found at {feat_path}. Run Phase 1a first."
        )

    df = pd.read_parquet(feat_path)
    non_feat_cols = {"doc_id", "domain", "split", "text_preview"}
    feat_names = [c for c in df.columns if c not in non_feat_cols]

    train_df = df[df["split"] == "train"]
    eval_df = df[df["split"] == "eval"]

    Y_train = train_df[feat_names].values.astype(np.float32)
    Y_eval = eval_df[feat_names].values.astype(np.float32)
    train_doc_ids = list(train_df["doc_id"])
    eval_doc_ids = list(eval_df["doc_id"])

    # Load activation caches
    train_cache = activation_cache_path(root, model_id, step, "probe_train")
    eval_cache = activation_cache_path(root, model_id, step, "probe_eval")

    if not train_cache.exists() or not eval_cache.exists():
        raise FileNotFoundError(
            f"Activation cache missing for model={model_id} step={step}. "
            "Run activation extraction before probes."
        )

    train_hidden = np.load(train_cache)["hidden_states"]   # (N_tr, L, H)
    eval_hidden = np.load(eval_cache)["hidden_states"]     # (N_ev, L, H)

    # Align rows between hidden states and feature parquet
    # (in case extraction order differs from parquet order)
    train_cached_ids = list(np.load(train_cache, allow_pickle=True)["passage_ids"])
    eval_cached_ids = list(np.load(eval_cache, allow_pickle=True)["passage_ids"])

    def _reindex(hidden, cached_ids, df_ids):
        id_to_row = {pid: i for i, pid in enumerate(cached_ids)}
        order = [id_to_row[pid] for pid in df_ids if pid in id_to_row]
        if len(order) < len(df_ids):
            logger.warning(
                "Alignment: %d/%d doc_ids matched in activation cache.",
                len(order), len(df_ids),
            )
        return hidden[order]

    train_hidden = _reindex(train_hidden, train_cached_ids, train_doc_ids)
    eval_hidden = _reindex(eval_hidden, eval_cached_ids, eval_doc_ids)

    # Trim to matched rows
    n_tr = min(len(train_hidden), len(Y_train))
    n_ev = min(len(eval_hidden), len(Y_eval))
    train_hidden = train_hidden[:n_tr]
    Y_train = Y_train[:n_tr]
    eval_hidden = eval_hidden[:n_ev]
    Y_eval = Y_eval[:n_ev]

    logger.info(
        "Phase 1b: fitting probes for model=%s step=%s  "
        "train=%d eval=%d n_layers=%d n_features=%d",
        model_id, step, n_tr, n_ev, train_hidden.shape[1], len(feat_names),
    )

    r2_grid = run_layer_probes(
        train_hidden, eval_hidden, Y_train, Y_eval, feat_names
    )

    np.savez_compressed(
        probe_path,
        r2_per_layer_per_feature=r2_grid,
        feature_names=np.array(feat_names),
        layer_indices=np.arange(r2_grid.shape[0]),
    )
    logger.info("Saved probes to %s  shape=%s", probe_path, r2_grid.shape)
    return r2_grid
