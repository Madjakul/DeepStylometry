# deep_stylometry/experiments/mechanistic/training_dynamics.py
"""Phase 3: orchestrate per-checkpoint reruns for training dynamics."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def _compute_inflection_layer(recovery: np.ndarray) -> int:
    """Return the first layer index where mean recovery crosses 50%."""
    mean_rec = recovery.mean(axis=0)  # (n_layers,)
    for ell in range(len(mean_rec)):
        if mean_rec[ell] >= 50.0:
            return ell
    return int(np.argmax(mean_rec))


def _emergence_layer(r2_grid: np.ndarray, threshold: float = 0.5) -> int:
    """Return the first layer where any LISA feature exceeds R² threshold."""
    for ell in range(r2_grid.shape[0]):
        if r2_grid[ell].max() >= threshold:
            return ell
    return -1


def run_phase3(
    cfg: "MechanisticConfig",  # noqa: F821
    model_ids: Optional[List[str]] = None,
    steps: Optional[List[Union[int, str]]] = None,
    resume: bool = False,
) -> None:
    """Run training-dynamics analysis across checkpoints.

    Calls Phase 1b and Phase 2 for each (model_id, step), then aggregates
    results into a summary parquet.
    """
    import pandas as pd
    from deep_stylometry.experiments.mechanistic.io_utils import (
        output_root,
        output_exists,
        phase0_path,
        phase1_probe_path,
        phase2_path,
        phase3_path,
        activation_cache_path,
    )
    from deep_stylometry.experiments.mechanistic.activation_extractor import (
        load_ds_model,
        extract_and_cache,
    )
    from deep_stylometry.experiments.mechanistic.linear_probes import run_phase1b
    from deep_stylometry.experiments.mechanistic.residual_patching import (
        run_phase2,
        _resolve_checkpoint,
    )
    import json
    import torch

    root = output_root(cfg)

    # Load probe set
    probe_set_file = phase0_path(root, "probe_set.json")
    if not probe_set_file.exists():
        raise FileNotFoundError("Probe set not found. Run Phase 0 first.")
    with open(probe_set_file) as f:
        probe_set = json.load(f)

    if model_ids is None:
        model_ids = list(cfg.models.keys())
    if steps is None:
        steps = cfg.checkpoints.selected_steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load probe-train and probe-eval texts once (shared across models)
    feat_path = root / "phase1_lisa" / "features.parquet"
    if feat_path.exists():
        df_feats = pd.read_parquet(feat_path)
        non_feat = {"doc_id", "domain", "split", "text_preview"}
        train_rows = df_feats[df_feats["split"] == "train"]
        eval_rows = df_feats[df_feats["split"] == "eval"]
        train_texts = list(train_rows.get("text_preview", train_rows.iloc[:, 3]))
        eval_texts = list(eval_rows.get("text_preview", eval_rows.iloc[:, 3]))
        train_doc_ids = list(train_rows["doc_id"])
        eval_doc_ids = list(eval_rows["doc_id"])
    else:
        logger.warning("LISA feature parquet not found; skipping probe training in phase 3.")
        train_texts = eval_texts = train_doc_ids = eval_doc_ids = []

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

    summary_rows: List[Dict] = []

    for model_id in model_ids:
        if model_id == "e5":
            logger.info("Skipping E5 for phase 3 dynamics (not supported yet).")
            continue
        model_entry = cfg.models.get(model_id)
        if model_entry is None:
            logger.warning("Unknown model_id %s; skipping.", model_id)
            continue

        for step in steps:
            logger.info("Phase 3: model=%s step=%s", model_id, step)

            ckpt_path = _resolve_checkpoint(model_entry.checkpoint_pattern, step)
            model, ds_cfg = load_ds_model(model_entry.config, ckpt_path, device)
            model.eval()

            # Extract activations if not cached
            if train_texts:
                extract_and_cache(
                    model_id=model_id,
                    step=step,
                    texts=train_texts,
                    passage_ids=train_doc_ids,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    output_root_dir=root,
                    split_name="probe_train",
                    is_e5=False,
                    resume=resume,
                )
                extract_and_cache(
                    model_id=model_id,
                    step=step,
                    texts=eval_texts,
                    passage_ids=eval_doc_ids,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    output_root_dir=root,
                    split_name="probe_eval",
                    is_e5=False,
                    resume=resume,
                )

                r2_grid = run_phase1b(cfg, model_id, step, resume=resume)
            else:
                r2_grid = None

            recovery = run_phase2(cfg, model_id, step, probe_set, resume=resume, tiers=["A"])

            # Inflection layer from final checkpoint analysis
            inflection = int(_compute_inflection_layer(recovery))
            emergence = int(_emergence_layer(r2_grid)) if r2_grid is not None else -1

            # Tier A accuracy
            tier_a_entries = [e for e in probe_set if e["tier"] == "A"]
            n_tier_a = len(tier_a_entries)
            tier_a_recovery = recovery[:n_tier_a] if n_tier_a > 0 else np.zeros((0, 1))

            row: Dict[str, Any] = {
                "model_id": model_id,
                "step": str(step),
                "inflection_layer": inflection,
                "emergence_layer": emergence,
                "mean_recovery_at_inflection": float(
                    recovery[:, inflection].mean() if inflection < recovery.shape[1] else 0.0
                ),
                "n_tier_a": n_tier_a,
            }

            if r2_grid is not None:
                r2_max_per_feature = r2_grid.max(axis=0)
                top3_idx = np.argsort(r2_max_per_feature)[-3:][::-1]
                row["top3_feature_r2"] = float(r2_max_per_feature[top3_idx].mean())

            summary_rows.append(row)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        out_path = phase3_path(root, "summary.parquet")
        summary_df.to_parquet(out_path, index=False)
        logger.info("Saved phase 3 summary to %s", out_path)
