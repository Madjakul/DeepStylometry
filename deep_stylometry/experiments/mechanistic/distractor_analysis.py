# deep_stylometry/experiments/mechanistic/distractor_analysis.py
"""Phase 4: predictable failure analysis on Tier B and C."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_DISTRACTOR_POOL_SIZE = 1000


# ---------------------------------------------------------------------------
# Distractor pool scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def _score_against_pool(
    anchor_embs: torch.Tensor,
    anchor_mask: torch.Tensor,
    pool_embs: torch.Tensor,
    pool_mask: torch.Tensor,
    pooling_method: str,
    cfg,
    anchor_input_ids: Optional[torch.Tensor] = None,
    punc_token_ids: Optional[torch.Tensor] = None,
    pli_module=None,
) -> np.ndarray:
    """Score anchor against a pool of candidates. Returns (pool_size,) scores."""
    if pooling_method in ("mean", "layerwise"):
        def _pool(e, m):
            mf = m.unsqueeze(-1).float()
            return F.normalize((e * mf).sum(1) / mf.sum(1).clamp(min=1e-9), p=2, dim=-1)
        q = _pool(anchor_embs, anchor_mask)         # (1, H)
        k = _pool(pool_embs, pool_mask)             # (N, H)
        scores = (q * k).sum(dim=-1)               # (N,)

    elif pooling_method == "li":
        q_norm = F.normalize(anchor_embs, p=2, dim=-1)     # (1, S_q, H)
        k_norm = F.normalize(pool_embs, p=2, dim=-1)       # (N, S_k, H)
        sim = torch.einsum("ash,bth->abst", q_norm, k_norm)  # (1, N, S_q, S_k)

        min_val = -float(anchor_mask.sum())
        mask_inv = (1.0 - pool_mask.float()).unsqueeze(0).unsqueeze(2)
        sim = sim + mask_inv * min_val

        max_scores = sim.max(dim=-1).values  # (1, N, S_q)

        if punc_token_ids is not None and anchor_input_ids is not None:
            punc_mask = torch.isin(anchor_input_ids, punc_token_ids)
            keep = (~punc_mask).float()  # convert to float before padding
            S_q = max_scores.shape[-1]
            if keep.shape[-1] < S_q:
                keep = F.pad(keep, (0, S_q - keep.shape[-1]), value=0.0)
            max_scores = max_scores * keep.unsqueeze(1)

        max_scores = max_scores * anchor_mask.unsqueeze(1).float()
        scores = max_scores.sum(dim=-1)[0]  # (N,)

    elif pooling_method == "pli":
        # Pool entries lack input_ids so keys fall back to ngram patching inside pli_module
        scores = pli_module(
            query_embs=anchor_embs,
            key_embs=pool_embs,
            q_mask=anchor_mask,
            k_mask=pool_mask,
            q_input_ids=anchor_input_ids,
        )[0]  # (N,)

    else:
        raise ValueError(f"Unknown pooling_method: {pooling_method}")

    return scores.cpu().float().numpy()


# ---------------------------------------------------------------------------
# Phase 4 main
# ---------------------------------------------------------------------------

def run_phase4(
    cfg: "MechanisticConfig",  # noqa: F821
    model_id: str,
    probe_set: List[Dict],
    resume: bool = False,
) -> None:
    import json
    import pandas as pd
    from pathlib import Path
    from deep_stylometry.experiments.mechanistic.io_utils import (
        output_root,
        output_exists,
        phase4_path,
    )
    from deep_stylometry.experiments.mechanistic.residual_patching import (
        compute_triplet_recovery,
        _resolve_checkpoint,
        _mean_score,
        _li_score,
        _pli_score,
    )
    from deep_stylometry.experiments.mechanistic.activation_extractor import load_ds_model

    root = output_root(cfg)
    rankings_path = phase4_path(root, model_id, "rankings.parquet")
    causal_path = phase4_path(root, model_id, "causal.npz")

    if resume and output_exists(rankings_path, causal_path):
        logger.info("Phase 4: loading cached results for model=%s", model_id)
        return

    if model_id == "e5":
        logger.warning("Phase 4 E5 not yet implemented.")
        return

    model_entry = cfg.models.get(model_id)
    if model_entry is None:
        raise ValueError(f"Unknown model_id: {model_id}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = _resolve_checkpoint(model_entry.checkpoint_pattern, "final")
    model, ds_cfg = load_ds_model(model_entry.config, ckpt_path, device)
    model.eval()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

    # Extract interaction module once — avoids redundant vocab scans per entry
    pooling_method = ds_cfg.model.pooling_method
    punc_ids = None
    pli_module = None
    if pooling_method == "li":
        li_mod = model.contrastive_loss.pool
        raw_punc_ids = getattr(li_mod, "punc_token_ids", None)
        if raw_punc_ids is not None:
            punc_ids = raw_punc_ids.to(device)
    elif pooling_method == "pli":
        pli_module = model.contrastive_loss.pool

    # Load validation distractor pool
    logger.info("Sampling distractor pool (%d passages) from base-4 valid...", _DISTRACTOR_POOL_SIZE)
    import datasets as hf_datasets
    import random
    rng = random.Random(cfg.io.seed)
    valid_ds = hf_datasets.load_dataset(
        "almanach/halvest-contrastive",
        name=cfg.base_data_subset,
        split="valid",
    )

    probe_doc_ids = {e.get("anchor_doc_id", "") for e in probe_set}
    distractor_texts = []
    distractor_ids = []
    for i, row in enumerate(valid_ds):
        if len(distractor_texts) >= _DISTRACTOR_POOL_SIZE:
            break
        doc_id = row.get("query_id", str(i))
        if doc_id in probe_doc_ids:
            continue
        text = row.get("query", "")
        if text:
            distractor_texts.append(text)
            distractor_ids.append(doc_id)

    # Encode distractor pool
    def _enc_batch(texts):
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
            add_special_tokens=True,
        ).to(device)
        with torch.no_grad():
            embs = model(enc["input_ids"], enc["attention_mask"])
        return embs, enc["attention_mask"]

    logger.info("Encoding distractor pool...")
    pool_embs_list = []
    pool_mask_list = []
    batch_size = 32
    for start in range(0, len(distractor_texts), batch_size):
        batch = distractor_texts[start: start + batch_size]
        e, m = _enc_batch(batch)
        pool_embs_list.append(e.cpu())
        pool_mask_list.append(m.cpu())

    # Pad to common length
    max_pool_len = max(e.shape[1] for e in pool_embs_list)
    pool_embs_padded = [
        F.pad(e, (0, 0, 0, max_pool_len - e.shape[1])) for e in pool_embs_list
    ]
    pool_mask_padded = [
        F.pad(m, (0, max_pool_len - m.shape[1])) for m in pool_mask_list
    ]
    all_pool_embs = torch.cat(pool_embs_padded, dim=0).to(device)    # (N_pool, S, H)
    all_pool_mask = torch.cat(pool_mask_padded, dim=0).to(device)    # (N_pool, S)

    # Process Tier B and C
    rankings_rows = []
    all_recovery_b = []
    all_recovery_c = []
    tier_a_entries = [e for e in probe_set if e["tier"] == "A"]
    tier_b_entries = [e for e in probe_set if e["tier"] == "B"]
    tier_c_entries = [e for e in probe_set if e["tier"] == "C"]

    # Load Tier A recovery for comparison
    from deep_stylometry.experiments.mechanistic.io_utils import phase2_path
    tier_a_rec_file = phase2_path(root, model_id, "final", "recovery.npz")
    tier_a_recovery = None
    if tier_a_rec_file.exists():
        data = np.load(tier_a_rec_file, allow_pickle=True)
        tl = list(data["tier_labels"])
        a_idx = [i for i, t in enumerate(tl) if t == "A"]
        tier_a_recovery = data["recovery"][a_idx]

    for tier_entries, tier_label in [(tier_b_entries, "B"), (tier_c_entries, "C")]:
        if not tier_entries:
            continue

        logger.info("Processing Tier %s (%d entries)...", tier_label, len(tier_entries))
        recovery_list = []

        for entry in tier_entries:
            # Encode anchor and positive
            def _tok1(text):
                return tokenizer(
                    text, truncation=True, max_length=512,
                    add_special_tokens=True, return_tensors="pt",
                ).to(device)

            a_enc = _tok1(entry["anchor_text"])
            p_enc = _tok1(entry["positive_text"])
            n_enc = _tok1(entry["negative_text"])

            with torch.no_grad():
                a_embs = model(a_enc["input_ids"], a_enc["attention_mask"])
                p_embs = model(p_enc["input_ids"], p_enc["attention_mask"])
                n_embs = model(n_enc["input_ids"], n_enc["attention_mask"])

            # Pad to pool length
            S_a = a_embs.shape[1]
            pool_s = all_pool_embs.shape[1]

            a_embs_pad = F.pad(a_embs, (0, 0, 0, max(0, pool_s - S_a)))
            a_mask_pad = F.pad(a_enc["attention_mask"], (0, max(0, pool_s - S_a)))
            a_ids_pad = F.pad(a_enc["input_ids"], (0, max(0, pool_s - S_a)), value=0)

            pool_scores = _score_against_pool(
                a_embs_pad, a_mask_pad,
                all_pool_embs, all_pool_mask,
                pooling_method, ds_cfg,
                anchor_input_ids=a_ids_pad,
                punc_token_ids=punc_ids,
                pli_module=pli_module,
            )

            # Score pos and neg (consistent with pool scoring — apply same skip-list)
            if pooling_method in ("mean", "layerwise"):
                pos_score = _mean_score(a_embs, p_embs, a_enc["attention_mask"], p_enc["attention_mask"])
                neg_score = _mean_score(a_embs, n_embs, a_enc["attention_mask"], n_enc["attention_mask"])
            elif pooling_method == "li":
                pos_score = _li_score(
                    a_embs, p_embs, a_enc["attention_mask"], p_enc["attention_mask"],
                    anchor_input_ids=a_enc["input_ids"], punc_token_ids=punc_ids,
                )
                neg_score = _li_score(
                    a_embs, n_embs, a_enc["attention_mask"], n_enc["attention_mask"],
                    anchor_input_ids=a_enc["input_ids"], punc_token_ids=punc_ids,
                )
            else:  # pli
                pos_score = _pli_score(
                    a_embs, p_embs, a_enc["attention_mask"], p_enc["attention_mask"],
                    pli_module, a_enc["input_ids"], p_enc["input_ids"],
                )
                neg_score = _pli_score(
                    a_embs, n_embs, a_enc["attention_mask"], n_enc["attention_mask"],
                    pli_module, a_enc["input_ids"], n_enc["input_ids"],
                )

            # Rank of negative among pool+negative
            all_scores = np.append(pool_scores, float(neg_score))
            neg_rank = int(np.sum(all_scores > float(neg_score)))

            rankings_rows.append({
                "tier": tier_label,
                "triplet_id": entry["triplet_id"],
                "pos_score": float(pos_score),
                "neg_score": float(neg_score),
                "neg_higher_than_pos": float(neg_score) > float(pos_score),
                "neg_rank": neg_rank,
                "pool_size": len(all_scores),
            })

            # Causal patching
            result = compute_triplet_recovery(
                model=model, cfg=ds_cfg,
                anchor_text=entry["anchor_text"],
                pos_text=entry["positive_text"],
                neg_text=entry["negative_text"],
                tokenizer=tokenizer, device=device,
                n_patch_points=cfg.patching.n_layers,
            )
            recovery_list.append(result["recovery"])

        if tier_label == "B":
            all_recovery_b = recovery_list
        else:
            all_recovery_c = recovery_list

    rankings_df = pd.DataFrame(rankings_rows)
    rankings_df.to_parquet(rankings_path, index=False)
    logger.info("Saved rankings to %s", rankings_path)

    rec_b = np.stack(all_recovery_b, axis=0) if all_recovery_b else np.zeros((0, cfg.patching.n_layers))
    rec_c = np.stack(all_recovery_c, axis=0) if all_recovery_c else np.zeros((0, cfg.patching.n_layers))

    np.savez_compressed(
        causal_path,
        recovery_b=rec_b,
        recovery_c=rec_c,
        tier_a_recovery=tier_a_recovery if tier_a_recovery is not None else np.zeros((0, cfg.patching.n_layers)),
    )
    logger.info("Saved causal arrays to %s", causal_path)
