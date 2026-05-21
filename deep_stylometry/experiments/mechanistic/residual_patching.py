# deep_stylometry/experiments/mechanistic/residual_patching.py
"""Phase 2: causal residual patching for recovery-curve analysis."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from deep_stylometry.experiments.mechanistic.nnsight_helpers import (
    get_all_hidden_states,
    patch_layer_and_forward,
    _hook_patch_all_hidden,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def _mean_score(anchor_embs: torch.Tensor, cand_embs: torch.Tensor,
                anchor_mask: torch.Tensor, cand_mask: torch.Tensor) -> float:
    """Cosine similarity between mean-pooled embeddings."""
    def _pool(embs, mask):
        m = mask.unsqueeze(-1).float()
        return F.normalize((embs * m).sum(1) / m.sum(1).clamp(min=1e-9), p=2, dim=-1)

    q = _pool(anchor_embs, anchor_mask)
    k = _pool(cand_embs, cand_mask)
    return float((q * k).sum())


def _li_score(
    anchor_embs: torch.Tensor, cand_embs: torch.Tensor,
    anchor_mask: torch.Tensor, cand_mask: torch.Tensor,
    anchor_input_ids: Optional[torch.Tensor] = None,
    punc_token_ids: Optional[torch.Tensor] = None,
) -> float:
    """MaxSim late-interaction score."""
    q_norm = F.normalize(anchor_embs, p=2, dim=-1)   # (1, S_q, H)
    k_norm = F.normalize(cand_embs, p=2, dim=-1)     # (1, S_k, H)

    # (1, 1, S_q, S_k)
    sim = torch.einsum("ash,bth->abst", q_norm, k_norm)

    # Mask padding key positions
    min_val = -float(anchor_mask.sum())
    mask_inv = (1.0 - cand_mask.float()).unsqueeze(0).unsqueeze(2)
    sim = sim + mask_inv * min_val

    # MaxSim over key positions per query position (1, 1, S_q)
    max_scores = sim.max(dim=-1).values

    # Skip-list mask for punctuation
    if punc_token_ids is not None and anchor_input_ids is not None:
        punc_mask = torch.isin(anchor_input_ids, punc_token_ids)
        keep = ~punc_mask
        max_scores = max_scores * keep.unsqueeze(1).float()

    # Apply query mask and sum
    max_scores = max_scores * anchor_mask.unsqueeze(1).float()
    return float(max_scores.sum())


def _pli_score(
    anchor_embs: torch.Tensor, cand_embs: torch.Tensor,
    anchor_mask: torch.Tensor, cand_mask: torch.Tensor,
    pli_module,
    anchor_input_ids: Optional[torch.Tensor] = None,
    cand_input_ids: Optional[torch.Tensor] = None,
) -> float:
    """PLI MaxSim score using the model's PatchInteraction module."""
    scores = pli_module(
        query_embs=anchor_embs,
        key_embs=cand_embs,
        q_mask=anchor_mask,
        k_mask=cand_mask,
        q_input_ids=anchor_input_ids,
        k_input_ids=cand_input_ids,
    )
    return float(scores[0, 0])


# ---------------------------------------------------------------------------
# Layerwise-aware embedding helper
# ---------------------------------------------------------------------------

def _forward_to_embeddings(model, encoder_output_or_hidden_states, cfg):
    """Apply head (and layer_attention if layerwise) to get scoring embeddings.

    Parameters
    ----------
    model : DeepStylometry
        The full model (needed for layer_attention and head).
    encoder_output_or_hidden_states : Union[torch.Tensor, Tuple[torch.Tensor, ...]]
        Either the last hidden state (for mean/li/pli) or the full tuple of
        hidden states (for layerwise).
    cfg : BaseConfig
        Model config to check pooling_method.

    Returns
    -------
    torch.Tensor
        Post-head per-token embeddings (B, S, H).
    """
    if cfg.model.pooling_method == "layerwise":
        combined = model.layer_attention(
            list(encoder_output_or_hidden_states), attention_mask=None
        )
        return model.head(combined)
    else:
        return model.head(encoder_output_or_hidden_states)


# ---------------------------------------------------------------------------
# Per-triplet patching
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_triplet_recovery(
    model,
    cfg,
    anchor_text: str,
    pos_text: str,
    neg_text: str,
    tokenizer,
    device: torch.device,
    n_patch_points: int = 23,
) -> Dict[str, Any]:
    """Compute recovery curve for a single triplet.

    Returns dict with keys:
        recovery: np.ndarray (n_patch_points,)
        clean_score: float
        corrupt_score: float
        patched_scores: np.ndarray (n_patch_points,)
        clipped: np.ndarray bool (n_patch_points,)
    """
    def _tok(text):
        return tokenizer(
            text,
            truncation=True,
            max_length=512,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(device)

    a_enc = _tok(anchor_text)
    p_enc = _tok(pos_text)
    n_enc = _tok(neg_text)

    encoder = model.lm.model
    is_layerwise = cfg.model.pooling_method == "layerwise"

    # 1. Encode anchor
    if is_layerwise:
        a_enc_out = encoder(
            input_ids=a_enc["input_ids"],
            attention_mask=a_enc["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
        )
        a_embs = _forward_to_embeddings(model, a_enc_out.hidden_states, cfg)
    else:
        a_enc_out = encoder(
            input_ids=a_enc["input_ids"],
            attention_mask=a_enc["attention_mask"],
            return_dict=True,
        )
        a_embs = _forward_to_embeddings(model, a_enc_out.last_hidden_state, cfg)

    # 2. Get all hidden states for positive and negative
    pos_hidden_states = get_all_hidden_states(
        encoder, p_enc["input_ids"], p_enc["attention_mask"]
    )
    neg_hidden_states = get_all_hidden_states(
        encoder, n_enc["input_ids"], n_enc["attention_mask"]
    )

    # Full forward for clean/corrupt scores
    if is_layerwise:
        pos_embs = _forward_to_embeddings(model, pos_hidden_states, cfg)
        neg_embs = _forward_to_embeddings(model, neg_hidden_states, cfg)
    else:
        pos_embs = _forward_to_embeddings(model, pos_hidden_states[-1], cfg)
        neg_embs = _forward_to_embeddings(model, neg_hidden_states[-1], cfg)

    pooling_method = cfg.model.pooling_method

    # Determine skip_list / pli_module using the model's own trained modules
    punc_token_ids = None
    pli_module = None
    if pooling_method == "li" and cfg.model.skip_list:
        li_mod = model.contrastive_loss.pool
        if hasattr(li_mod, "punc_token_ids"):
            punc_token_ids = li_mod.punc_token_ids.to(device)
    if pooling_method == "pli":
        pli_module = model.contrastive_loss.pool

    def _score(anchor_e, cand_e, a_mask, c_mask, cand_ids=None):
        if pooling_method in ("mean", "layerwise"):
            return _mean_score(anchor_e, cand_e, a_mask, c_mask)
        elif pooling_method == "li":
            return _li_score(
                anchor_e, cand_e, a_mask, c_mask,
                a_enc["input_ids"], punc_token_ids,
            )
        elif pooling_method == "pli":
            return _pli_score(
                anchor_e, cand_e, a_mask, c_mask, pli_module,
                a_enc["input_ids"], cand_ids,
            )
        else:
            raise ValueError(f"Unknown pooling_method: {pooling_method}")

    clean_score = _score(
        a_embs, pos_embs, a_enc["attention_mask"], p_enc["attention_mask"],
        p_enc["input_ids"],
    )
    corrupt_score = _score(
        a_embs, neg_embs, a_enc["attention_mask"], n_enc["attention_mask"],
        n_enc["input_ids"],
    )

    gap = clean_score - corrupt_score

    # 3. For each patch point ℓ, run patched forward
    patched_scores = np.zeros(n_patch_points, dtype=np.float32)

    # Patch tensors must match the negative encoder's output length (S_neg) exactly,
    # or torch.where fails when S_pos != S_neg.  Pad to S_neg; positions beyond
    # the overlap get valid_3d=False so the original negative values are kept.
    p_mask = p_enc["attention_mask"][0]  # (S_p,)
    n_mask = n_enc["attention_mask"][0]  # (S_n,)
    n_neg_len = n_enc["input_ids"].shape[1]

    for ell in range(n_patch_points):
        h_pos = pos_hidden_states[ell]  # (1, S_p, H)
        overlap = min(h_pos.shape[1], n_neg_len)

        # Valid mask: True only where both sequences have real (non-padding) tokens,
        # padded with False out to the full negative sequence length.
        valid_clip = (p_mask[:overlap] > 0) & (n_mask[:overlap] > 0)
        valid_1d = torch.zeros(n_neg_len, dtype=torch.bool, device=device)
        valid_1d[:overlap] = valid_clip
        valid_3d = valid_1d.unsqueeze(0).unsqueeze(-1)  # (1, S_neg, 1)

        # Patch value: positive hidden state, zero-padded to full negative length.
        patch_val = h_pos.new_zeros(1, n_neg_len, h_pos.shape[-1])
        patch_val[:, :overlap, :] = h_pos[:, :overlap, :]

        if is_layerwise:
            # For layerwise models, capture ALL hidden states from the patched forward
            # so they can be passed through layer_attention → head.
            patched_hidden_states = _hook_patch_all_hidden(
                encoder,
                n_enc["input_ids"],
                n_enc["attention_mask"],
                patch_layer_idx=ell,
                patch_value=patch_val,
                valid_mask_3d=valid_3d,
            )
            patched_embs = _forward_to_embeddings(model, patched_hidden_states, cfg)
        else:
            patched_last = patch_layer_and_forward(
                encoder,
                n_enc["input_ids"],
                n_enc["attention_mask"],
                patch_layer_idx=ell,
                patch_value=patch_val,
                valid_mask_3d=valid_3d,
            )
            patched_embs = _forward_to_embeddings(model, patched_last, cfg)

        patched_score = _score(
            a_embs, patched_embs,
            a_enc["attention_mask"], n_enc["attention_mask"],
            n_enc["input_ids"],
        )
        patched_scores[ell] = float(patched_score)

    # Recovery = (patched - corrupt) / (clean - corrupt) * 100
    if abs(gap) < 1e-8:
        recovery = np.zeros(n_patch_points, dtype=np.float32)
    else:
        recovery = (patched_scores - corrupt_score) / gap * 100.0

    return {
        "recovery": recovery,
        "clean_score": float(clean_score),
        "corrupt_score": float(corrupt_score),
        "patched_scores": patched_scores,
    }


# ---------------------------------------------------------------------------
# Phase 2 entry point
# ---------------------------------------------------------------------------

def run_phase2(
    cfg: "MechanisticConfig",  # noqa: F821
    model_id: str,
    step: Union[int, str],
    probe_set: List[Dict],
    resume: bool = False,
    tiers: Optional[List[str]] = None,
) -> np.ndarray:
    """Run causal patching on the probe set for a given (model_id, step).

    Returns recovery array of shape (n_triplets, n_patch_points).
    """
    import json
    from deep_stylometry.experiments.mechanistic.io_utils import (
        output_root,
        output_exists,
        phase2_path,
    )
    from deep_stylometry.experiments.mechanistic.activation_extractor import load_ds_model, load_e5_model

    root = output_root(cfg)
    recovery_file = phase2_path(root, model_id, step, "recovery.npz")

    if resume and output_exists(recovery_file):
        data = np.load(recovery_file, allow_pickle=True)
        cached_tiers = set(data["tier_labels"].tolist())
        needed = set(tiers if tiers is not None else ["A", "B", "C"])
        if needed.issubset(cached_tiers):
            logger.info("Phase 2: loading cached recovery from %s", recovery_file)
            return data["recovery"]
        logger.info(
            "Phase 2: cached recovery has tiers %s but need %s; recomputing.",
            cached_tiers, needed,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_id == "e5":
        raise NotImplementedError("E5 patching requires separate implementation")

    model_entry = cfg.models.get(model_id)
    if model_entry is None:
        raise ValueError(f"Unknown model_id: {model_id}")

    from deep_stylometry.experiments.mechanistic.io_utils import _step_label
    ckpt_path = _resolve_checkpoint(model_entry.checkpoint_pattern, step)
    model, ds_cfg = load_ds_model(model_entry.config, ckpt_path, device)
    model.eval()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

    if tiers is None:
        tiers = ["A", "B", "C"]
    entries = [e for e in probe_set if e["tier"] in tiers]

    n_triplets = len(entries)
    n_patch_points = cfg.patching.n_layers

    recovery_arr = np.zeros((n_triplets, n_patch_points), dtype=np.float32)
    clean_scores = np.zeros(n_triplets, dtype=np.float32)
    corrupt_scores = np.zeros(n_triplets, dtype=np.float32)
    triplet_ids = []
    tier_labels = []

    logger.info(
        "Phase 2: patching %d triplets for model=%s step=%s  device=%s",
        n_triplets, model_id, step, device,
    )

    for i, entry in enumerate(entries):
        if i % 10 == 0:
            logger.info("  Triplet %d/%d", i, n_triplets)

        result = compute_triplet_recovery(
            model=model,
            cfg=ds_cfg,
            anchor_text=entry["anchor_text"],
            pos_text=entry["positive_text"],
            neg_text=entry["negative_text"],
            tokenizer=tokenizer,
            device=device,
            n_patch_points=n_patch_points,
        )
        recovery_arr[i] = result["recovery"]
        clean_scores[i] = result["clean_score"]
        corrupt_scores[i] = result["corrupt_score"]
        triplet_ids.append(entry["triplet_id"])
        tier_labels.append(entry["tier"])

    np.savez_compressed(
        recovery_file,
        recovery=recovery_arr,
        clean_scores=clean_scores,
        corrupt_scores=corrupt_scores,
        triplet_ids=np.array(triplet_ids),
        tier_labels=np.array(tier_labels),
    )
    logger.info("Saved recovery to %s  shape=%s", recovery_file, recovery_arr.shape)
    return recovery_arr


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------

def _resolve_checkpoint(pattern: Optional[str], step: Union[int, str]) -> Optional[str]:
    """Find a checkpoint file matching the step under the pattern directory."""
    if step == 0 or str(step) == "0":
        return None  # pretrained-only baseline

    if pattern is None:
        return None

    from pathlib import Path
    p = Path(pattern)

    if str(step) == "final":
        candidates = ["last.ckpt"]
        for c in candidates:
            candidate = p / c
            if candidate.exists():
                return str(candidate)
        # glob for last
        ckpts = sorted(p.glob("*.ckpt"))
        return str(ckpts[-1]) if ckpts else None

    # Step-specific: look for checkpoints with step number in name
    step_int = int(step)
    ckpts = sorted(p.glob("*.ckpt"))
    # Find closest step
    best = None
    best_diff = float("inf")
    for c in ckpts:
        # Try to parse step number from filename
        import re
        m = re.search(r"step[_=]?(\d+)", c.name)
        if m:
            s = int(m.group(1))
            diff = abs(s - step_int)
            if diff < best_diff:
                best_diff = diff
                best = str(c)
    if best is not None and best_diff < 500:
        return best

    logger.warning(
        "Could not find checkpoint for step=%s under %s; using last.ckpt",
        step, pattern,
    )
    ckpts = sorted(p.glob("*.ckpt"))
    return str(ckpts[-1]) if ckpts else None
