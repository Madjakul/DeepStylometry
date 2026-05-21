# deep_stylometry/experiments/mechanistic/activation_extractor.py
"""Phase 1+2 shared: extract per-layer hidden states from models."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# E5 encoder (no fine-tuning, used as topical control)
# ---------------------------------------------------------------------------

def load_e5_model(device: torch.device):
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")
    model = AutoModel.from_pretrained("intfloat/multilingual-e5-base")
    model.eval().to(device)
    return model, tokenizer


def _e5_forward_with_hidden_states(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """Forward pass through E5 returning (last_hidden, all_hidden_states)."""
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
    return out.last_hidden_state, out.hidden_states


# ---------------------------------------------------------------------------
# DeepStylometry encoder helpers
# ---------------------------------------------------------------------------

def load_ds_model(config_path: str, checkpoint_path: Optional[str], device: torch.device):
    """Load a DeepStylometry model for inference.

    If checkpoint_path is None, returns pretrained encoder + random head (step 0).
    """
    from deep_stylometry.modules.modeling_deep_stylometry import DeepStylometry
    from deep_stylometry.utils.configs import BaseConfig

    cfg = BaseConfig(mode="test").from_yaml(config_path)

    if checkpoint_path is not None:
        # Reconcile config with what was actually saved: if the checkpoint has no
        # mean_centerer buffers but the config requests one, disable it to avoid
        # a strict load_state_dict mismatch (checkpoint was trained without it).
        if getattr(cfg.model, "mean_center", False):
            ckpt_keys = set(torch.load(checkpoint_path, map_location="cpu", weights_only=True)["state_dict"].keys())
            if "mean_centerer.mu" not in ckpt_keys:
                logger.warning(
                    "Checkpoint %s has no mean_centerer; overriding mean_center=False for loading.",
                    checkpoint_path,
                )
                cfg.model.mean_center = False
        model = DeepStylometry.load_from_checkpoint(checkpoint_path, cfg=cfg)
    else:
        model = DeepStylometry(cfg)

    model.eval().to(device)
    return model, cfg


def _ds_forward_with_hidden_states(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """Forward pass returning (post_head_embs, all_encoder_hidden_states).

    post_head_embs: (B, S, H) after the MLP head
    all_encoder_hidden_states: tuple of length n_encoder_layers+1
    """
    with torch.no_grad():
        enc_out = model.lm.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = enc_out.last_hidden_state
        hidden_states = enc_out.hidden_states  # (n_layers+1,) each (B,S,H)

        # Apply head to last hidden state
        post_head = model.head(last_hidden)

    return post_head, hidden_states


# ---------------------------------------------------------------------------
# Attention-mask-aware mean pooling
# ---------------------------------------------------------------------------

def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean pool (B, S, H) -> (B, H) using attention mask."""
    mask_f = mask.unsqueeze(-1).float()
    summed = (hidden * mask_f).sum(dim=1)
    counts = mask_f.sum(dim=1).clamp(min=1e-9)
    return summed / counts


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------

def extract_hidden_states_ds(
    model,
    texts: List[str],
    tokenizer,
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract mean-pooled hidden states for DeepStylometry models.

    Returns:
        hidden_states: (N, n_patch_points, H)  — mean-pooled per layer
        post_head: (N, H)                       — mean-pooled post-head embeddings
    """
    all_hidden: List[np.ndarray] = []
    all_post_head: List[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start: start + batch_size]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
            add_special_tokens=True,
        ).to(device)

        post_head, hidden_states = _ds_forward_with_hidden_states(
            model, enc["input_ids"], enc["attention_mask"]
        )
        mask = enc["attention_mask"]

        # Mean pool post-head
        ph_pooled = _mean_pool(post_head, mask)  # (B, H)
        all_post_head.append(ph_pooled.cpu().float().numpy())

        # Mean pool each hidden state layer
        layer_pooled = torch.stack(
            [_mean_pool(h, mask) for h in hidden_states], dim=1
        )  # (B, n_layers+1, H)
        all_hidden.append(layer_pooled.cpu().float().numpy())

    return (
        np.concatenate(all_hidden, axis=0),   # (N, L, H)
        np.concatenate(all_post_head, axis=0), # (N, H)
    )


def extract_hidden_states_e5(
    model,
    texts: List[str],
    tokenizer,
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract mean-pooled hidden states for E5 (no instruction prefix).

    Returns:
        hidden_states: (N, n_patch_points, H)
        pooled: (N, H)  — mean-pooled last hidden state
    """
    all_hidden: List[np.ndarray] = []
    all_pooled: List[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start: start + batch_size]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
            add_special_tokens=True,
        ).to(device)

        last_hidden, hidden_states = _e5_forward_with_hidden_states(
            model, enc["input_ids"], enc["attention_mask"]
        )
        mask = enc["attention_mask"]

        pooled = _mean_pool(last_hidden, mask)
        all_pooled.append(pooled.cpu().float().numpy())

        layer_pooled = torch.stack(
            [_mean_pool(h, mask) for h in hidden_states], dim=1
        )
        all_hidden.append(layer_pooled.cpu().float().numpy())

    return (
        np.concatenate(all_hidden, axis=0),
        np.concatenate(all_pooled, axis=0),
    )


# ---------------------------------------------------------------------------
# Cache-backed extraction
# ---------------------------------------------------------------------------

def extract_and_cache(
    model_id: str,
    step: Union[int, str],
    texts: List[str],
    passage_ids: List[str],
    model,
    tokenizer,
    device: torch.device,
    output_root_dir,
    split_name: str = "probe_train",
    is_e5: bool = False,
    resume: bool = True,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract hidden states, cache to disk, and return (N, L, H) array."""
    from deep_stylometry.experiments.mechanistic.io_utils import activation_cache_path

    cache_file = activation_cache_path(output_root_dir, model_id, step, split_name)

    if resume and cache_file.exists():
        logger.info("Loading cached activations from %s", cache_file)
        data = np.load(cache_file)
        return data["hidden_states"]

    logger.info(
        "Extracting hidden states for model=%s step=%s split=%s (%d passages)...",
        model_id, step, split_name, len(texts),
    )

    if is_e5:
        hidden, _ = extract_hidden_states_e5(model, texts, tokenizer, device, batch_size)
    else:
        hidden, _ = extract_hidden_states_ds(model, texts, tokenizer, device, batch_size)

    np.savez_compressed(
        cache_file,
        hidden_states=hidden.astype(np.float32),
        passage_ids=np.array(passage_ids),
    )
    logger.info("Saved activations to %s  shape=%s", cache_file, hidden.shape)
    return hidden
