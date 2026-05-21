# deep_stylometry/utils/configs/model_config.py

from dataclasses import dataclass
from typing import Literal

from deep_stylometry.utils.helpers import DictAccessMixin


@dataclass
class ModelConfig(DictAccessMixin):
    base_checkpoint: str = "answerdotai/ModernBERT-base"
    dropout: float = 0.1
    expansion_ratio: int = 4
    pooling_method: Literal["mean", "li", "pli", "layerwise"] = "li"
    skip_list: bool = False
    lm_hidden_size: int = 768  # Hidden size of backbone; set to match base_checkpoint
    # --- Layerwise attention (IDIOLEX-style) ---
    mean_center: bool = False  # Apply running-mean centering + L2 norm during training
    # --- Patch-level late interaction ---
    patch_method: Literal["none", "whitespace", "wholeword", "ngram", "learned"] = "none"
    patch_size: int = 3  # n-gram size (used when patch_method="ngram")
    patch_compression: Literal["mean", "max", "cross_attention"] = "mean"
    patch_cross_attn_dim: int = 128  # d_k for cross-attention compression
    patch_cross_attn_heads: int = 2  # Number of attention heads
    patch_lambda: float = 0.1  # Coefficient for L_patch regularizer
    patch_target_rate: float = 1.0 / 3.0  # Target mean cut probability (~patch length 3)
    # --- Gumbel-Softmax annealing (for learned patching) ---
    gumbel_tau_init: float = 1.0
    gumbel_tau_final: float = 0.1
    gumbel_anneal_steps: int = 10000
    # --- Attention implementation ---
    # "sdpa" works on all CUDA GPUs; "flash_attention_2" requires Ampere (sm80+).
    attn_implementation: str = "sdpa"
