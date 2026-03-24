# deep_stylometry/modules/cross_attention_compressor.py

import math
import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int


class CrossAttentionCompressor(nn.Module):
    """Compresses variable-length patches into fixed-size vectors via
    cross-attention.

    One learnable query vector per patch attends to the patch's constituent
    tokens.  The implementation builds a padded ``(B, P, T_max, H)`` view
    from the flat ``(B, S, H)`` token embeddings, runs batched multi-head
    cross-attention, and reshapes back to ``(B, P, H)``.

    Parameters
    ----------
    hidden_size:
        Dimensionality of the input token embeddings and output patch vectors.
    d_k:
        Key/Value dimensionality (per head) inside the cross-attention.
    n_heads:
        Number of attention heads.
    """

    def __init__(self, hidden_size: int, d_k: int, n_heads: int) -> None:
        super().__init__()
        self.d_k = d_k
        self.n_heads = n_heads
        self.d_model = d_k * n_heads

        # Learned patch query (shared across all patches)
        self.query = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        self.W_K = nn.Linear(hidden_size, self.d_model, bias=False)
        self.W_V = nn.Linear(hidden_size, self.d_model, bias=False)
        self.W_O = nn.Linear(self.d_model, hidden_size, bias=False)

        logging.info(
            f"CrossAttentionCompressor: hidden={hidden_size}, "
            f"d_k={d_k}, n_heads={n_heads}"
        )

    def forward(
        self,
        token_embs: Float[torch.Tensor, "batch seq hidden"],
        patch_ids: Int[torch.Tensor, "batch seq"],
        mask: Int[torch.Tensor, "batch seq"],
        max_patches: int,
    ) -> Tuple[
        Float[torch.Tensor, "batch patches hidden"],
        Int[torch.Tensor, "batch patches"],
    ]:
        """Compress token embeddings into patch embeddings.

        Parameters
        ----------
        token_embs:
            ``(B, S, H)`` contextualised token representations.
        patch_ids:
            ``(B, S)`` integer patch assignments; -1 for padding.
        mask:
            ``(B, S)`` attention mask.
        max_patches:
            ``P`` — the number of patch slots to produce.

        Returns
        -------
        patch_embs:
            ``(B, P, H)`` compressed patch representations.
        patch_mask:
            ``(B, P)`` binary mask; 1 for real patches, 0 for empty slots.
        """
        B, S, H = token_embs.shape
        P = max_patches
        device = token_embs.device
        dtype = token_embs.dtype

        valid = (mask > 0) & (patch_ids >= 0)  # (B, S)

        # Compute patch sizes: (B, P)
        ids_clamped = patch_ids.clamp(min=0)
        one_hot = F.one_hot(ids_clamped, num_classes=P).float()  # (B, S, P)
        one_hot = one_hot * valid.unsqueeze(-1).float()
        patch_sizes = one_hot.sum(dim=1)  # (B, P)
        max_tok_per_patch = max(1, int(patch_sizes.max().item()))

        # Build padded token tensor: (B, P, T_max, H)
        patch_tok_embs = torch.zeros(B, P, max_tok_per_patch, H,
                                     device=device, dtype=dtype)
        patch_tok_mask = torch.zeros(B, P, max_tok_per_patch,
                                     device=device, dtype=dtype)

        for b in range(B):
            for p in range(P):
                tok_idx = ((patch_ids[b] == p) & valid[b]).nonzero(as_tuple=True)[0]
                n = len(tok_idx)
                if n > 0:
                    n = min(n, max_tok_per_patch)
                    patch_tok_embs[b, p, :n] = token_embs[b, tok_idx[:n]]
                    patch_tok_mask[b, p, :n] = 1.0

        # Flatten to (B*P, T_max, H) for batched attention
        x = patch_tok_embs.view(B * P, max_tok_per_patch, H)
        attn_mask = patch_tok_mask.view(B * P, max_tok_per_patch)  # (B*P, T)

        # Compute K, V
        K = self.W_K(x)  # (B*P, T, d_model)
        V = self.W_V(x)  # (B*P, T, d_model)

        # Expand query
        Q = self.query.expand(B * P, -1, -1)  # (B*P, 1, d_model)

        # Attention scores: (B*P, 1, T)
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.d_k)

        # Ensure at least one valid position per patch (avoid NaN from empty)
        all_masked = (attn_mask.sum(dim=-1) == 0)  # (B*P,)
        safe_mask = attn_mask.clone()
        safe_mask[all_masked, 0] = 1.0  # Dummy position for empty patches

        scores = scores + (1.0 - safe_mask.unsqueeze(1)) * (-10000.0)
        attn_weights = F.softmax(scores, dim=-1)  # (B*P, 1, T)

        # Weighted sum of values: (B*P, 1, d_model) → (B*P, d_model)
        out = torch.bmm(attn_weights, V).squeeze(1)
        out = self.W_O(out)  # (B*P, H)

        patch_embs = out.view(B, P, H)

        # Zero out empty patches
        patch_mask = (patch_sizes > 0).long()  # (B, P)
        patch_embs = patch_embs * patch_mask.unsqueeze(-1).float()

        return patch_embs, patch_mask
