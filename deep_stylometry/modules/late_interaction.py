# deep_stylometry/modules/late_interaction.py

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LateInteraction(nn.Module):
    def __init__(
        self,
        do_distance: bool,
        exp_decay: bool,
        seq_len: int,
        alpha: float = 0.5,
        use_max: bool = False,
    ):
        super().__init__()
        self.do_distance = do_distance
        self.use_max = use_max
        if self.do_distance:
            self.alpha_raw = nn.Parameter(torch.tensor(alpha))
            positions = torch.arange(seq_len)
            distance = (positions.unsqueeze(1) - positions.unsqueeze(0)).abs().float()
            self.register_buffer("distance", distance)
        self.exp_decay = exp_decay
        self.logit_scale = nn.Parameter(
            torch.log(torch.tensor(1.0))
        )  # Learnable logit scale

    @property
    def alpha(self):
        """Softplus ensures alpha > 0."""
        return F.softplus(self.alpha_raw)

    def forward(
        self,
        query_embs: torch.Tensor,
        key_embs: torch.Tensor,
        q_mask: torch.Tensor,
        k_mask: torch.Tensor,
        gumbel_temp: Optional[float] = None,
    ):
        # Normalize embeddings to preserve cosine similarity
        query_embs = query_embs.unsqueeze(1)  # (B, 1, S, H)
        query_embs = F.normalize(query_embs, p=2, dim=-1)
        q_mask = q_mask.unsqueeze(1)  # (B, 1, S)
        key_embs = key_embs.unsqueeze(0)  # (1, B, S, H)
        key_embs = F.normalize(key_embs, p=2, dim=-1)
        k_mask = k_mask.unsqueeze(0)  # (1, B, S)

        # Compute token-level cosine similarities
        sim_matrix = torch.einsum("insh, mjth->ijst", query_embs, key_embs)

        if self.do_distance:
            if self.exp_decay:
                w = torch.exp(-self.alpha * self.distance)
            else:
                w = 1.0 / (1.0 + self.alpha * self.distance)
            sim_matrix = sim_matrix * w.to(sim_matrix.device)

        # Compute valid mask for token pairs
        valid_mask = torch.einsum("ixs, xjt->ijst", q_mask, k_mask).bool()

        if self.use_max:  # Max-based interaction
            masked_sim = sim_matrix.masked_fill(~valid_mask, -float("inf"))
            max_sim_values, _ = masked_sim.max(dim=-1)  # (B, B, S)
            scores = (max_sim_values * q_mask.squeeze(1).unsqueeze(1)).sum(dim=-1)
            scores = scores / q_mask.squeeze(1).sum(dim=-1, keepdim=True).clamp(min=1)
            return scores

        # Scale similarities with learnable logit scale
        scale = torch.exp(self.logit_scale)
        logits = scale * sim_matrix
        logits = logits.masked_fill(
            ~valid_mask, -float("inf")
        )  # Mask invalid positions

        # Apply Gumbel-Softmax with straight-through estimator
        if gumbel_temp is not None:
            soft_p_ij = F.gumbel_softmax(logits, tau=gumbel_temp, hard=False, dim=-1)
            if self.training:
                # Straight-through estimator: hard in forward, soft in backward
                hard_p_ij = F.one_hot(
                    logits.argmax(dim=-1), num_classes=logits.size(-1)
                ).float()
                p_ij = hard_p_ij + (
                    soft_p_ij - soft_p_ij.detach()
                )  # Gradient flows through soft
            else:
                p_ij = soft_p_ij  # Use soft probabilities during inference
        else:
            p_ij = F.softmax(logits, dim=-1)  # Fallback to softmax if no temp

        # Aggregate key embeddings using attention weights
        key_embs_squeezed = key_embs.squeeze(0)  # (B, S, H)
        aggregated = torch.einsum("ijst,jth->ijsh", p_ij, key_embs_squeezed)

        # Compute final scores
        query_embs_expanded = query_embs.squeeze(1)  # (B, S, H)
        scores = (query_embs_expanded.unsqueeze(1) * aggregated).sum(dim=-1)
        scores = scores * q_mask.squeeze(1).unsqueeze(1)
        scores = scores.sum(dim=-1) / q_mask.squeeze(1).sum(dim=-1, keepdim=True).clamp(
            min=1
        )

        return scores
