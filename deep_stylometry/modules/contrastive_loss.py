# deep_stylometry/modules/ccontrastive_loss.py

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_stylometry.modules.late_interaction import LateInteraction


class ContrastiveLoss(nn.Module):

    def __init__(
        self,
        seq_len: int,
        use_max: bool = True,
        alpha: float = 1.0,
        margin: float = 0.2,
        pooling_method: str = "mean",
        distance_weightning: str = "none",
    ):
        super().__init__()
        assert pooling_method in (
            "mean",
            "li",
        ), "Pooling method must be 'mean' or 'li' for late interaction."

        self.margin = margin

        if pooling_method == "li":
            self.pool = LateInteraction(
                alpha=alpha,
                seq_len=seq_len,
                use_max=use_max,
                distance_weightning=distance_weightning,
            )
        else:
            self.pool = self.mean_pooling

    def forward(
        self,
        query_embs: torch.Tensor,
        pos_embs: torch.Tensor,
        neg_embs: torch.Tensor,
        q_mask: torch.Tensor,
        pos_mask: torch.Tensor,
        neg_mask: torch.Tensor,
        gumbel_temp: Optional[float] = None,
    ):
        batch_size = query_embs.size(0)

        # --- 1. Compute the (B, B) similarity matrix ---
        pos_scores = self.pool(
            query_embs=query_embs,  # (B, S, H)
            key_embs=pos_embs,  # (B, S, H)
            q_mask=q_mask,  # (B, S)
            k_mask=pos_mask,  # (B, S)
            gumbel_temp=gumbel_temp,
        ).diag()

        neg_scores = self.pool(
            query_embs=query_embs,  # (B, S, H)
            key_embs=neg_embs,  # (B, S, H)
            q_mask=q_mask,  # (B, S)
            k_mask=neg_mask,  # (B, S)
            gumbel_temp=gumbel_temp,
        ).diag()

        loss = F.relu(self.margin - pos_scores + neg_scores).mean()

        return pos_scores, neg_scores, loss

    @staticmethod
    def mean_pooling(
        query_embs: torch.Tensor,
        key_embs: torch.Tensor,
        q_mask: torch.Tensor,
        k_mask: torch.Tensor,
        **kwargs,
    ):
        # Mean pooling and normalization
        q_mask_sum = q_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        query_vec = (query_embs * q_mask.unsqueeze(-1)).sum(dim=1) / q_mask_sum
        query_vec = F.normalize(query_vec, p=2, dim=-1)

        k_mask_sum = k_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        key_vec = (key_embs * k_mask.unsqueeze(-1)).sum(dim=1) / k_mask_sum
        key_vec = F.normalize(key_vec, p=2, dim=-1)

        # Calculate cosine similarity matrix (B, B)
        all_scores = torch.matmul(query_vec, key_vec.T)
        return all_scores
