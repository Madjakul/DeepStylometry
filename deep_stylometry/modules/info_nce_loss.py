# deep_stylometry/modules/info_nce_loss.py

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_stylometry.modules.late_interaction import LateInteraction


class InfoNCELoss(nn.Module):

    def __init__(
        self,
        seq_len: int,
        use_max: bool = True,
        alpha: float = 1.0,
        temperature: float = 0.07,
        pooling_method: str = "mean",
        distance_weightning: str = "none",
    ):
        super().__init__()
        assert pooling_method in (
            "mean",
            "li",
        ), "Pooling method must be 'mean' or 'li' for late interaction."

        self.temperature = temperature

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
        key_embs: torch.Tensor,
        labels: torch.Tensor,
        q_mask: torch.Tensor,
        k_mask: torch.Tensor,
        gumbel_temp: Optional[float] = None,
    ):
        batch_size = query_embs.size(0)

        # --- 1. Compute the (B, B) similarity matrix ---
        all_scores = (
            self.pool(
                query_embs=query_embs,  # (B, S, H)
                key_embs=key_embs,  # (B, S, H)
                q_mask=q_mask,  # (B, S)
                k_mask=k_mask,  # (B, S)
                gumbel_temp=gumbel_temp,
            )
            / self.temperature
        )

        # --- 2. Identify positive pairs ---
        # labels are 1 if (q_i, k_i) is a positive pair, 0 otherwise
        pos_mask = labels.bool()  # (B,)

        # Early exit if no positive pairs in the batch
        if pos_mask.sum() == 0:
            return (
                torch.empty(0, batch_size, device=query_embs.device),
                torch.empty(0, dtype=torch.long, device=query_embs.device),
                torch.tensor(0.0, device=query_embs.device, requires_grad=True),
            )

        # --- 3. Prepare for CrossEntropyLoss ---
        # Select rows corresponding to positive queries
        pos_query_scores = all_scores[pos_mask]  # (num_pos, B)

        # Targets: For a positive query i, the positive key is at index i
        pos_query_targets = torch.arange(batch_size, device=all_scores.device)[
            pos_mask
        ]  # (num_pos,)

        # --- 4. Compute InfoNCE loss ---
        contrastive_loss = F.cross_entropy(
            pos_query_scores, pos_query_targets, reduction="mean"
        )

        return all_scores, pos_query_scores, pos_query_targets, contrastive_loss

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
