# deep_stylometry/modules/info_nce_loss.py

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_stylometry.modules.late_interaction import LateInteraction


class InfoNCELoss(nn.Module):
    def __init__(
        self,
        do_late_interaction: bool,
        do_distance: bool,
        exp_decay: bool,
        seq_len: int,
        alpha: float = 0.5,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.temperature = temperature
        self.do_late_interaction = do_late_interaction
        self.do_distance = do_distance
        if self.do_late_interaction:
            self.late_interaction = LateInteraction(
                do_distance=do_distance,
                exp_decay=exp_decay,
                alpha=alpha,
                seq_len=seq_len,
            )

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
        pos_mask = labels.bool()

        # Compute all pairwise scores (query x key)
        if self.do_late_interaction:
            all_scores = (
                self.late_interaction(
                    query_embs.unsqueeze(1),  # (B, 1, S, H)
                    key_embs.unsqueeze(0),  # (1, B, S, H)
                    q_mask.unsqueeze(1),
                    k_mask.unsqueeze(0),
                    gumbel_temp=gumbel_temp,
                )
                / self.temperature
            )
        else:
            # Mean pooling and normalization
            q_mask_sum = q_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
            query_vec = (query_embs * q_mask.unsqueeze(-1)).sum(dim=1) / q_mask_sum
            query_vec = F.normalize(query_vec, p=2, dim=-1)

            k_mask_sum = k_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
            key_vec = (key_embs * k_mask.unsqueeze(-1)).sum(dim=1) / k_mask_sum
            key_vec = F.normalize(key_vec, p=2, dim=-1)

            all_scores = torch.matmul(query_vec, key_vec.T) / self.temperature

        # For each positive pair (query, key), mask out the positive key in the denominator
        # and include all other keys (in-batch + hard negatives)
        pos_indices = torch.arange(batch_size, device=query_embs.device)
        loss = F.cross_entropy(
            all_scores,
            pos_indices,
            reduction="none",
        )

        # Apply mask to only compute loss for positive pairs
        loss = loss[pos_mask].mean()

        return loss if not torch.isnan(loss) else torch.tensor(0.0, device=loss.device)
