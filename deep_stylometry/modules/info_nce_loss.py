# deep_stylometry/modules/info_nce_loss.py

from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int

from deep_stylometry.modules.late_interaction import LateInteraction

if TYPE_CHECKING:
    from deep_stylometry.utils.configs import BaseConfig


class InfoNCELoss(nn.Module):

    def __init__(self, cfg: "BaseConfig") -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("IGNORE", torch.tensor(float("-inf")))

        if self.cfg.model.pooling_method == "li":
            self.pool = LateInteraction(self.cfg)
        else:
            self.pool = self.mean_pooling

    def forward(
        self,
        query_embs: Float[torch.Tensor, "batch seq hidden"],
        key_embs: Float[torch.Tensor, "three_times_batch seq hidden"],
        q_mask: Int[torch.Tensor, "batch seq"],
        k_mask: Int[torch.Tensor, "three_times_batch seq"],
        gumbel_temp: Optional[float] = None,
    ) -> Tuple[
        Float[torch.Tensor, "batch three_times_batch"],
        Float[torch.Tensor, "batch"],
        Float[torch.Tensor, ""],
    ]:
        batch_size = query_embs.size(0)

        # Compute the (B, 3B) similarity matrix
        all_scores = (
            self.pool(
                query_embs=query_embs,  # (B, S, H)
                key_embs=key_embs,  # (3B, S, H)
                q_mask=q_mask,  # (B, S)
                k_mask=k_mask,  # (3B, S)
                gumbel_temp=gumbel_temp,
            )
            / self.cfg.model.contrastive_temp
        )

        # Mask out the anchor–anchor scores in columns [0, ..., B-1]
        # Not in buffer because it would require a fixe batch size
        idx = torch.arange(batch_size)
        all_scores[idx, idx] = self.IGNORE

        # Get positive columns
        # For row i, the positive column is i + 1
        targets = idx + batch_size

        contrastive_loss = F.cross_entropy(all_scores, targets, reduction="mean")

        return all_scores, targets, contrastive_loss

    @staticmethod
    def mean_pooling(
        query_embs: Float[torch.Tensor, "batch seq hidden"],
        key_embs: Float[torch.Tensor, "three_times_batch seq hidden"],
        q_mask: Int[torch.Tensor, "batch seq"],
        k_mask: Int[torch.Tensor, "three_times_batch seq"],
        **kwargs,
    ) -> Float[torch.Tensor, "batch three_times_batch"]:
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
