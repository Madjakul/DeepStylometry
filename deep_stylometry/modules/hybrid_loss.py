# deep_stylometry/modules/hybrid_loss.py

from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int

from deep_stylometry.modules.late_interaction import LateInteraction

if TYPE_CHECKING:
    from deep_stylometry.utils.configs import BaseConfig


class HybridLoss(nn.Module):

    def __init__(self, cfg: "BaseConfig"):
        super().__init__()
        assert cfg.train.margin is not None
        self.cfg = cfg

        if cfg.model.pooling_method == "li":
            self.pool = LateInteraction(self.cfg)
        else:
            self.pool = self.mean_pooling

    def forward(
        self,
        query_embs: Float[torch.Tensor, "batch seq hidden"],
        key_embs: Float[torch.Tensor, "two_times_batch seq hidden"],
        q_mask: Int[torch.Tensor, "batch seq"],
        k_mask: Int[torch.Tensor, "two_times_batch seq"],
        gumbel_temp: Optional[float] = None,
    ) -> Tuple[
        Float[torch.Tensor, "batch two_times_batch"],
        Int[torch.Tensor, "batch"],
        Float[torch.Tensor, ""],
    ]:
        batch_size = query_embs.size(0)

        # Compute the (B, 2B) similarity matrix
        all_scores = self.pool(
            query_embs=query_embs,  # (B, S, H)
            key_embs=key_embs,  # (2B, S, H)
            q_mask=q_mask,  # (B, S)
            k_mask=k_mask,  # (2B, S)
            gumbel_temp=gumbel_temp,
        )
        all_distances = 1 - all_scores

        targets = torch.arange(batch_size, device=query_embs.device)

        contrastive_loss = F.cross_entropy(all_scores, targets, reduction="mean")

        poss = all_distances[targets, targets]
        negs = all_distances[targets, targets + batch_size]

        # Select hard positive and hard negative pairs
        # Negatives that are too close
        negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
        # Positives that are too far
        positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]

        positive_loss = positive_pairs.pow(2).sum()
        negative_loss = F.relu(self.cfg.execution.margin - negative_pairs).pow(2).sum()
        margin_loss = (positive_loss + negative_loss) / batch_size

        loss = contrastive_loss + margin_loss

        return all_scores, targets, loss

    @staticmethod
    def mean_pooling(
        query_embs: Float[torch.Tensor, "batch seq hidden"],
        key_embs: Float[torch.Tensor, "two_times_batch seq hidden"],
        q_mask: Int[torch.Tensor, "batch seq"],
        k_mask: Int[torch.Tensor, "two_times_batch seq"],
        **kwargs,
    ) -> Float[torch.Tensor, "batch two_times_batch"]:
        # Mean pooling and normalization
        q_mask_sum = q_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        query_vec = (query_embs * q_mask.unsqueeze(-1)).sum(dim=1) / q_mask_sum
        query_vec = F.normalize(query_vec, p=2, dim=-1)

        k_mask_sum = k_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        key_vec = (key_embs * k_mask.unsqueeze(-1)).sum(dim=1) / k_mask_sum
        key_vec = F.normalize(key_vec, p=2, dim=-1)

        all_scores = torch.matmul(query_vec, key_vec.T)
        return all_scores
