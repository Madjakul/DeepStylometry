# deep_stylometry/modules/hybrid_loss.py

from typing import TYPE_CHECKING, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int

from deep_stylometry.modules.late_interaction import LateInteraction

if TYPE_CHECKING:
    from deep_stylometry.utils.configs import BaseConfig


class HybridLoss(nn.Module):
    """Hybrid Loss for deep stylometry models. This loss function combines
    InfoNCE and triplet loss to optimize the model's performance on both
    similarity and margin-based similarity tasks.

    Parameters
    ----------
    cfg : BaseConfig
        Configuration object containing model and execution parameters, including the
        margin.

    Attributes
    ----------
    cfg : BaseConfig
        Configuration object with execution parameters.
    pool : nn.Module
        Pooling method used to compute similarity scores between query and key
        embeddings. Can be LateInteraction or mean pooling based on the configuration.
    tau : torch.Tensor
        Temperature parameter for scaling the similarity scores.
    """

    def __init__(self, cfg: "BaseConfig"):
        super().__init__()
        assert cfg.execution.margin is not None
        assert cfg.execution.lambda_ < 1.0 and cfg.execution.lambda_ > 0.0  # type: ignore
        self.cfg = cfg
        self.register_buffer("tau", torch.tensor(self.cfg.execution.tau))

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
    ) -> Dict[str, torch.Tensor]:
        batch_size = query_embs.size(0)

        # Compute the (B, 2B) similarity matrix
        all_scores = self.pool(
            query_embs=query_embs,  # (B, S, H)
            key_embs=key_embs,  # (2B, S, H)
            q_mask=q_mask,  # (B, S)
            k_mask=k_mask,  # (2B, S)
            gumbel_temp=gumbel_temp,
        )
        all_dists = 1 - all_scores
        all_scaled_scores = all_scores / self.tau  # type: ignore

        targets = torch.arange(batch_size, device=query_embs.device)
        poss = all_scores[targets, targets]
        pos_dists = all_dists[targets, targets]
        negs = all_scores[targets, targets + batch_size]
        neg_dists = all_dists[targets, targets + batch_size]

        info_nce_loss = F.cross_entropy(all_scaled_scores, targets, reduction="mean")

        triplet_loss = F.relu(pos_dists - neg_dists + self.cfg.execution.margin).mean()  # type: ignore

        loss = (
            self.cfg.execution.lambda_ * triplet_loss  # type: ignore
            + (1 - self.cfg.execution.lambda_) * info_nce_loss  # type: ignore
        )

        return {
            "all_scores": all_scores,
            "targets": targets,
            "poss": poss,
            "negs": negs,
            "loss": loss,
        }

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
