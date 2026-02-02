# deep_stylometry/modules/triplet_loss.py

from typing import TYPE_CHECKING, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int

from deep_stylometry.modules.late_interaction import LateInteraction
from deep_stylometry.modules.mean_interaction import MeanInteraction

if TYPE_CHECKING:
    from deep_stylometry.utils.configs import BaseConfig


class TripletLoss(nn.Module):

    def __init__(self, cfg: "BaseConfig") -> None:
        super().__init__()
        assert cfg.execution.margin is not None
        self.cfg = cfg

        if cfg.model.pooling_method == "li":
            self.pool = LateInteraction(self.cfg)
        else:
            self.pool = MeanInteraction()

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
        assert (
            torch.all(all_scores >= -1.0 - 1e-6)
            and torch.all(all_scores <= 1.0 + 1e-6),
            f"Scores out of bounds: {all_scores.min()} to {all_scores.max()}",
        )
        all_dists = 1 - all_scores
        q_mask_sum = q_mask.sum(dim=1)

        targets = torch.arange(batch_size, device=query_embs.device)
        poss = all_scores[targets, targets]
        pos_dists = all_dists[targets, targets]
        negs = all_scores[targets, targets + batch_size]
        neg_dists = all_dists[targets, targets + batch_size]

        base_margin = self.cfg.execution.margin
        if self.cfg.model.pooling_method == "li":
            dynamic_margin = base_margin / q_mask_sum.float()  # shape: (B,)
        else:
            dynamic_margin = base_margin
        loss = F.relu(pos_dists - neg_dists + dynamic_margin).mean()  # type: ignore

        return {
            "all_scores": all_scores,
            "targets": targets,
            "poss": poss,
            "negs": negs,
            "loss": loss,
        }
