# deep_stylometry/modules/triplet_loss.py

from typing import TYPE_CHECKING, Dict

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
        targets: Int[torch.Tensor, "batch"],
        q_input_ids: Int[torch.Tensor, "batch seq"],
    ) -> Dict[str, torch.Tensor]:
        batch_size = query_embs.size(0)
        neg_offset = key_embs.size(0) // 2

        # Compute the (B, 2B_global) similarity matrix
        all_scores = self.pool(
            query_embs=query_embs,
            key_embs=key_embs,
            q_mask=q_mask,
            k_mask=k_mask,
            q_input_ids=q_input_ids,
        )

        all_dists = 1 - all_scores

        rows = torch.arange(batch_size, device=query_embs.device)
        poss = all_scores[rows, targets]
        pos_dists = all_dists[rows, targets]

        negs = all_scores[rows, targets + neg_offset]
        neg_dists = all_dists[rows, targets + neg_offset]

        loss = F.relu(pos_dists - neg_dists + self.cfg.execution.margin).mean()

        return {
            "all_scores": all_scores,
            "targets": targets,
            "poss": poss,
            "negs": negs,
            "loss": loss,
        }
