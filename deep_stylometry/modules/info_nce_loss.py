# deep_stylometry/modules/info_nce_loss.py

from typing import TYPE_CHECKING, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int

from deep_stylometry.modules.late_interaction import LateInteraction
from deep_stylometry.modules.mean_interaction import MeanInteraction

if TYPE_CHECKING:
    from deep_stylometry.utils.configs import BaseConfig


class InfoNCELoss(nn.Module):

    def __init__(self, cfg: "BaseConfig") -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("tau", torch.tensor(self.cfg.execution.tau))

        if self.cfg.model.pooling_method == "li":
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
        all_scaled_scores = all_scores / self.tau  # type: ignore

        targets = torch.arange(batch_size, device=query_embs.device)
        poss = all_scores[targets, targets]
        negs = all_scores[targets, targets + batch_size]

        loss = F.cross_entropy(all_scaled_scores, targets, reduction="mean")

        return {
            "all_scores": all_scores,
            "targets": targets,
            "poss": poss,
            "negs": negs,
            "loss": loss,
        }
