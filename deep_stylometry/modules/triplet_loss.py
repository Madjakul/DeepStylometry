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
    """Margin-based triplet ranking loss.

    Scores each query against the full key set, then applies a hinge loss
    that penalises configurations where the positive distance exceeds the
    negative distance by less than ``cfg.train.margin``.

    Parameters
    ----------
    cfg : BaseConfig
        Global configuration. ``cfg.train.margin`` must be set.
        ``cfg.model.pooling_method`` selects the scoring function.
    """

    def __init__(self, cfg: "BaseConfig") -> None:
        super().__init__()
        assert cfg.train.margin is not None
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
        """Compute triplet loss and per-example positive/negative scores.

        Parameters
        ----------
        query_embs : Float[Tensor, "batch seq hidden"]
            Per-token query embeddings.
        key_embs : Float[Tensor, "two_times_batch seq hidden"]
            Concatenated positive and negative key embeddings.
        q_mask : Int[Tensor, "batch seq"]
            Attention mask for queries.
        k_mask : Int[Tensor, "two_times_batch seq"]
            Attention mask for keys.
        targets : Int[Tensor, "batch"]
            Index into ``key_embs`` of each query's positive.
        q_input_ids : Int[Tensor, "batch seq"]
            Query token IDs forwarded to LateInteraction for skip-list masking.

        Returns
        -------
        dict
            Keys: ``loss``, ``all_scores``, ``targets``, ``poss``, ``negs``.
        """
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

        loss = F.relu(pos_dists - neg_dists + self.cfg.train.margin).mean()

        return {
            "all_scores": all_scores,
            "targets": targets,
            "poss": poss,
            "negs": negs,
            "loss": loss,
        }
