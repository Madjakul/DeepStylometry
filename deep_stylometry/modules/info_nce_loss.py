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
    """In-batch InfoNCE (NT-Xent) contrastive loss.

    Scores each query against all in-batch keys using the configured
    interaction function (mean, late interaction, or patch interaction),
    then applies cross-entropy loss against the correct positive index.

    Parameters
    ----------
    cfg : BaseConfig
        Global configuration. ``cfg.train.tau`` controls the temperature;
        ``cfg.model.pooling_method`` selects the interaction function.
    """

    def __init__(self, cfg: "BaseConfig") -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("tau", torch.tensor(self.cfg.train.tau))

        if self.cfg.model.pooling_method == "li":
            self.pool = LateInteraction(self.cfg)
        elif self.cfg.model.pooling_method == "pli":
            from deep_stylometry.modules.patch_interaction import PatchInteraction
            self.pool = PatchInteraction(self.cfg)
        else:
            self.pool = MeanInteraction()

    def forward(
        self,
        query_embs: Float[torch.Tensor, "batch seq hidden"],
        key_embs: Float[torch.Tensor, "two_times_batch seq hidden"],
        q_mask: Int[torch.Tensor, "batch seq"],
        k_mask: Int[torch.Tensor, "two_times_batch seq"],
        targets: Int[torch.Tensor, "batch"],
        q_input_ids: Optional[Int[torch.Tensor, "batch seq"]] = None,
        k_input_ids: Optional[Int[torch.Tensor, "two_times_batch seq"]] = None,
        step: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute InfoNCE loss and per-example positive/negative scores.

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
        q_input_ids : Int[Tensor, "batch seq"], optional
            Query token IDs (required by LateInteraction and PatchInteraction).
        k_input_ids : Int[Tensor, "two_times_batch seq"], optional
            Key token IDs (required by PatchInteraction).
        step : int, optional
            Current training step (for Gumbel temperature annealing).

        Returns
        -------
        dict
            Keys: ``loss``, ``all_scores``, ``poss``, ``negs``, and optionally
            ``patch_reg_loss`` when learned PLI is active.
        """
        batch_size = query_embs.size(0)
        neg_offset = key_embs.size(0) // 2

        # Build pool kwargs (PatchInteraction accepts extra args)
        pool_kwargs: Dict = dict(
            query_embs=query_embs,
            key_embs=key_embs,
            q_mask=q_mask,
            k_mask=k_mask,
        )
        from deep_stylometry.modules.patch_interaction import PatchInteraction

        if isinstance(self.pool, (LateInteraction, PatchInteraction)):
            pool_kwargs["q_input_ids"] = q_input_ids
        if isinstance(self.pool, PatchInteraction):
            pool_kwargs["k_input_ids"] = k_input_ids
            pool_kwargs["step"] = step

        all_scores = self.pool(**pool_kwargs)
        all_scaled_scores = all_scores / self.tau  # type: ignore

        rows = torch.arange(batch_size, device=query_embs.device)
        poss = all_scores[rows, targets]
        negs = all_scores[rows, targets + neg_offset]

        loss = F.cross_entropy(all_scaled_scores, targets, reduction="mean")

        result: Dict[str, torch.Tensor] = {
            "all_scores": all_scores,
            "poss": poss,
            "negs": negs,
            "loss": loss,
        }

        # Patch regulariser (only for PatchInteraction with learned patching)
        if isinstance(self.pool, PatchInteraction):
            patch_reg = self.pool.get_patch_reg_loss()
            if patch_reg is not None:
                result["patch_reg_loss"] = patch_reg

        return result
