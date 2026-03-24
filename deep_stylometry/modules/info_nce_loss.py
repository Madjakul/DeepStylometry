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
