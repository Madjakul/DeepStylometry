# deep_stylometry/modules/late_interaction.py

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int

if TYPE_CHECKING:
    from deep_stylometry.utils.configs import BaseConfig


class LateInteraction(nn.Module):

    def __init__(self, cfg: "BaseConfig") -> None:
        super().__init__()
        self.cfg = cfg

        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))
        self.register_buffer("IGNORE", torch.tensor(float("-inf")))

        if self.cfg.model.distance_weightning != "none":
            # self.distance: torch.Tensor
            self.alpha_raw = nn.Parameter(torch.tensor(self.cfg.model.alpha))
            positions = torch.arange(self.cfg.data.max_length)
            distance = (positions.unsqueeze(1) - positions.unsqueeze(0)).abs().float()
            self.register_buffer("distance", distance)

    @property
    def alpha(self) -> Float[torch.Tensor, ""]:
        """Leaky ReLU ensures alpha >= 0 and has a non-saturating gradient for
        positive values and won't be stuck when it gets slightly below zero."""
        return F.leaky_relu(self.alpha_raw)

    @property
    def scale(self) -> Float[torch.Tensor, ""]:
        """Exponentiate the scale to get the actual scaling factor."""
        return torch.exp(self.logit_scale)

    def forward(
        self,
        query_embs: Float[torch.Tensor, "batch seq hidden"],
        key_embs: Float[torch.Tensor, "two_times_batch seq hidden"],
        q_mask: Int[torch.Tensor, "batch seq"],
        k_mask: Int[torch.Tensor, "two_times_batch seq"],
        gumbel_temp: Optional[float] = None,
    ) -> Float[torch.Tensor, "batch two_times_batch"]:
        # Normalize embeddings to preserve cosine similarity
        query_embs = query_embs.unsqueeze(1)  # (B, 1, S, H)
        query_embs = F.normalize(query_embs, p=2, dim=-1)
        q_mask = q_mask.unsqueeze(1)  # (B, 1, S)
        key_embs = key_embs.unsqueeze(0)  # (1, 2B, S, H)
        key_embs = F.normalize(key_embs, p=2, dim=-1)
        k_mask = k_mask.unsqueeze(0)  # (1, 2B, S)

        # Compute token-level cosine similarities
        sim_matrix = torch.einsum("insh, mjth->ijst", query_embs, key_embs)

        if self.cfg.model.distance_weightning == "exp":
            w = torch.exp(-self.alpha * self.distance)
            sim_matrix = sim_matrix * w  # .to(sim_matrix.device)
        elif self.cfg.model.distance_weightning == "linear":
            w = 1.0 / (1.0 + self.alpha * self.distance)
            sim_matrix = sim_matrix * w  # .to(sim_matrix.device)

        # Compute valid mask for token pairs
        valid_mask = torch.einsum("ixs, xjt->ijst", q_mask, k_mask).bool()

        if not self.cfg.model.use_softmax:
            # Max-based interaction
            sim_matrix_scaled = self.scale * sim_matrix
            masked_sim = sim_matrix_scaled.masked_fill(~valid_mask, self.IGNORE)
            max_sim_values, _ = masked_sim.max(dim=-1)  # (B, B, S)
            scores = (max_sim_values * q_mask.squeeze(1).unsqueeze(1)).sum(dim=-1)
            return scores

        # Scale similarities with learnable logit scale
        logits = self.scale * sim_matrix

        # Mask the padding tokens
        logits = logits.masked_fill(~valid_mask, self.IGNORE)

        if gumbel_temp is not None:
            p_ij = F.gumbel_softmax(logits, tau=gumbel_temp, hard=False)
            p_ij = torch.nan_to_num(p_ij, nan=0.0)
        else:
            p_ij = F.softmax(logits, dim=-1)
            p_ij = torch.nan_to_num(p_ij, nan=0.0)

        key_embs_squeezed = key_embs.squeeze(0)  # (2B, S, H)
        aggregated = torch.einsum("ijst, jth->ijsh", p_ij, key_embs_squeezed)

        # Compute final scores
        query_embs_expanded = query_embs.squeeze(1)  # (B, S, H)
        scores = (query_embs_expanded.unsqueeze(1) * aggregated).sum(dim=-1)
        scores = scores * q_mask.squeeze(1).unsqueeze(1)
        scores = scores.sum(dim=-1)

        return scores
