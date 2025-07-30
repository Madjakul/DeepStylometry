# deep_stylometry/modules/late_interaction.py

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int

if TYPE_CHECKING:
    from deep_stylometry.utils.configs import BaseConfig


class LateInteraction(nn.Module):
    """Custom late interaction module for deep stylometry models. This module
    computes the similarity scores between query and key embeddings using
    cosine similarity, applies distance weighting if specified, and aggregates
    the results to produce final scores. The module supports both max-based
    interaction and softmax-based interaction based on the configuration.

    Parameters
    ----------
    cfg : BaseConfig
        Configuration object containing model parameters, including distance weighting
        and alpha values.

    Attributes
    ----------
    cfg : BaseConfig
        Configuration object with model parameters.
    IGNORE : torch.Tensor
        A tensor used to mask invalid interactions, initialized to negative infinity.
    alpha_raw : nn.Parameter
        A learnable parameter representing the alpha value for distance weighting.
    distance : torch.Tensor
        A precomputed distance matrix used for distance weighting, registered as a
        buffer.
    """

    def __init__(self, cfg: "BaseConfig") -> None:
        super().__init__()
        self.cfg = cfg

        self.register_buffer("IGNORE", torch.tensor(float("-inf")))

        if self.cfg.model.distance_weightning != "none":
            self.alpha_raw = nn.Parameter(torch.tensor(self.cfg.model.alpha))
            positions = torch.arange(self.cfg.data.max_length)
            distance = (positions.unsqueeze(1) - positions.unsqueeze(0)).abs().float()
            self.register_buffer("distance", distance)

    @property
    def alpha(self) -> Float[torch.Tensor, ""]:
        """Leaky ReLU ensures alpha >= 0 and has a non-saturating gradient for
        positive values and won't be stuck when it gets slightly below zero."""
        return F.leaky_relu(self.alpha_raw)

    def forward(
        self,
        query_embs: Float[torch.Tensor, "batch seq hidden"],
        key_embs: Float[torch.Tensor, "two_times_batch seq hidden"],
        q_mask: Int[torch.Tensor, "batch seq"],
        k_mask: Int[torch.Tensor, "two_times_batch seq"],
        gumbel_temp: Optional[float] = None,
    ) -> Float[torch.Tensor, "batch two_times_batch"]:
        query_embs = query_embs.unsqueeze(1)  # (B, 1, S, H)
        query_embs = F.normalize(query_embs, p=2, dim=-1)
        q_mask = q_mask.unsqueeze(1)  # (B, 1, S)
        key_embs = key_embs.unsqueeze(0)  # (1, 2B, S, H)
        key_embs = F.normalize(key_embs, p=2, dim=-1)
        k_mask = k_mask.unsqueeze(0)  # (1, 2B, S)

        # Compute token-level cosine similarities
        sim_matrix = torch.einsum("insh, mjth->ijst", query_embs, key_embs)

        if self.cfg.model.distance_weightning == "exp":
            w = torch.exp(-self.alpha * self.distance)  # type: ignore
            sim_matrix = sim_matrix * w  # .to(sim_matrix.device)
        elif self.cfg.model.distance_weightning == "linear":
            w = 1.0 / (1.0 + self.alpha * self.distance)  # type: ignore
            sim_matrix = sim_matrix * w

        # Compute valid mask for token pairs
        valid_mask = torch.einsum("ixs, xjt->ijst", q_mask, k_mask).bool()

        if not self.cfg.model.use_softmax:
            # Max-based interaction
            masked_sim = sim_matrix.masked_fill(~valid_mask, self.IGNORE)  # type: ignore
            max_sim_values, _ = masked_sim.max(dim=-1)  # (B, B, S)
            is_padding_mask = q_mask == 0
            masked_max_sim = max_sim_values.masked_fill(is_padding_mask, 0.0)
            scores = masked_max_sim.sum(dim=-1)
            return scores

        # Mask the padding tokens
        logits = sim_matrix.masked_fill(~valid_mask, self.IGNORE)  # type: ignore

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
