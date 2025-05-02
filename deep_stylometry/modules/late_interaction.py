# deep_stylometry/modules/late_interaction.py

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LateInteraction(nn.Module):
    def __init__(
        self,
        do_distance: bool,
        exp_decay: bool,
        seq_len: int,
        alpha: float = 0.5,
        use_max: bool = False,  # New parameter
    ):
        super().__init__()
        self.do_distance = do_distance
        self.distance: torch.Tensor
        self.use_max = use_max  # Track max mode

        if self.do_distance:
            self.alpha_raw = nn.Parameter(torch.tensor(alpha))
            # Pre-compute the distance matrix (doesn't depend on alpha)
            # and register it as a buffer
            positions = torch.arange(seq_len)
            i = positions.unsqueeze(1)
            j = positions.unsqueeze(0)
            distance = (i - j).abs().float()  # Ensure float type
            self.register_buffer("distance", distance)  # Store distance

        self.exp_decay = exp_decay
        self.eps = 1e-9

    @property
    def alpha(self):
        """Softplus ensures alpha > 0."""
        return F.softplus(self.alpha_raw)  # log(1 + exp(x))

    def forward(
        self,
        query_embs: torch.Tensor,
        key_embs: torch.Tensor,
        q_mask: torch.Tensor,
        k_mask: torch.Tensor,
        gumbel_temp: Optional[float] = None,
    ):
        # Normalize embeddings
        query_embs = query_embs.unsqueeze(1)  # (B, 1, S, H)
        query_embs = F.normalize(query_embs, p=2, dim=-1)
        q_mask = q_mask.unsqueeze(1)  # (B, 1, S)
        key_embs = key_embs.unsqueeze(0)  # (1, B, S, H)
        key_embs = F.normalize(key_embs, p=2, dim=-1)
        k_mask = k_mask.unsqueeze(0)  # (1, B, S)

        # Compute token-level similarities
        sim_matrix = torch.einsum("i n s h, m j t h -> i j s t", query_embs, key_embs)

        if self.do_distance:
            if self.exp_decay:
                print(self.alpha)
                w = torch.exp(-self.alpha * self.distance)  # Differentiable!
            else:
                w = 1.0 / (1.0 + self.alpha * self.distance)
            sim_matrix = sim_matrix * w  # Create combined mask for valid token pairs

        valid_mask = torch.einsum("i x s, x j t -> i j s t", q_mask, k_mask)

        if self.use_max:  # MAX-BASED INTERACTION ############################
            # Replace invalid positions with -inf
            masked_sim = sim_matrix * valid_mask - (1 - valid_mask) * 1e9
            # Take max over key tokens (dim=-1)
            max_sim_values, _ = masked_sim.max(dim=-1)  # (B, B, S)
            # Aggregate over query tokens
            scores = (max_sim_values * q_mask.squeeze(1).unsqueeze(1)).sum(dim=-1)
            scores = scores / q_mask.squeeze(1).sum(dim=-1, keepdim=True).clamp(min=1)
            return scores

        # Original Gumbel/Softmax logic ######################################
        # Apply Gumbel noise during training
        if self.training and gumbel_temp is not None:
            uniform = torch.rand_like(sim_matrix) * (1 - self.eps) + self.eps
            gumbel_noise = -torch.log(-torch.log(uniform))
            noisy_sim = (sim_matrix + gumbel_noise * valid_mask) / gumbel_temp
        else:
            noisy_sim = (
                sim_matrix / gumbel_temp if gumbel_temp is not None else sim_matrix
            )

        # Mask invalid positions
        noisy_sim = noisy_sim * valid_mask - (1 - valid_mask) * 1e9

        # Compute attention distribution
        p_ij = F.softmax(noisy_sim, dim=-1)  # (B, B, S, S)

        # Aggregate key embeddings
        key_embs_squeezed = key_embs.squeeze(0)  # (B, S, H)
        aggregated = torch.einsum("ijst,jth->ijsh", p_ij, key_embs_squeezed)

        # Compute scores (B, B)
        query_embs_expanded = query_embs.squeeze(1)  # (B, S, H)
        scores = (query_embs_expanded.unsqueeze(1) * aggregated).sum(dim=-1)
        scores = scores * q_mask.squeeze(1).unsqueeze(1)
        scores = scores.sum(dim=-1) / q_mask.squeeze(1).sum(dim=-1, keepdim=True).clamp(
            min=1
        )

        return scores
