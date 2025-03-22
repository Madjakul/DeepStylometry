# deep_stylometry/modules/late_interaction.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class LateInteraction(nn.Module):
    def __init__(self, gumbel_temp: float = 0.5):
        super().__init__()
        self.gumbel_temp = gumbel_temp
        self.eps = 1e-10

    def forward(
        self,
        query_embs: torch.Tensor,
        key_embs: torch.Tensor,
        q_mask: torch.Tensor,
        k_mask: torch.Tensor,
    ):
        """
        Args:
            query_embs: (batch_size, seq_len, hidden_size)
            key_embs: (batch_size, seq_len, hidden_size)
            q_mask: (batch_size, seq_len)
            k_mask: (batch_size, seq_len)
        Returns:
            scores: (batch_size,)
        """
        # Normalize embeddings
        query_embs = F.normalize(query_embs, p=2, dim=-1)
        key_embs = F.normalize(key_embs, p=2, dim=-1)

        # Compute token-level similarities
        sim_matrix = torch.einsum("bqh,bkh->bqk", query_embs, key_embs)

        # Create combined mask for valid token pairs
        valid_mask = torch.einsum("bq,bk->bqk", q_mask, k_mask)

        if self.training:
            # Add Gumbel noise to valid positions only
            uniform = torch.rand_like(sim_matrix) * (1 - self.eps) + self.eps
            gumbel_noise = -torch.log(-torch.log(uniform))
            noisy_sim = (sim_matrix + gumbel_noise * valid_mask) / self.gumbel_temp
        else:
            noisy_sim = sim_matrix / self.gumbel_temp

        # Mask invalid positions with large negative value
        noisy_sim = noisy_sim * valid_mask - (1 - valid_mask) * 1e9

        # Compute attention distribution
        p_ij = F.softmax(noisy_sim, dim=-1)

        # Aggregate key embeddings
        aggregated = torch.einsum("bqk,bkh->bqh", p_ij, key_embs)

        # Compute final scores (masked mean)
        scores = (query_embs * aggregated).sum(dim=-1) * q_mask
        scores = scores.sum(dim=-1) / q_mask.sum(dim=-1).clamp(min=1)
        return scores
