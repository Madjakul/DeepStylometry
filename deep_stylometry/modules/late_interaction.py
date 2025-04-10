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
    ):
        super().__init__()
        self.do_distance = do_distance
        self.distance: torch.Tensor
        if self.do_distance:
            positions = torch.arange(seq_len)
            i = positions.unsqueeze(1)
            j = positions.unsqueeze(0)
            distance = (i - j).abs()
            self.register_buffer("distance", distance)
        self.exp_decay = exp_decay
        self.alpha = alpha
        self.eps = 1e-9

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
            w = (
                torch.exp(-self.alpha * self.distance)
                if self.exp_decay
                else 1.0 / (1.0 + self.distance)
            )
            sim_matrix = sim_matrix * w

        # Create combined mask for valid token pairs
        valid_mask = torch.einsum("i x s, x j t -> i j s t", q_mask, k_mask)

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
        aggregated = torch.einsum(
            "ijst,jth->ijsh", p_ij, key_embs_squeezed
        )  # (B, B, S, H)

        # Compute scores (B, B)
        query_embs_expanded = query_embs.squeeze(1)  # (B, S, H)
        scores = (query_embs_expanded.unsqueeze(1) * aggregated).sum(
            dim=-1
        )  # (B, B, S)

        # Apply mask and normalize
        scores = scores * q_mask.squeeze(1).unsqueeze(1)  # Corrected line: (B, 1, S)
        scores = scores.sum(dim=-1) / q_mask.squeeze(1).sum(dim=-1, keepdim=True).clamp(
            min=1
        )

        return scores


# deep_stylometry/modules/late_interaction.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class LateInteraction(nn.Module):
#     def __init__(self, do_distance, exp_decay, seq_len, alpha=0.5):
#         super().__init__()
#         self.do_distance = do_distance
#         if self.do_distance:
#             positions = torch.arange(seq_len)
#             i = positions.unsqueeze(1)
#             j = positions.unsqueeze(0)
#             distance = (i - j).abs()
#             self.register_buffer("distance", distance, persistent=False)  # Important!
#         self.exp_decay = exp_decay
#         self.alpha = alpha
#         self.eps = 1e-9
#
#     def forward(self, query_embs, key_embs, q_mask, k_mask, gumbel_temp):
#         print(f"\n--- Inside LateInteraction ---")  # Added newline for clarity
#         N = query_embs.shape[0]
#         M = key_embs.shape[1]
#         print(f"Batch sizes N(query)={N}, M(key)={M}")
#         print(
#             f"Input shapes: Q={query_embs.shape}, K={key_embs.shape}, qM={q_mask.shape}, kM={k_mask.shape}"
#         )
#
#         query_embs = F.normalize(query_embs, p=2, dim=-1)
#         key_embs = F.normalize(key_embs, p=2, dim=-1)
#
#         sim_matrix = torch.einsum("ilqh,mjkh->ijqk", query_embs, key_embs)
#         print(
#             f"Shape after einsum 1 (sim_matrix): {sim_matrix.shape} -- Expected ({N}, {M}, S, S)"
#         )
#
#         # ... (Optional: distance weighting) ...
#
#         valid_mask = q_mask.unsqueeze(3) * k_mask.unsqueeze(2)
#         valid_mask = valid_mask.float()
#         print(
#             f"Shape after valid_mask calc: {valid_mask.shape} -- Expected ({N}, {M}, S, S)"
#         )
#
#         # ... (Optional: Gumbel noise calculation) ...
#         # Use sim_matrix directly if not training or no gumbel_temp
#         if self.training and gumbel_temp is not None:
#             uniform = torch.rand_like(sim_matrix) * (1 - self.eps) + self.eps
#             gumbel_noise = -torch.log(-torch.log(uniform))
#             noisy_sim = (sim_matrix + gumbel_noise * valid_mask) / gumbel_temp
#         else:
#             noisy_sim = sim_matrix
#
#         noisy_sim = noisy_sim.masked_fill(valid_mask == 0, -1e9)
#
#         p_ijkq = F.softmax(noisy_sim, dim=-1)
#         print(
#             f"Shape after softmax (p_ijkq): {p_ijkq.shape} -- Expected ({N}, {M}, S, S)"
#         )
#
#         aggregated = torch.einsum("ijqk,mjkh->ijqh", p_ijkq, key_embs)
#         print(
#             f"Shape after einsum 2 (aggregated): {aggregated.shape} -- Expected ({N}, {M}, S, H)"
#         )
#
#         scores_per_token = (query_embs * aggregated).sum(dim=-1)
#         print(
#             f"Shape after scores_per_token sum: {scores_per_token.shape} -- Expected ({N}, {M}, S)"
#         )
#
#         masked_scores = scores_per_token * q_mask
#         print(
#             f"Shape after masking scores (masked_scores): {masked_scores.shape} -- Expected ({N}, {M}, S)"
#         )
#
#         summed_scores = masked_scores.sum(dim=-1)  # Sum over S
#         print(
#             f"Shape after summing scores (summed_scores): {summed_scores.shape} -- Expected ({N}, {M})"
#         )
#
#         q_mask_sum = q_mask.sum(dim=-1).clamp(min=self.eps)
#         print(f"Shape of q_mask_sum: {q_mask_sum.shape} -- Expected ({N}, 1)")
#
#         final_scores = summed_scores / q_mask_sum
#         print(
#             f"Shape after final division (final_scores): {final_scores.shape} -- Expected ({N}, {M})"
#         )
#         print(f"--- Exiting LateInteraction ---")
#
#         # Add final check to be extra sure
#         if final_scores.shape != (N, M):
#             raise RuntimeError(
#                 f"LateInteraction Error: final_scores shape is {final_scores.shape}, expected ({N}, {M})"
#             )
#
#         return final_scores


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class LateInteraction(nn.Module):
#     def __init__(
#         self,
#         do_distance: bool,
#         exp_decay: bool,
#         seq_len: int,
#         alpha: float = 0.5,
#     ):
#         super().__init__()
#         self.do_distance = do_distance
#         self.distance: torch.Tensor
#         if self.do_distance:
#             positions = torch.arange(seq_len)
#             i = positions.unsqueeze(1)
#             j = positions.unsqueeze(0)
#             distance = (i - j).abs()
#             self.register_buffer("distance", distance)
#         self.exp_decay = exp_decay
#         self.alpha = alpha
#         self.eps = 1e-9
#
#     def forward(
#         self,
#         query_embs: torch.Tensor,
#         key_embs: torch.Tensor,
#         q_mask: torch.Tensor,
#         k_mask: torch.Tensor,
#         gumbel_temp: float,
#     ):
#         """
#         Args:
#             query_embs: (batch_size, 1, seq_len, hidden_size)  # (B, 1, S, H)
#             key_embs: (1, batch_size, seq_len, hidden_size)    # (1, B, S, H)
#             q_mask: (batch_size, 1, seq_len)                   # (B, 1, S)
#             k_mask: (1, batch_size, seq_len)                   # (1, B, S)
#         Returns:
#             scores: (batch_size, batch_size)                   # (B, B)
#         """
#         # Normalize embeddings
#         query_embs = F.normalize(query_embs, p=2, dim=-1)  # (B, 1, S, H)
#         key_embs = F.normalize(key_embs, p=2, dim=-1)  # (1, B, S, H)
#
#         # Compute token-level similarities for all (query, key) pairs
#         sim_matrix = torch.einsum(
#             "b q s h, c t u h -> b c s u", query_embs, key_embs
#         )  # (B, B, S, S)
#         # 'b q s h': (B, 1, S, H)
#         # 'c t u h': (1, B, S, H)
#         # -> 'b c s u': (B, B, S, S), where q=1 and t=1 broadcast
#
#         if self.do_distance:
#             if self.exp_decay:
#                 w = torch.exp(-self.alpha * self.distance)
#             else:
#                 w = 1.0 / (1.0 + self.distance)
#             # w: (S, S), broadcast to (B, B, S, S)
#             sim_matrix = sim_matrix * w
#
#         # Create combined mask for valid token pairs
#         valid_mask = torch.einsum(
#             "b q s, c t u -> b c s u", q_mask, k_mask
#         )  # (B, B, S, S)
#         # 'b q s': (B, 1, S)
#         # 'c t u': (1, B, S)
#         # -> 'b c s u': (B, B, S, S)
#
#         if self.training:
#             # Add Gumbel noise to valid positions only
#             uniform = torch.rand_like(sim_matrix) * (1 - self.eps) + self.eps
#             gumbel_noise = -torch.log(-torch.log(uniform))
#             noisy_sim = (sim_matrix + gumbel_noise * valid_mask) / gumbel_temp
#         else:
#             noisy_sim = sim_matrix / gumbel_temp  # (B, B, S, S)
#
#         # Mask invalid positions
#         noisy_sim = noisy_sim * valid_mask - (1 - valid_mask) * 1e9
#
#         # Compute attention distribution over key tokens
#         p_ij = F.softmax(noisy_sim, dim=-1)  # (B, B, S, S), softmax over 'u'
#
#         # Aggregate key embeddings
#         aggregated = torch.einsum(
#             "b c s u, c t u h -> b c s h", p_ij, key_embs
#         )  # (B, B, S, H)
#         # 'b c s u': (B, B, S, S)
#         # 'c t u h': (1, B, S, H), where t=1 broadcasts
#         # -> 'b c s h': (B, B, S, H)
#
#         # Compute scores per query token
#         scores = (query_embs * aggregated).sum(dim=-1)  # (B, B, S)
#         # query_embs: (B, 1, S, H) * aggregated: (B, B, S, H) -> (B, B, S, H)
#         # sum over h -> (B, B, S)
#
#         # Apply mask and aggregate over sequence
#         scores = scores * q_mask  # (B, B, S) * (B, 1, S) -> (B, B, S)
#         scores = scores.sum(dim=-1)  # (B, B)
#
#         # Normalize by number of valid query tokens
#         q_mask_sum = q_mask.sum(dim=-1).clamp(min=1)  # (B, 1)
#         scores = scores / q_mask_sum  # (B, B) / (B, 1) -> (B, B)
#
#         return scores

# def forward(
#     self,
#     query_embs: torch.Tensor,
#     key_embs: torch.Tensor,
#     q_mask: torch.Tensor,
#     k_mask: torch.Tensor,
#     gumbel_temp: float,
# ):
#     """
#     Args:
#         query_embs: (batch_size, seq_len, hidden_size)
#         key_embs: (batch_size, seq_len, hidden_size)
#         q_mask: (batch_size, seq_len)
#         k_mask: (batch_size, seq_len)
#     Returns:
#         scores: (batch_size,)
#     """
#     # batch_size, seq_len, hidden_size = query_embs.shape
#     # Normalize embeddings
#     query_embs = F.normalize(query_embs, p=2, dim=-1)
#     key_embs = F.normalize(key_embs, p=2, dim=-1)
#
#     # Compute token-level similarities
#     sim_matrix = torch.einsum("bqh,bkh->bqk", query_embs, key_embs)
#
#     if self.do_distance:
#         if self.exp_decay:
#             w = torch.exp(-self.alpha * self.distance)
#         else:
#             w = 1.0 / (1.0 + self.distance)
#
#         sim_matrix = sim_matrix * w
#
#     # Create combined mask for valid token pairs
#     valid_mask = torch.einsum("bq,bk->bqk", q_mask, k_mask)
#
#     if self.training:
#         # Add Gumbel noise to valid positions only
#         uniform = torch.rand_like(sim_matrix) * (1 - self.eps) + self.eps
#         gumbel_noise = -torch.log(-torch.log(uniform))
#         noisy_sim = (sim_matrix + gumbel_noise * valid_mask) / gumbel_temp
#     else:
#         noisy_sim = sim_matrix / gumbel_temp
#
#     # Mask invalid positions with large negative value
#     noisy_sim = noisy_sim * valid_mask - (1 - valid_mask) * 1e9
#
#     # Compute attention distribution
#     p_ij = F.softmax(noisy_sim, dim=-1)
#
#     # Aggregate key embeddings
#     aggregated = torch.einsum("bqk,bkh->bqh", p_ij, key_embs)
#
#     # Compute final scores (masked mean)
#     scores = (query_embs * aggregated).sum(dim=-1) * q_mask
#     scores = scores.sum(dim=-1) / q_mask.sum(dim=-1).clamp(min=1)
#     return scores
