# deep_stylometry/modules/info_nce_loss.py

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_stylometry.modules.late_interaction import LateInteraction

# deep_stylometry/modules/info_nce_loss.py


class InfoNCELoss(nn.Module):
    def __init__(
        self,
        do_late_interaction: bool,
        do_distance: bool,
        exp_decay: bool,
        seq_len: int,
        alpha: float = 0.5,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.temperature = temperature
        self.do_late_interaction = do_late_interaction
        self.do_distance = do_distance
        if self.do_late_interaction:
            self.late_interaction = LateInteraction(
                do_distance=do_distance,
                exp_decay=exp_decay,
                alpha=alpha,
                seq_len=seq_len,
            )
        # Add a learnable bias term to handle class imbalance
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        query_embs: torch.Tensor,
        key_embs: torch.Tensor,
        labels: torch.Tensor,  # (batch_size,) 0/1 indicating same author
        q_mask: torch.Tensor,
        k_mask: torch.Tensor,
        gumbel_temp: Optional[float] = None,
    ):
        batch_size = query_embs.size(0)

        # --- 1. Compute the (B, B) similarity matrix ---
        if self.do_late_interaction:
            # Ensure late_interaction returns scores of shape (B, B)
            all_scores = (
                self.late_interaction(
                    query_embs,
                    key_embs,  # (1, B, S, H)
                    q_mask,  # (B, 1, S)
                    k_mask,  # (1, B, S)
                    gumbel_temp,
                )
                / self.temperature
            )  # Apply temperature scaling AFTER late interaction
        else:
            # Mean pooling and normalization
            q_mask_sum = q_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
            query_vec = (query_embs * q_mask.unsqueeze(-1)).sum(dim=1) / q_mask_sum
            query_vec = F.normalize(query_vec, p=2, dim=-1)

            k_mask_sum = k_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
            key_vec = (key_embs * k_mask.unsqueeze(-1)).sum(dim=1) / k_mask_sum
            key_vec = F.normalize(key_vec, p=2, dim=-1)

            # Calculate cosine similarity matrix (B, B)
            all_scores = torch.matmul(query_vec, key_vec.T) / self.temperature

        # --- 2. Identify positive pairs ---
        # labels are 1 if (q_i, k_i) is a positive pair, 0 otherwise
        pos_mask = labels.bool()  # Shape (B,)

        # Early exit if no positive pairs in the batch
        if pos_mask.sum() == 0:
            return torch.tensor(
                0.0, device=query_embs.device, requires_grad=True
            )  # Ensure requires_grad if needed

        # --- 3. Prepare for CrossEntropyLoss ---
        # Select rows corresponding to positive queries
        pos_query_scores = all_scores[pos_mask]  # Shape (num_pos, B)

        # Targets: For a positive query i, the positive key is at index i
        pos_query_targets = torch.arange(batch_size, device=all_scores.device)[
            pos_mask
        ]  # Shape (num_pos,)

        # --- 4. Compute InfoNCE loss ---
        loss = F.cross_entropy(pos_query_scores, pos_query_targets, reduction="mean")

        return loss

    # def forward(
    #     self,
    #     query_embs: torch.Tensor,
    #     key_embs: torch.Tensor,
    #     labels: torch.Tensor,  # (batch_size,) 0/1 indicating same author
    #     q_mask: torch.Tensor,
    #     k_mask: torch.Tensor,
    #     gumbel_temp: Optional[float] = None,
    # ):
    #     batch_size = query_embs.size(0)
    #
    #     # Compute similarity scores
    #     if self.do_late_interaction:
    #         all_scores = self.late_interaction(
    #             query_embs.unsqueeze(1),
    #             key_embs.unsqueeze(0),
    #             q_mask.unsqueeze(1),
    #             k_mask.unsqueeze(0),
    #             gumbel_temp,
    #         )
    #     else:
    #         # Mean pooling and normalization
    #         q_mask_sum = q_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
    #         query_vec = (query_embs * q_mask.unsqueeze(-1)).sum(dim=1) / q_mask_sum
    #         query_vec = F.normalize(query_vec, p=2, dim=-1)
    #
    #         k_mask_sum = k_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
    #         key_vec = (key_embs * k_mask.unsqueeze(-1)).sum(dim=1) / k_mask_sum
    #         key_vec = F.normalize(key_vec, p=2, dim=-1)
    #
    #         all_scores = torch.matmul(query_vec, key_vec.T) / self.temperature
    #
    #     # Extract diagonal scores (i,i pairs)
    #     diag_scores = all_scores.diag().view(-1, 1)  # (batch_size, 1)
    #
    #     # Create targets: 1 for positives (label=1), 0 for negatives (label=0)
    #     targets = labels.float().view(-1, 1)  # (batch_size, 1)
    #
    #     # Compute binary cross-entropy loss with logits
    #     loss = F.binary_cross_entropy_with_logits(
    #         diag_scores + self.bias,  # Add bias to handle class imbalance
    #         targets,
    #         reduction="mean",
    #     )
    #
    #     return loss


# class InfoNCELoss(nn.Module):
#     def __init__(
#         self,
#         do_late_interaction: bool,
#         do_distance: bool,
#         exp_decay: bool,
#         seq_len: int,
#         alpha: float = 0.5,
#         temperature: float = 0.07,
#     ):
#         super().__init__()
#         self.temperature = temperature
#         self.do_late_interaction = do_late_interaction
#         self.do_distance = do_distance
#         if self.do_late_interaction:
#             self.late_interaction = LateInteraction(
#                 do_distance=do_distance,
#                 exp_decay=exp_decay,
#                 alpha=alpha,
#                 seq_len=seq_len,
#             )
#
#     def forward(
#         self,
#         query_embs: torch.Tensor,
#         key_embs: torch.Tensor,
#         labels: torch.Tensor,  # Binary tensor: 1 for same author, 0 for different
#         q_mask: torch.Tensor,
#         k_mask: torch.Tensor,
#         gumbel_temp: Optional[float] = None,
#     ):
#         batch_size = query_embs.size(0)
#
#         # Compute similarity matrix
#         if self.do_late_interaction:
#             sim_matrix = (
#                 self.late_interaction(
#                     query_embs.unsqueeze(1),  # (B, 1, S, H)
#                     key_embs.unsqueeze(0),  # (1, B, S, H)
#                     q_mask.unsqueeze(1),  # (B, 1, S)
#                     k_mask.unsqueeze(0),  # (1, B, S)
#                     gumbel_temp,
#                 )
#                 / self.temperature
#             )
#         else:
#             # Mean pooling and normalization
#             q_mask_sum = q_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
#             query_vec = (query_embs * q_mask.unsqueeze(-1)).sum(dim=1) / q_mask_sum
#             query_vec = F.normalize(query_vec, p=2, dim=-1)
#
#             k_mask_sum = k_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
#             key_vec = (key_embs * k_mask.unsqueeze(-1)).sum(dim=1) / k_mask_sum
#             key_vec = F.normalize(key_vec, p=2, dim=-1)
#
#             sim_matrix = torch.matmul(query_vec, key_vec.T) / self.temperature
#
#         # Scale similarities to [0,1] range for BCE loss
#         # Using sigmoid or tanh to scale
#         probs = torch.sigmoid(sim_matrix)
#
#         # Ensure labels are in the correct shape
#         labels = labels.view(batch_size, 1)  # (B, 1)
#         # Create full label matrix - ones on diagonal where labels==1, zeros elsewhere
#         label_matrix = torch.zeros_like(probs)
#         for i in range(batch_size):
#             if labels[i] == 1:
#                 label_matrix[i, i] = 1.0
#
#         # Apply BCE loss
#         loss = F.binary_cross_entropy(probs, label_matrix, reduction="mean")
#
#         return loss

# def forward(
#     self,
#     query_embs: torch.Tensor,
#     key_embs: torch.Tensor,
#     labels: torch.Tensor,
#     q_mask: torch.Tensor,
#     k_mask: torch.Tensor,
#     gumbel_temp: Optional[float] = None,
# ):
#     batch_size = query_embs.size(0)
#
#     # Ensure labels are 1D and boolean
#     pos_mask = labels.bool().squeeze(-1) if labels.dim() > 1 else labels.bool()
#     assert pos_mask.dim() == 1, f"pos_mask must be 1D, got {pos_mask.shape}"
#
#     # Early exit if no positives
#     num_pos = pos_mask.sum()
#     if num_pos == 0:
#         return torch.tensor(0.0, device=query_embs.device)
#
#     # Compute scores matrix
#     if self.do_late_interaction:
#         all_scores = (
#             self.late_interaction(
#                 query_embs.unsqueeze(1),  # (B, 1, S, H)
#                 key_embs.unsqueeze(0),  # (1, B, S, H)
#                 q_mask.unsqueeze(1),  # (B, 1, S)
#                 k_mask.unsqueeze(0),  # (1, B, S)
#                 gumbel_temp,
#             )
#             / self.temperature
#         )
#     else:
#         # Existing mean pooling logic
#         # Mean pooling and normalization
#         q_mask_sum = q_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
#         query_vec = (query_embs * q_mask.unsqueeze(-1)).sum(dim=1) / q_mask_sum
#         query_vec = F.normalize(query_vec, p=2, dim=-1)
#
#         k_mask_sum = k_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
#         key_vec = (key_embs * k_mask.unsqueeze(-1)).sum(dim=1) / k_mask_sum
#         key_vec = F.normalize(key_vec, p=2, dim=-1)
#
#         all_scores = torch.matmul(query_vec, key_vec.T) / self.temperature
#
#     # Validate scores matrix shape
#     assert all_scores.shape == (
#         batch_size,
#         batch_size,
#     ), f"Scores matrix must be (B, B), got {all_scores.shape}"
#
#     # Filter valid positive queries and their targets
#     pos_scores = all_scores[pos_mask]  # (num_pos, B)
#     pos_targets = torch.arange(batch_size, device=query_embs.device)[pos_mask]
#
#     # Ensure targets are within valid range
#     if (pos_targets >= batch_size).any():
#         invalid_targets = pos_targets[pos_targets >= batch_size]
#         raise ValueError(
#             f"Targets {invalid_targets.tolist()} exceed batch size {batch_size}"
#         )
#
#     # Compute contrastive loss
#     loss = F.cross_entropy(pos_scores, pos_targets, reduction="mean")
#     return loss

# def forward(
#     self,
#     query_embs: torch.Tensor,
#     key_embs: torch.Tensor,
#     labels: torch.Tensor,
#     q_mask: torch.Tensor,
#     k_mask: torch.Tensor,
#     gumbel_temp: Optional[float] = None,
# ):
#     batch_size = query_embs.size(0)
#     pos_mask = labels.bool()  # Shape: (batch_size,), True where labels == 1
#
#     # Compute all pairwise scores (query x key)
#     if self.do_late_interaction:
#         all_scores = (
#             self.late_interaction(
#                 query_embs.unsqueeze(1),  # (batch_size, 1, seq_len, hidden_size)
#                 key_embs.unsqueeze(0),  # (1, batch_size, seq_len, hidden_size)
#                 q_mask.unsqueeze(1),
#                 k_mask.unsqueeze(0),
#                 gumbel_temp=gumbel_temp,
#             )
#             / self.temperature
#         )
#     else:
#         # Mean pooling and normalization
#         q_mask_sum = q_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
#         query_vec = (query_embs * q_mask.unsqueeze(-1)).sum(dim=1) / q_mask_sum
#         query_vec = F.normalize(query_vec, p=2, dim=-1)
#
#         k_mask_sum = k_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
#         key_vec = (key_embs * k_mask.unsqueeze(-1)).sum(dim=1) / k_mask_sum
#         key_vec = F.normalize(key_vec, p=2, dim=-1)
#
#         all_scores = torch.matmul(query_vec, key_vec.T) / self.temperature
#
#     # Only compute loss for positive pairs
#     # pos_indices = torch.arange(
#     #     batch_size, device=query_embs.device
#     # )  # Shape: (batch_size,)
#     # pos_scores = all_scores[pos_mask]  # Select rows where labels == 1
#     # pos_targets = pos_indices[pos_mask]  # Select corresponding targets
#
#     pos_scores = all_scores[pos_mask]  # Shape: (n_pos, batch_size)
#     pos_targets = torch.arange(pos_scores.size(0), device=query_embs.device)
#
#     # Compute cross-entropy loss only if there are positive pairs
#     if pos_scores.size(0) > 0:
#         loss = F.cross_entropy(pos_scores, pos_targets, reduction="mean")
#     else:
#         loss = torch.tensor(0.0, device=query_embs.device)
#
#     return loss

# def forward(
#     self,
#     query_embs: torch.Tensor,
#     key_embs: torch.Tensor,
#     labels: torch.Tensor,
#     q_mask: torch.Tensor,
#     k_mask: torch.Tensor,
#     gumbel_temp: Optional[float] = None,
# ):
#     batch_size = query_embs.size(0)
#     pos_mask = labels.bool()  # (B,)
#     num_pos = pos_mask.sum()
#
#     # Early exit if no positive pairs
#     if num_pos == 0:
#         return torch.tensor(0.0, device=query_embs.device)
#
#     # Compute scores for all pairs
#     if self.do_late_interaction:
#         all_scores = (
#             self.late_interaction(
#                 query_embs.unsqueeze(1),
#                 key_embs.unsqueeze(0),
#                 q_mask.unsqueeze(1),
#                 k_mask.unsqueeze(0),
#                 gumbel_temp,
#             )
#             / self.temperature
#         )
#     else:
#         # Mean pooling and normalization
#         q_mask_sum = q_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
#         query_vec = (query_embs * q_mask.unsqueeze(-1)).sum(dim=1) / q_mask_sum
#         query_vec = F.normalize(query_vec, p=2, dim=-1)
#
#         k_mask_sum = k_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
#         key_vec = (key_embs * k_mask.unsqueeze(-1)).sum(dim=1) / k_mask_sum
#         key_vec = F.normalize(key_vec, p=2, dim=-1)
#
#         all_scores = torch.matmul(query_vec, key_vec.T) / self.temperature
#     # Select only positive queries and their targets
#     all_scores_pos = all_scores[pos_mask]  # (num_pos, B)
#     pos_indices = torch.arange(batch_size, device=query_embs.device)[pos_mask]
#
#     # Compute cross_entropy loss for valid positives
#     loss = F.cross_entropy(all_scores_pos, pos_indices, reduction="mean")
#     return loss

# def forward(
#     self,
#     query_embs: torch.Tensor,
#     key_embs: torch.Tensor,
#     labels: torch.Tensor,
#     q_mask: torch.Tensor,
#     k_mask: torch.Tensor,
#     gumbel_temp: Optional[float] = None,
# ):
#     batch_size = query_embs.size(0)
#     pos_mask = labels.bool()
#
#     # Compute all pairwise scores (query x key)
#     if self.do_late_interaction:
#         all_scores = (
#             self.late_interaction(
#                 query_embs.unsqueeze(1),  # (B, 1, S, H)
#                 key_embs.unsqueeze(0),  # (1, B, S, H)
#                 q_mask.unsqueeze(1),
#                 k_mask.unsqueeze(0),
#                 gumbel_temp=gumbel_temp,
#             )
#             / self.temperature
#         )
#     else:
#         # Mean pooling and normalization
#         q_mask_sum = q_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
#         query_vec = (query_embs * q_mask.unsqueeze(-1)).sum(dim=1) / q_mask_sum
#         query_vec = F.normalize(query_vec, p=2, dim=-1)
#
#         k_mask_sum = k_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
#         key_vec = (key_embs * k_mask.unsqueeze(-1)).sum(dim=1) / k_mask_sum
#         key_vec = F.normalize(key_vec, p=2, dim=-1)
#
#         all_scores = torch.matmul(query_vec, key_vec.T) / self.temperature
#
#     # For each positive pair (query, key), mask out the positive key in the denominator
#     # and include all other keys (in-batch + hard negatives)
#     pos_indices = torch.arange(batch_size, device=query_embs.device)
#     loss = F.cross_entropy(
#         all_scores,
#         pos_indices,
#         reduction="none",
#     )
#
#     # Apply mask to only compute loss for positive pairs
#     loss = loss[pos_mask].mean()
#
#     return loss if not torch.isnan(loss) else torch.tensor(0.0, device=loss.device)
