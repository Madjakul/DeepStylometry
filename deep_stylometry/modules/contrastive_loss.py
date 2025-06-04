# deep_stylometry/modules/ccontrastive_loss.py

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_stylometry.modules.late_interaction import LateInteraction


class ContrastiveLoss(nn.Module):

    def __init__(
        self,
        do_late_interaction: bool = False,
        margin: float = 0.5,
        temperature: float = 1.0,
        do_distance: bool = False,
        exp_decay: bool = False,
        seq_len: int = 512,
        use_max: bool = True,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        self.do_late_interaction = do_late_interaction
        if self.do_late_interaction:
            self.late_interaction = LateInteraction(
                do_distance=do_distance,
                exp_decay=exp_decay,
                alpha=alpha,
                seq_len=seq_len,
                use_max=use_max,
            )

    def forward(
        self,
        query_embs: torch.Tensor,
        key_embs: torch.Tensor,
        labels: torch.Tensor,
        q_mask: torch.Tensor,
        k_mask: torch.Tensor,
        gumbel_temp: Optional[float] = None,
    ):
        batch_size = query_embs.size(0)

        # --- 1. Compute the (B, B) similarity matrix ---
        if self.do_late_interaction:
            all_scores = (
                self.late_interaction(
                    query_embs=query_embs,
                    key_embs=key_embs,  # (1, B, S, H)
                    q_mask=q_mask,  # (B, 1, S)
                    k_mask=k_mask,  # (1, B, S)
                    gumbel_temp=gumbel_temp,
                )
                / self.temperature
            )
        else:
            # --- 1. Compute sentence embeddings (mean pooling and normalization) ---
            # Shape (B, 1)
            q_len = q_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
            # Shape (B, H)
            query_vec = (query_embs * q_mask.unsqueeze(-1)).sum(dim=1) / q_len
            # Shape (B, H)
            query_vec = F.normalize(query_vec, p=2, dim=-1)
            k_len = k_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
            key_vec = (key_embs * k_mask.unsqueeze(-1)).sum(dim=1) / k_len
            key_vec = F.normalize(key_vec, p=2, dim=-1)

            # --- 2. Compute the (B, B) raw cosine similarity matrix ---
            all_scores = torch.matmul(query_vec, key_vec.T)  # Shape (B, B)

        # --- 3. Compute Contrastive Loss on the diagonal elements (q_i, k_i) ---
        diag_cos_sim = torch.diag(all_scores)  # Shape (B,)

        # Compute squared Euclidean distances from cosine similarities for these diag
        # dist^2 = 2 - 2 * cos_sim (since vectors are normalized)
        sq_dist_diag = torch.clamp(2.0 - 2.0 * diag_cos_sim, min=0.0)  # Shape (B,)
        dist_diag = torch.sqrt(sq_dist_diag)  # Shape (B,)

        # Ensure labels are float for arithmetic operations in the loss
        labels_float = labels.to(dist_diag.dtype)

        # If labels_float[i]=1 (positive pair): loss_pair = sq_dist_diag[i]
        # If labels_float[i]=0 (negative pair): loss_pair = max(0, margin - dist_diag[i])^2
        loss_pos_terms = labels_float * sq_dist_diag

        neg_margin_terms = torch.relu(self.margin - dist_diag)
        loss_neg_terms = (1.0 - labels_float) * (neg_margin_terms**2)

        # Combine positive and negative loss terms for each diagonal pair
        individual_losses = 0.5 * (loss_pos_terms + loss_neg_terms)
        contrastive_loss = individual_losses.mean()  # Mean over the batch

        # --- 4. Prepare outputs for metrics (similar to InfoNCELoss structure) ---
        # Scale the full raw similarity matrix by temperature for output.
        # This is for consistency if downstream metrics expect temperature-scaled scores (e.g., for softmax).
        all_scores_output = all_scores / self.temperature

        # Identify "positive queries" using the 'labels' tensor.
        # A query q_i is considered for retrieval metrics if labels[i] is 1 (meaning k_i is its positive).
        pos_mask = labels.bool()  # Shape (B,)

        if pos_mask.sum() == 0:
            # If no positive queries are indicated by 'labels', return empty tensors for these.
            pos_query_scores_output = torch.empty(
                0, batch_size, device=query_embs.device
            )
            pos_query_targets = torch.empty(
                0, dtype=torch.long, device=query_embs.device
            )
        else:
            # Select rows from the temperature-scaled scores corresponding to positive queries
            pos_query_scores_output = all_scores_output[pos_mask]  # Shape (num_pos, B)

            # Targets: For a positive query i (where labels[i]=1), the positive key is k_i (index i)
            # Shape (num_pos,)
            pos_query_targets = torch.arange(
                batch_size, device=all_scores_output.device
            )[pos_mask]

        return (
            all_scores_output,
            pos_query_scores_output,
            pos_query_targets,
            contrastive_loss,
        )
