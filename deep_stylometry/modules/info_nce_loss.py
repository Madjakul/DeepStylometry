# deep_stylometry/modules/info_nce_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_stylometry.modules.late_interaction import LateInteraction


class InfoNCELoss(nn.Module):
    def __init__(
        self,
        do_late_interaction: bool,
        do_distance_based: bool,
        exp_decay: bool,
        alpha: float = 0.5,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.temperature = temperature
        self.do_late_interaction = do_late_interaction
        self.do_distance_based = do_distance_based
        if self.do_late_interaction:
            self.late_interaction = LateInteraction(
                do_distance_based=do_distance_based, exp_decay=exp_decay, alpha=alpha
            )

    def forward(
        self,
        query_embs: torch.Tensor,
        key_embs: torch.Tensor,
        labels: torch.Tensor,
        q_mask: torch.Tensor,
        k_mask: torch.Tensor,
    ):
        batch_size = query_embs.size(0)
        pos_mask = labels.bool()  # True where label=1

        # Compute all pairwise scores
        if self.do_late_interaction:
            all_scores = (
                self.late_interaction(
                    query_embs.unsqueeze(1),
                    key_embs.unsqueeze(0),
                    q_mask.unsqueeze(1),
                    k_mask.unsqueeze(0),
                )
                / self.temperature
            )
        else:
            # TODO: cosine similarities (normalized dot products) between query and key embeddings
            pass

        # For each query with label=1, its positive is the corresponding key
        pos_indices = torch.arange(batch_size, device=query_embs.device)[pos_mask]
        if pos_indices.numel() == 0:
            return torch.tensor(0.0, device=query_embs.device)  # No positive pairs

        filtered_scores = all_scores[pos_mask]
        filtered_labels = pos_indices
        return F.cross_entropy(filtered_scores, filtered_labels)
