# deep_stylometry/modules/info_nce_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_stylometry.modules.late_interaction import LateInteraction


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.late_interaction = LateInteraction()

    def forward(
        self,
        query_embs: torch.Tensor,
        key_embs: torch.Tensor,
        labels: torch.Tensor,
        q_mask: torch.Tensor,
        k_mask: torch.Tensor,
    ):
        """
        Args:
            query_embs: (batch_size, seq_len, hidden_size)
            key_embs: (batch_size, seq_len, hidden_size)
            labels: (batch_size,) - 1 for positive pairs, 0 for negatives
            q_mask: (batch_size, seq_len)
            k_mask: (batch_size, seq_len)
        """
        batch_size = query_embs.size(0)
        pos_mask = labels.bool()  # True where label=1

        # Compute all pairwise scores
        all_scores = (
            self.late_interaction(
                query_embs.unsqueeze(1),
                key_embs.unsqueeze(0),
                q_mask.unsqueeze(1),
                k_mask.unsqueeze(0),
            )
            / self.temperature
        )

        # For each query with label=1, its positive is the corresponding key
        pos_indices = torch.arange(batch_size, device=query_embs.device)[pos_mask]
        if pos_indices.numel() == 0:
            return torch.tensor(0.0, device=query_embs.device)  # No positive pairs

        filtered_scores = all_scores[pos_mask]
        filtered_labels = pos_indices
        return F.cross_entropy(filtered_scores, filtered_labels)
