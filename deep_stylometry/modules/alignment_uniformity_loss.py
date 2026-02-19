# deep_stylometry/modules/alignment_uniformity_loss.py

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int


class AlignmentUniformityLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @torch.inference_mode()
    def forward(
        self,
        query_embs: Float[torch.Tensor, "batch seq hidden"],
        key_embs: Float[torch.Tensor, "n_keys seq hidden"],
        q_mask: Int[torch.Tensor, "batch seq"],
        k_mask: Int[torch.Tensor, "n_keys seq"],
        targets: Int[torch.Tensor, "batch"],
    ) -> Dict[str, torch.Tensor]:

        # Get mean-pooled representations for each sequence
        q_lengths = q_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        k_lengths = k_mask.sum(dim=-1, keepdim=True).clamp(min=1)

        # Mean pooling
        query_pooled = (query_embs * q_mask.unsqueeze(-1)).sum(dim=1) / q_lengths
        key_pooled = (key_embs * k_mask.unsqueeze(-1)).sum(dim=1) / k_lengths

        # L2 normalize the pooled representations
        query_pooled = F.normalize(query_pooled, p=2, dim=-1)
        key_pooled = F.normalize(key_pooled, p=2, dim=-1)

        positive_keys = key_pooled[targets]

        # Compute squared L2 distance between positive pairs
        alignment_loss = (
            torch.norm(query_pooled - positive_keys, p=2, dim=-1).pow(2).mean()
        )

        # Combine all embeddings for uniformity calculation
        all_embeddings = torch.cat([query_pooled, key_pooled], dim=0)
        pairwise_dists = torch.cdist(all_embeddings, all_embeddings, p=2).pow(2)

        # Apply the log uniformity formula
        mask = ~torch.eye(
            pairwise_dists.size(0), dtype=torch.bool, device=pairwise_dists.device
        )
        valid_dists = pairwise_dists[mask]

        uniformity_loss = torch.logsumexp(-2 * valid_dists, dim=0) - torch.log(
            torch.tensor(valid_dists.size(0), device=valid_dists.device)
        )

        return {"alignment_loss": alignment_loss, "uniformity_loss": uniformity_loss}
