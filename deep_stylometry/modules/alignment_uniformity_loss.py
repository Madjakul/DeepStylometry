# deep_stylometry/modules/alignment_uniformity_loss.py

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int


class AlignmentUniformityLoss(nn.Module):
    """Alignment and uniformity metrics for embedding quality monitoring.

    Computes the two diagnostic losses from Wang & Isola (2020):
    *alignment* measures how close positive pairs are, and *uniformity*
    measures how evenly embeddings are spread on the hypersphere.  Both are
    computed in inference mode and logged during validation.
    """

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
        """Compute alignment and uniformity losses.

        Parameters
        ----------
        query_embs : Float[Tensor, "batch seq hidden"]
            Per-token query embeddings.
        key_embs : Float[Tensor, "n_keys seq hidden"]
            Per-token key embeddings.
        q_mask : Int[Tensor, "batch seq"]
            Attention mask for queries.
        k_mask : Int[Tensor, "n_keys seq"]
            Attention mask for keys.
        targets : Int[Tensor, "batch"]
            Index into ``key_embs`` of each query's positive.

        Returns
        -------
        dict
            Keys: ``alignment_loss`` (scalar) and ``uniformity_loss`` (scalar).
        """

        # Get mean-pooled representations for each sequence
        q_lengths = q_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        k_lengths = k_mask.sum(dim=-1, keepdim=True).clamp(min=1)

        # Mean pooling
        query_pooled = (query_embs * q_mask.unsqueeze(-1)).sum(dim=1) / q_lengths
        key_pooled = (key_embs * k_mask.unsqueeze(-1)).sum(dim=1) / k_lengths

        query_pooled = F.normalize(query_pooled, p=2, dim=-1)
        key_pooled = F.normalize(key_pooled, p=2, dim=-1)

        positive_keys = key_pooled[targets]

        alignment_loss = (
            torch.norm(query_pooled - positive_keys, p=2, dim=-1).pow(2).mean()
        )

        all_embeddings = torch.cat([query_pooled, key_pooled], dim=0)
        pairwise_dists = torch.cdist(all_embeddings, all_embeddings, p=2).pow(2)

        mask = ~torch.eye(
            pairwise_dists.size(0), dtype=torch.bool, device=pairwise_dists.device
        )
        valid_dists = pairwise_dists[mask]

        uniformity_loss = torch.logsumexp(-2 * valid_dists, dim=0) - torch.log(
            torch.tensor(valid_dists.size(0), device=valid_dists.device)
        )

        return {"alignment_loss": alignment_loss, "uniformity_loss": uniformity_loss}
