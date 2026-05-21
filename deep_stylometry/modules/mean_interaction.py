# deep_stylometry/modules/mean_interaction.py

import logging

import torch

logger = logging.getLogger(__name__)
import torch.nn.functional as F
from jaxtyping import Float, Int


class MeanInteraction(torch.nn.Module):
    """Mean-pooling baseline with cosine similarity scoring.

    Masks-aware mean pooling collapses the sequence dimension; the resulting
    vectors are L2-normalised before computing a dot-product similarity matrix.
    """

    def __init__(self) -> None:
        super().__init__()
        logger.info("Using mean pooling and cosine similarity")

    def forward(
        self,
        query_embs: Float[torch.Tensor, "batch seq hidden"],
        key_embs: Float[torch.Tensor, "two_times_batch seq hidden"],
        q_mask: Int[torch.Tensor, "batch seq"],
        k_mask: Int[torch.Tensor, "two_times_batch seq"],
        **kwargs,
    ) -> Float[torch.Tensor, "batch two_times_batch"]:
        """Compute masked mean-pooled cosine similarity scores.

        Parameters
        ----------
        query_embs : Float[Tensor, "batch seq hidden"]
            Per-token query embeddings.
        key_embs : Float[Tensor, "two_times_batch seq hidden"]
            Per-token key embeddings (positives followed by negatives).
        q_mask : Int[Tensor, "batch seq"]
            Attention mask for queries.
        k_mask : Int[Tensor, "two_times_batch seq"]
            Attention mask for keys.

        Returns
        -------
        Float[Tensor, "batch two_times_batch"]
            Pairwise cosine similarity matrix.
        """
        query_vec = (query_embs * q_mask.unsqueeze(-1)).sum(dim=1)
        query_vec = F.normalize(query_vec, p=2, dim=-1)

        key_vec = (key_embs * k_mask.unsqueeze(-1)).sum(dim=1)
        key_vec = F.normalize(key_vec, p=2, dim=-1)

        all_scores = torch.matmul(query_vec, key_vec.T)
        return all_scores
