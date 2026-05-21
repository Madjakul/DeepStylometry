# deep_stylometry/modules/mean_centerer.py
"""Running mean centering with L2 normalisation for embedding regularisation.

Adapted from IDIOLEX (Kantharuban et al., 2026). Maintains a Welford-style
online running mean across training batches and subtracts it before L2
normalisation. Single-GPU only (no DDP all_reduce).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float


class MeanCenterer(nn.Module):
    """Online running-mean centerer with L2 normalisation.

    The running mean is updated during training via :meth:`update` and
    subtracted from embeddings (then L2-normalised) via :meth:`apply`.
    Both buffers (``mu`` and ``n``) are saved in the model checkpoint.

    Args:
        dim: Dimensionality of the embeddings.
        dtype: Data type for the running mean buffer (default ``float32``).
    """

    def __init__(self, dim: int, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.register_buffer("mu", torch.zeros(dim, dtype=dtype))
        self.register_buffer("n", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update(
        self, embeddings: Float[torch.Tensor, "batch hidden"]
    ) -> None:
        """Update the running mean with a new batch of embeddings.

        Uses a Welford-style online update so that the mean converges to the
        global mean across all training batches without storing all data.

        Args:
            embeddings: Batch of pooled embeddings, shape ``(batch, hidden)``.
                If the batch is empty, this is a no-op.
        """
        if embeddings.size(0) == 0:
            return
        batch_mean = embeddings.detach().float().mean(dim=0)
        count = torch.tensor(
            embeddings.size(0), device=embeddings.device, dtype=torch.long
        )
        weight = count.float() / (self.n.float() + count.float())
        self.mu.add_((batch_mean - self.mu) * weight)
        self.n.add_(count)

    def apply(  # type: ignore[override]
        self, embeddings: Float[torch.Tensor, "batch hidden"]
    ) -> Float[torch.Tensor, "batch hidden"]:
        """Centre and L2-normalise a batch of embeddings.

        Subtracts the running mean (``mu``) and applies L2 normalisation.
        Before any :meth:`update` calls, ``mu`` is zero so this reduces to
        plain L2 normalisation.

        Args:
            embeddings: Batch of pooled embeddings, shape ``(batch, hidden)``.

        Returns:
            Centred and L2-normalised embeddings of the same shape.
        """
        centred = embeddings - self.mu.to(embeddings.dtype)
        return F.normalize(centred, p=2, dim=-1)
