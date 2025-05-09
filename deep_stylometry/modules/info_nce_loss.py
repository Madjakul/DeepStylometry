# deep_stylometry/modules/info_nce_loss.py

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_stylometry.modules.late_interaction import LateInteraction


class InfoNCELoss(nn.Module):
    """Compute the InfoNCE loss for a batch of query-key sentences pairs. The
    distance between the query and key sentences is computed using cosine
    similarity over the average embeddings of the sentences or using late
    interaction. The loss is computed using cross-entropy loss.

    Parameters
    ----------
    do_late_interaction: bool
        If True, use late interaction to compute the similarity scores.
    do_distance: bool
        Only if do_late_interaction is True. If True, use distance-based late interaction.
        Each query token position is weighted by the distance to the key token position.
    exp_decay: bool
        Only if do_late_interaction is True. If True, use exponential decay for the distance
        weights.
    seq_len: int
        The maximum sequence length of the input sentences.
    use_max: bool
        If True, use maximum cosine similarity for late interaction. If False, use Gumbel softmax.
    alpha: float
        The alpha parameter for the exponential decay function.
    temperature: float
        The temperature parameter for scaling the similarity scores.

    Attributes
    ----------
    temperature: float
        The temperature parameter for scaling the similarity scores.
    do_late_interaction: bool
        If True, use late interaction to compute the similarity scores.
    do_distance: bool
        If True, use distance-based late interaction.
    late_interaction: LateInteraction
        The LateInteraction module used for computing similarity scores.
    """

    def __init__(
        self,
        do_late_interaction: bool,
        do_distance: bool,
        exp_decay: bool,
        seq_len: int,
        use_max: bool = True,
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
        """Compute the InfoNCE loss for a batch of query-key sentence pairs.

        Parameters
        ----------
        query_embs: torch.Tensor
            The embeddings of the query sentences. Shape (B, S, H).
        key_embs: torch.Tensor
            The embeddings of the key sentences. Shape (1, B, S, H).
        labels: torch.Tensor
            The labels indicating the positive query-key pairs. Shape (B,).
        q_mask: torch.Tensor
            The attention mask for the query sentences. Shape (B, S).
        k_mask: torch.Tensor
            The attention mask for the key sentences. Shape (1, B, S).
        gumbel_temp: Optional[float]
            The temperature parameter for the Gumbel softmax. Only used if
            do_late_interaction is True and use_max is False.

        Returns
        -------
        pos_query_scores: torch.Tensor
            The similarity scores for the positive query-key pairs. Shape (num_pos, B).
        pos_query_targets: torch.Tensor
            The target indices for the positive query-key pairs. Shape (num_pos,).
        contrastive_loss: torch.Tensor
            The computed InfoNCE loss. Shape (1,).
        """
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
            return torch.tensor(0.0, device=query_embs.device)

        # --- 3. Prepare for CrossEntropyLoss ---
        # Select rows corresponding to positive queries
        pos_query_scores = all_scores[pos_mask]  # Shape (num_pos, B)

        # Targets: For a positive query i, the positive key is at index i
        pos_query_targets = torch.arange(batch_size, device=all_scores.device)[
            pos_mask
        ]  # Shape (num_pos,)

        # --- 4. Compute InfoNCE loss ---
        contrastive_loss = F.cross_entropy(
            pos_query_scores, pos_query_targets, reduction="mean"
        )

        return pos_query_scores, pos_query_targets, contrastive_loss
