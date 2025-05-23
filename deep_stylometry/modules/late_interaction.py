# deep_stylometry/modules/late_interaction.py

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LateInteraction(nn.Module):
    r"""Late interaction module for computing similarity scores between query
    and key embeddings. This module can use either distance-based weighting or
    cosine similarity for computing the similarity scores. The module also
    supports Gumbel softmax for sampling from the similarity scores.

    Parameters
    ----------
    do_distance: bool
        If True, use distance-based weighting for late interaction.
    exp_decay: bool
        If True, use exponential decay for the distance weights:
        $w = exp(-\alpha \cdot d)$, where $d$ is the distance between token positions.
        If False, use the formula $w = 1 / (1 + \alpha \cdot d)$.
    seq_len: int
        The maximum sequence length of the input sentences.
    alpha: float
        The alpha parameter for the distance weighting function.
    use_max: bool
        If True, use maximum cosine similarity for late interaction. If False, use Gumbel softmax.

    Attributes
    ----------
    alpha: float
        The alpha parameter for the distance weighting function, constrained to be positive.
    distance: torch.Tensor
        The distance matrix for computing distance-based weights.
    logit_scale: torch.nn.Parameter
        The learnable parameter for scaling the logits.
    exp_decay: bool
        If True, use exponential decay for the distance weights.
    do_distance: bool
        If True, use distance-based weighting for late interaction.
    use_max: bool
        If True, use maximum cosine similarity for late interaction. If False, use Gumbel softmax.
    """

    def __init__(
        self,
        do_distance: bool,
        exp_decay: bool,
        seq_len: int,
        alpha: float = 1.0,
        use_max: bool = False,
    ):
        super().__init__()
        self.do_distance = do_distance
        self.use_max = use_max
        if self.do_distance:
            self.distance: torch.Tensor
            self.alpha_raw = nn.Parameter(torch.tensor(alpha))
            positions = torch.arange(seq_len)
            distance = (positions.unsqueeze(1) - positions.unsqueeze(0)).abs().float()
            self.register_buffer("distance", distance)
        self.exp_decay = exp_decay
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0)))

    @property
    def alpha(self):
        """Softplus ensures alpha > 0."""
        return F.softplus(self.alpha_raw)

    def forward(
        self,
        query_embs: torch.Tensor,
        key_embs: torch.Tensor,
        q_mask: torch.Tensor,
        k_mask: torch.Tensor,
        gumbel_temp: Optional[float] = None,
    ):
        """Compute similarity scores between query and key embeddings using
        late interaction.

        Parameters
        ----------
        query_embs: torch.Tensor
            The query embeddings of shape (B, S, H), where B is the batch size,
            S is the sequence length, and H is the hidden size.
        key_embs: torch.Tensor
            The key embeddings of shape (B, S, H).
        q_mask: torch.Tensor
            The attention mask for the query embeddings of shape (B, S).
        k_mask: torch.Tensor
            The attention mask for the key embeddings of shape (B, S).
        gumbel_temp: Optional[float]
            The temperature for Gumbel softmax. If None, use softmax.

        Returns
        -------
        scores: torch.Tensor
            The computed similarity scores of shape (B, B) for each query-key pair.
        """
        # Normalize embeddings to preserve cosine similarity
        query_embs = query_embs.unsqueeze(1)  # (B, 1, S, H)
        query_embs = F.normalize(query_embs, p=2, dim=-1)
        q_mask = q_mask.unsqueeze(1)  # (B, 1, S)
        key_embs = key_embs.unsqueeze(0)  # (1, B, S, H)
        key_embs = F.normalize(key_embs, p=2, dim=-1)
        k_mask = k_mask.unsqueeze(0)  # (1, B, S)

        # Compute token-level cosine similarities
        sim_matrix = torch.einsum("insh, mjth->ijst", query_embs, key_embs)

        if self.do_distance:
            if self.exp_decay:
                w = torch.exp(-self.alpha * self.distance)
            else:
                w = 1.0 / (1.0 + self.alpha * self.distance)
            sim_matrix = sim_matrix * w.to(sim_matrix.device)

        # Compute valid mask for token pairs
        valid_mask = torch.einsum("ixs, xjt->ijst", q_mask, k_mask).bool()

        if self.use_max:  # Max-based interaction
            masked_sim = sim_matrix.masked_fill(~valid_mask, -float("inf"))
            max_sim_values, _ = masked_sim.max(dim=-1)  # (B, B, S)
            scores = (max_sim_values * q_mask.squeeze(1).unsqueeze(1)).sum(dim=-1)
            scores = scores / q_mask.squeeze(1).sum(dim=-1, keepdim=True).clamp(min=1)
            return scores

        # Scale similarities with learnable logit scale
        scale = torch.exp(self.logit_scale)
        logits = scale * sim_matrix
        logits = logits.masked_fill(
            ~valid_mask, -float("inf")
        )  # Mask invalid positions

        all_inf_slices = torch.all(logits == -float("inf"), dim=-1, keepdim=True)

        # For softmax calculation, replace all-inf slices with zeros. Softmax of zeros is uniform.
        # This prevents NaN from softmax itself.
        safe_logits_for_softmax = torch.where(
            all_inf_slices.expand_as(logits), torch.zeros_like(logits), logits
        )

        if gumbel_temp is not None:
            soft_p_ij = F.gumbel_softmax(
                safe_logits_for_softmax, tau=gumbel_temp, hard=False, dim=-1
            )
            if self.training:
                # use straight-through estimator during training
                # early training logits are more uniform
                hard_p_ij_temp = F.one_hot(
                    safe_logits_for_softmax.argmax(dim=-1), num_classes=logits.size(-1)
                ).float()
                p_ij_candidate = hard_p_ij_temp + (soft_p_ij - soft_p_ij.detach())
            else:
                # use softmax during evaluation
                p_ij_candidate = soft_p_ij
        else:
            p_ij_candidate = F.softmax(safe_logits_for_softmax, dim=-1)

        p_ij = torch.where(
            all_inf_slices.expand_as(p_ij_candidate),
            torch.zeros_like(p_ij_candidate),
            p_ij_candidate,
        )  # Aggregate key embeddings using attention weights
        key_embs_squeezed = key_embs.squeeze(0)  # (B, S, H)
        aggregated = torch.einsum("ijst,jth->ijsh", p_ij, key_embs_squeezed)

        # Compute final scores
        query_embs_expanded = query_embs.squeeze(1)  # (B, S, H)
        scores = (query_embs_expanded.unsqueeze(1) * aggregated).sum(dim=-1)
        scores = scores * q_mask.squeeze(1).unsqueeze(1)
        scores = scores.sum(dim=-1) / q_mask.squeeze(1).sum(dim=-1, keepdim=True).clamp(
            min=1
        )

        return scores
