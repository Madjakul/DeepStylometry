# deep_stylometry/modules/late_interaction.py

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LateInteraction(nn.Module):

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
            self.alpha_raw = nn.Parameter(torch.tensor(alpha).unsqueeze(0))
            positions = torch.arange(seq_len)
            distance = (positions.unsqueeze(1) - positions.unsqueeze(0)).abs().float()
            self.register_buffer("distance", distance)
        self.exp_decay = exp_decay
        # TODO: test other ways of scaling that does not require exp as it saturates the grad
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(10.0)).unsqueeze(0))

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
            # TODO: try the other distance weighting as it can be more stabe than exp decay when differentiated
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
        # TODO: remove this exp and the log in the init
        scale = torch.exp(self.logit_scale)
        logits = scale * sim_matrix

        # Mask invalid positions with a large negative value (not -inf)
        neg_inf = -1e9
        logits = torch.where(valid_mask, logits, neg_inf)

        # Add a tiny ridge term so no row is entirely flat
        eps = 1e-6
        logits = logits + eps

        if gumbel_temp is not None and self.training:
            # TODO: test the stability with on STE
            # --- DURING TRAINING ---
            # Use the soft, differentiable Gumbel-softmax probabilities directly.
            p_ij_candidate = F.gumbel_softmax(
                logits, tau=gumbel_temp, hard=False, dim=-1
            )
        elif not self.training and gumbel_temp is not None:
            # --- DURING EVALUATION (OPTIONAL) ---
            # For deterministic output, use the hard argmax. No gradients needed here.
            p_ij_candidate = F.one_hot(
                logits.argmax(dim=-1), num_classes=logits.size(-1)
            ).float()
        else:
            # Fallback to standard softmax if no Gumbel temperature is provided
            p_ij_candidate = F.softmax(logits, dim=-1)

        p_ij = p_ij_candidate

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
