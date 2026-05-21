# deep_stylometry/modules/patch_boundary_predictor.py

import logging
import math
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int

if TYPE_CHECKING:
    from deep_stylometry.utils.configs import BaseConfig

logger = logging.getLogger(__name__)


class PatchBoundaryPredictor(nn.Module):
    """Predicts patch boundary probabilities from contextualised token
    representations.  Uses Gumbel-Softmax for differentiable discrete
    boundary sampling during training and hard thresholding during inference.

    The first valid token per example is always treated as a boundary (start
    of patch 0).  During training the annealed temperature drives the
    Gumbel samples towards hard 0/1 decisions; during inference a fixed
    threshold of 0.5 is used.
    """

    def __init__(self, hidden_size: int, cfg: "BaseConfig") -> None:
        super().__init__()
        self.cfg = cfg
        mid = max(16, hidden_size // 4)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, mid),
            nn.GELU(),
            nn.Linear(mid, 1),
        )
        logger.info(
            f"PatchBoundaryPredictor: hidden={hidden_size}, mid={mid}, "
            f"tau_init={cfg.model.gumbel_tau_init}, "
            f"tau_final={cfg.model.gumbel_tau_final}"
        )

    # ------------------------------------------------------------------
    # Temperature annealing
    # ------------------------------------------------------------------

    def _current_tau(self, step: Optional[int]) -> float:
        """Cosine-then-clamp annealing of the Gumbel temperature."""
        cfg = self.cfg.model
        if step is None:
            return cfg.gumbel_tau_final
        frac = min(1.0, step / max(1, cfg.gumbel_anneal_steps))
        tau = cfg.gumbel_tau_final + 0.5 * (cfg.gumbel_tau_init - cfg.gumbel_tau_final) * (
            1 + math.cos(math.pi * frac)
        )
        return float(tau)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        token_embs: Float[torch.Tensor, "batch seq hidden"],
        mask: Int[torch.Tensor, "batch seq"],
        step: Optional[int] = None,
        training: bool = True,
    ) -> Tuple[
        Int[torch.Tensor, "batch seq"],
        Float[torch.Tensor, "batch seq"],
        Int[torch.Tensor, "batch"],
    ]:
        """Produce patch assignments for every token position.

        Parameters
        ----------
        token_embs:
            Contextualised representations ``(B, S, H)``.
        mask:
            Attention mask ``(B, S)``; 1 for real tokens, 0 for padding.
        step:
            Current training step used for Gumbel temperature annealing.
        training:
            Whether to use Gumbel-Softmax (True) or hard threshold (False).

        Returns
        -------
        patch_ids:
            ``(B, S)`` integer patch index for each token (-1 = padding).
        cut_probs:
            ``(B, S)`` raw sigmoid boundary probabilities (for regulariser).
        n_patches:
            ``(B,)`` number of patches per example.
        """
        B, S, _ = token_embs.shape

        # Cast to the FFN weight dtype — under 16-mixed precision the activations
        # arrive as fp16 but the Linear weights stay fp32.
        token_embs = token_embs.to(self.ffn[0].weight.dtype)

        # Raw boundary logits / probabilities
        logits = self.ffn(token_embs).squeeze(-1)  # (B, S)
        cut_probs = torch.sigmoid(logits)  # (B, S)

        # First valid token is ALWAYS a boundary; padding tokens are never.
        # Vectorised: position where cumulative mask count hits 1 (= first real token).
        first_valid = ((mask.cumsum(dim=1) == 1) & (mask > 0)).float()  # (B, S)

        if training:
            tau = self._current_tau(step)
            # Gumbel-Softmax trick: sample a hard-ish boundary decision
            # We treat the boundary decision as a Bernoulli with prob cut_probs.
            # Gumbel-Softmax for Bernoulli: use two-class formulation.
            log_p = torch.stack([torch.log(1 - cut_probs + 1e-8),
                                  torch.log(cut_probs + 1e-8)], dim=-1)  # (B, S, 2)
            gumbels = -torch.log(-torch.log(
                torch.rand_like(log_p).clamp(1e-8, 1 - 1e-8)
            ))
            y_soft = F.softmax((log_p + gumbels) / tau, dim=-1)  # (B, S, 2)
            # Straight-through: hard in forward, gradient through soft
            y_hard = (y_soft[..., 1] > 0.5).float()
            cuts = y_hard - y_soft[..., 1].detach() + y_soft[..., 1]  # (B, S)
        else:
            cuts = (cut_probs > 0.5).float()  # (B, S)

        # Force first valid token to always be a cut
        cuts = torch.clamp(cuts + first_valid, max=1.0)

        # Mask padding: only real tokens get cut decisions
        cuts = cuts * mask.float()

        # Cumulative sum of cuts gives monotone patch IDs (0-indexed)
        raw_ids = cuts.cumsum(dim=-1).long() - 1  # (B, S); starts at 0

        # Mark padding positions as -1
        patch_ids = raw_ids.masked_fill(mask == 0, -1)

        n_patches = raw_ids.masked_fill(mask == 0, 0).max(dim=-1).values + 1  # (B,)

        return patch_ids, cut_probs, n_patches
