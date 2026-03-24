# deep_stylometry/modules/patch_interaction.py

import logging
import math
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int

from deep_stylometry.utils.helpers import get_tokenizer

if TYPE_CHECKING:
    from deep_stylometry.utils.configs import BaseConfig


class PatchInteraction(nn.Module):
    """Patch-level late interaction.

    Groups token embeddings into variable- or fixed-length patches, compresses
    each patch into a single vector, then computes MaxSim between query patches
    and key patches — the patch-level analogue of ColBERT-style token MaxSim.

    Patching strategies (``cfg.model.patch_method``):
    - ``"whitespace"``     : word-boundary splits using the tokeniser's Ġ prefix.
    - ``"wholeword"``      : same as whitespace for ModernBERT (Ġ-based tokeniser).
    - ``"ngram"``          : non-overlapping n-grams of size ``cfg.model.patch_size``.
    - ``"learned"``        : dynamic boundaries from ``PatchBoundaryPredictor``.

    Compression methods (``cfg.model.patch_compression``):
    - ``"mean"``           : mean-pool tokens within each patch.
    - ``"max"``            : max-pool tokens within each patch.
    - ``"cross_attention"``  : learned cross-attention compression (requires
                              ``patch_method="learned"``).

    The forward signature matches ``LateInteraction`` so the module is a
    drop-in replacement inside ``InfoNCELoss`` and ``TestEvalCallback``.
    """

    def __init__(self, cfg: "BaseConfig") -> None:
        super().__init__()
        logging.info(
            f"PatchInteraction: method={cfg.model.patch_method}, "
            f"compression={cfg.model.patch_compression}, "
            f"patch_size={cfg.model.patch_size}"
        )
        self.cfg = cfg
        self.patch_method = cfg.model.patch_method
        self.patch_size = cfg.model.patch_size
        self.compression = cfg.model.patch_compression

        # Tokeniser (needed for whitespace / wholeword boundaries).
        # Build a (vocab_size,) boolean lookup table so forward() uses O(1)-per-
        # token direct indexing instead of O(S log V) torch.isin binary search.
        if self.patch_method in ("whitespace", "wholeword"):
            tokenizer = get_tokenizer(cfg.model.base_checkpoint)
            vocab = tokenizer.get_vocab()
            vocab_size = max(vocab.values()) + 1

            # is_word_start[token_id] = True  →  token begins a new word patch
            # is_special[token_id]    = True  →  token is a special token (singleton patch)
            is_word_start_lut = torch.zeros(vocab_size, dtype=torch.bool)
            is_special_lut = torch.zeros(vocab_size, dtype=torch.bool)

            special_ids_set: set = set()
            for attr in ("cls_token_id", "sep_token_id", "pad_token_id",
                         "bos_token_id", "eos_token_id", "mask_token_id"):
                val = getattr(tokenizer, attr, None)
                if val is not None:
                    special_ids_set.add(val)
            if special_ids_set:
                is_special_lut[torch.tensor(sorted(special_ids_set), dtype=torch.long)] = True

            n_word_start = 0
            for token_str, token_id in vocab.items():
                if token_str.startswith("Ġ") or token_str.startswith("▁"):
                    is_word_start_lut[token_id] = True
                    n_word_start += 1

            self.register_buffer("is_word_start_lut", is_word_start_lut, persistent=False)
            self.register_buffer("is_special_lut", is_special_lut, persistent=False)
            logging.info(
                f"PatchInteraction: boundary LUT built — "
                f"{n_word_start} word-start tokens, "
                f"{len(special_ids_set)} special tokens (vocab_size={vocab_size})."
            )
        else:
            self.is_word_start_lut = None
            self.is_special_lut = None

        # Learned patching sub-modules
        if self.patch_method == "learned":
            from deep_stylometry.modules.patch_boundary_predictor import (
                PatchBoundaryPredictor,
            )
            h = cfg.model.lm_hidden_size
            self.predictor = PatchBoundaryPredictor(h, cfg)

            if self.compression == "cross_attention":
                from deep_stylometry.modules.cross_attention_compressor import (
                    CrossAttentionCompressor,
                )
                self.compressor = CrossAttentionCompressor(
                    h,
                    cfg.model.patch_cross_attn_dim,
                    cfg.model.patch_cross_attn_heads,
                )
            else:
                self.compressor = None
        else:
            self.predictor = None
            self.compressor = None

        # Stored after each forward for regulariser access
        self._last_cut_probs: Optional[torch.Tensor] = None
        self._last_q_mask: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Patch-boundary helpers
    # ------------------------------------------------------------------

    def _whitespace_patches(
        self,
        input_ids: Int[torch.Tensor, "batch seq"],
        mask: Int[torch.Tensor, "batch seq"],
    ) -> Int[torch.Tensor, "batch seq"]:
        """Assign patch IDs based on Ġ-prefix word boundaries.

        A new patch begins at:
        - Special tokens (CLS, SEP, etc.) — each gets its own singleton patch.
        - Tokens whose decoded form starts with Ġ/▁ (word-start tokens).
        - The first valid (non-padding) token of each example.
        - The first non-special token after any special token (handles the case
          where the first word of a sentence lacks the Ġ prefix).

        Uses O(1)-per-token direct LUT indexing — no binary search, no Python
        loops, fully GPU-compatible and safe for DDP.
        """
        # Direct LUT indexing: O(B*S) gather, much faster than torch.isin
        is_special = self.is_special_lut[input_ids]      # (B, S)
        is_word_start = self.is_word_start_lut[input_ids]  # (B, S)

        # Token immediately after a special token starts a new patch even if it
        # lacks the Ġ prefix (e.g. first word after CLS).
        # Shift is_special one step to the right; pad the new position with False.
        is_after_special = F.pad(
            is_special[:, :-1], (1, 0), value=False
        )  # (B, S)

        # First valid (non-padding) position per example
        is_first_valid = (mask.cumsum(dim=1) == 1) & (mask > 0)  # (B, S)

        is_boundary = (
            is_special | is_word_start | is_after_special | is_first_valid
        ) & (mask > 0)

        patch_ids = is_boundary.long().cumsum(dim=-1) - 1  # 0-indexed
        patch_ids = patch_ids.masked_fill(mask == 0, -1)
        return patch_ids

    def _wholeword_patches(
        self,
        input_ids: Int[torch.Tensor, "batch seq"],
        mask: Int[torch.Tensor, "batch seq"],
    ) -> Int[torch.Tensor, "batch seq"]:
        """Assign patch IDs by merging sub-word continuations into whole words.

        For ModernBERT (Ġ-prefix tokeniser) this is identical to
        ``_whitespace_patches``; kept as a separate entry point for clarity
        and future compatibility with ``##``-style tokenisers.
        """
        return self._whitespace_patches(input_ids, mask)

    @staticmethod
    def _ngram_patches(
        mask: Int[torch.Tensor, "batch seq"],
        n: int,
    ) -> Int[torch.Tensor, "batch seq"]:
        """Assign patch IDs as ``position // n`` for non-padding tokens."""
        B, S = mask.shape
        positions = torch.arange(S, device=mask.device).unsqueeze(0).expand(B, -1)
        patch_ids = (positions // max(1, n)).clone()
        patch_ids[mask == 0] = -1
        return patch_ids

    def _learned_patches(
        self,
        embs: Float[torch.Tensor, "batch seq hidden"],
        mask: Int[torch.Tensor, "batch seq"],
        step: Optional[int],
        training: bool,
    ) -> Tuple[Int[torch.Tensor, "batch seq"], Float[torch.Tensor, "batch seq"]]:
        patch_ids, cut_probs, _ = self.predictor(embs, mask, step, training)
        return patch_ids, cut_probs

    # ------------------------------------------------------------------
    # Unified patch computation
    # ------------------------------------------------------------------

    def _compute_patches(
        self,
        embs: Float[torch.Tensor, "batch seq hidden"],
        mask: Int[torch.Tensor, "batch seq"],
        input_ids: Optional[Int[torch.Tensor, "batch seq"]],
        step: Optional[int],
        training: bool,
    ) -> Tuple[Int[torch.Tensor, "batch seq"], Optional[Float[torch.Tensor, "batch seq"]]]:
        method = self.patch_method
        # Fall back to ngram when input_ids are required but unavailable
        if input_ids is None and method in ("whitespace", "wholeword"):
            logging.warning(
                f"input_ids not provided for {method} patching; "
                f"silently falling back to ngram-{self.patch_size}. "
                f"Pass k_input_ids to avoid this."
            )
            method = "ngram"

        if method == "whitespace":
            return self._whitespace_patches(input_ids, mask), None
        elif method == "wholeword":
            return self._wholeword_patches(input_ids, mask), None
        elif method == "ngram":
            return self._ngram_patches(mask, self.patch_size), None
        elif method == "learned":
            return self._learned_patches(embs, mask, step, training)
        else:
            raise ValueError(f"Unknown patch_method: {method!r}")

    # ------------------------------------------------------------------
    # Patch compression
    # ------------------------------------------------------------------

    def _compress_patches(
        self,
        embs: Float[torch.Tensor, "batch seq hidden"],
        patch_ids: Int[torch.Tensor, "batch seq"],
        mask: Int[torch.Tensor, "batch seq"],
        max_patches: int,
        is_query: bool = True,
    ) -> Tuple[
        Float[torch.Tensor, "batch patches hidden"],
        Int[torch.Tensor, "batch patches"],
    ]:
        """Aggregate token embeddings into patch embeddings.

        Parameters
        ----------
        embs:       ``(B, S, H)`` token embeddings.
        patch_ids:  ``(B, S)`` integer patch assignments; -1 for padding.
        mask:       ``(B, S)`` attention mask.
        max_patches: ``P`` number of patch slots.
        is_query:   Whether these are query embeddings (used for cross-attention routing).

        Returns
        -------
        patch_embs:  ``(B, P, H)``
        patch_mask:  ``(B, P)``
        """
        B, S, H = embs.shape
        P = max_patches
        device = embs.device

        valid = (mask > 0) & (patch_ids >= 0)  # (B, S)
        ids_clamped = patch_ids.clamp(min=0)  # (B, S)

        if self.compression == "cross_attention" and is_query and self.compressor is not None:
            return self.compressor(embs, patch_ids, mask, P)

        # One-hot scatter for mean / max pooling
        one_hot = F.one_hot(ids_clamped, num_classes=P).float()  # (B, S, P)
        one_hot = one_hot * valid.unsqueeze(-1).float()

        if self.compression == "mean" or (
            self.compression == "cross_attention"
        ):
            # For cross_attention on keys (no compressor for keys), fall back to mean
            embs_valid = embs * valid.unsqueeze(-1).float()
            patch_embs = torch.einsum("bsh,bsp->bph", embs_valid, one_hot)  # (B, P, H)
            counts = one_hot.sum(dim=1)  # (B, P)
            patch_embs = patch_embs / counts.unsqueeze(-1).clamp(min=1.0)
            patch_mask = (counts > 0).long()

        else:  # max
            # Fill invalid with large negative, then scatter max
            embs_masked = embs * valid.unsqueeze(-1).float() + (
                1.0 - valid.unsqueeze(-1).float()
            ) * (-1e9)
            # (B, S, H) → expand to (B, S, P, H), then max over S → (B, P, H)
            # Memory-efficient: use scatter_reduce_ if available, else loop
            try:
                idx = ids_clamped.unsqueeze(-1).expand(-1, -1, H)  # (B, S, H)
                patch_embs = torch.full((B, P, H), -1e9,
                                        device=device, dtype=embs.dtype)
                patch_embs.scatter_reduce_(1, idx, embs_masked,
                                           reduce="amax", include_self=True)
                patch_embs = patch_embs.clamp(min=-1e8)  # Replace -1e9 sentinel
            except (RuntimeError, TypeError):
                # Fallback: mean pooling
                embs_valid = embs * valid.unsqueeze(-1).float()
                patch_embs = torch.einsum("bsh,bsp->bph", embs_valid, one_hot)
                counts = one_hot.sum(dim=1)
                patch_embs = patch_embs / counts.unsqueeze(-1).clamp(min=1.0)

            counts = one_hot.sum(dim=1)
            patch_embs = patch_embs * (counts > 0).unsqueeze(-1).float()
            patch_mask = (counts > 0).long()

        return patch_embs, patch_mask

    # ------------------------------------------------------------------
    # MaxSim
    # ------------------------------------------------------------------

    @staticmethod
    def _maxsim(
        q_patch_embs: Float[torch.Tensor, "bq pq hidden"],
        k_patch_embs: Float[torch.Tensor, "bk pk hidden"],
        q_patch_mask: Int[torch.Tensor, "bq pq"],
        k_patch_mask: Int[torch.Tensor, "bk pk"],
    ) -> Float[torch.Tensor, "bq bk"]:
        """Compute MaxSim scores between query and key patch sets."""
        q_norm = F.normalize(q_patch_embs, p=2, dim=-1)
        k_norm = F.normalize(k_patch_embs, p=2, dim=-1)

        # (Bq, Bk, Pq, Pk)
        sim = torch.einsum("ash,bth->abst", q_norm, k_norm)

        # Mask padding key patches
        k_mask_inv = (1.0 - k_patch_mask.float()).unsqueeze(0).unsqueeze(2)  # (1, Bk, 1, Pk)
        sim = sim + k_mask_inv * (-10000.0)

        # MaxSim over key patches per query patch → (Bq, Bk, Pq)
        max_scores = sim.max(dim=-1).values

        # Mask padding query patches and sum → (Bq, Bk)
        max_scores = max_scores * q_patch_mask.float().unsqueeze(1)
        return max_scores.sum(dim=-1)

    # ------------------------------------------------------------------
    # Regulariser
    # ------------------------------------------------------------------

    def get_patch_reg_loss(self) -> Optional[torch.Tensor]:
        """Return the patch-count regulariser L_patch = -log(sum(cut_probs)).

        Returns ``None`` when no cut probs are available (fixed patching).
        """
        if self._last_cut_probs is None or self._last_q_mask is None:
            return None
        cut_probs = self._last_cut_probs
        mask = self._last_q_mask
        n_cuts = (cut_probs * mask.float()).sum(dim=-1)  # (B,)
        return -torch.log(n_cuts + 1e-8).mean()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        query_embs: Float[torch.Tensor, "batch seq hidden"],
        key_embs: Float[torch.Tensor, "n_times_batch seq hidden"],
        q_mask: Int[torch.Tensor, "batch seq"],
        k_mask: Int[torch.Tensor, "n_times_batch seq"],
        q_input_ids: Optional[Int[torch.Tensor, "batch seq"]] = None,
        k_input_ids: Optional[Int[torch.Tensor, "n_times_batch seq"]] = None,
        step: Optional[int] = None,
    ) -> Float[torch.Tensor, "batch n_times_batch"]:
        """Compute patch-level MaxSim scores.

        Parameters
        ----------
        query_embs:   ``(Bq, S, H)``
        key_embs:     ``(Bk, S, H)``
        q_mask:       ``(Bq, S)``
        k_mask:       ``(Bk, S)``
        q_input_ids:  ``(Bq, S)`` optional; required for whitespace / wholeword.
        k_input_ids:  ``(Bk, S)`` optional; falls back to ngram if absent.
        step:         Current training step for Gumbel temperature annealing.

        Returns
        -------
        scores:       ``(Bq, Bk)``
        """
        training = self.training

        # Compute patch assignments
        q_patch_ids, q_cut_probs = self._compute_patches(
            query_embs, q_mask, q_input_ids, step, training
        )
        k_patch_ids, _ = self._compute_patches(
            key_embs, k_mask, k_input_ids, step=None, training=False
        )

        # Determine number of patch slots
        q_max_p = max(1, int(q_patch_ids.clamp(min=-1).max().item()) + 1) if q_patch_ids.max() >= 0 else 1
        k_max_p = max(1, int(k_patch_ids.clamp(min=-1).max().item()) + 1) if k_patch_ids.max() >= 0 else 1

        # Compress patches
        q_patch_embs, q_patch_mask = self._compress_patches(
            query_embs, q_patch_ids, q_mask, q_max_p, is_query=True
        )
        k_patch_embs, k_patch_mask = self._compress_patches(
            key_embs, k_patch_ids, k_mask, k_max_p, is_query=False
        )

        # Compute MaxSim
        scores = self._maxsim(q_patch_embs, k_patch_embs, q_patch_mask, k_patch_mask)

        # Store for regulariser
        self._last_cut_probs = q_cut_probs
        self._last_q_mask = q_mask

        return scores
