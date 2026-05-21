# tests/test_mean_interaction.py

import torch
import pytest

from deep_stylometry.modules.mean_interaction import MeanInteraction
from tests.conftest import make_random_embs


class TestMeanInteractionShape:
    def test_output_shape(self):
        mi = MeanInteraction()
        q_embs, q_mask = make_random_embs(4, 32, 64)
        k_embs, k_mask = make_random_embs(8, 32, 64)
        scores = mi(q_embs, k_embs, q_mask, k_mask)
        assert scores.shape == (4, 8)

    def test_output_shape_single(self):
        mi = MeanInteraction()
        q_embs, q_mask = make_random_embs(1, 16, 64)
        k_embs, k_mask = make_random_embs(1, 16, 64)
        scores = mi(q_embs, k_embs, q_mask, k_mask)
        assert scores.shape == (1, 1)


class TestMeanInteractionCosine:
    def test_reduces_to_cosine_sim_single_token(self):
        """For single valid token sequences, mean pool = that token,
        and score = cosine similarity."""
        mi = MeanInteraction()
        B, H = 3, 64
        # Single-token sequences: only position 0 is valid
        q_embs = torch.randn(B, 1, H)
        k_embs = torch.randn(B, 1, H)
        q_mask = torch.ones(B, 1, dtype=torch.long)
        k_mask = torch.ones(B, 1, dtype=torch.long)

        scores = mi(q_embs, k_embs, q_mask, k_mask)

        import torch.nn.functional as F
        q_norm = F.normalize(q_embs.squeeze(1), p=2, dim=-1)
        k_norm = F.normalize(k_embs.squeeze(1), p=2, dim=-1)
        expected = torch.matmul(q_norm, k_norm.T)

        assert torch.allclose(scores, expected, atol=1e-5)

    def test_padding_does_not_affect_result(self):
        """Adding zero-masked tokens at the end should not change the result."""
        mi = MeanInteraction()
        B, S, H = 4, 16, 64
        torch.manual_seed(1)
        embs = torch.randn(B, S, H)
        mask = torch.ones(B, S, dtype=torch.long)
        mask[:, -4:] = 0  # Last 4 tokens are padding

        scores_a = mi(embs, embs, mask, mask).detach()

        # Extend with 4 more padding positions filled with random garbage
        extra = torch.randn(B, 4, H)
        extra_mask = torch.zeros(B, 4, dtype=torch.long)
        embs_ext = torch.cat([embs, extra], dim=1)
        mask_ext = torch.cat([mask, extra_mask], dim=1)

        scores_b = mi(embs_ext, embs_ext, mask_ext, mask_ext).detach()

        assert torch.allclose(scores_a, scores_b, atol=1e-5)


class TestMeanInteractionRange:
    def test_scores_in_unit_range(self):
        """Cosine similarity scores should be in [-1, 1]."""
        mi = MeanInteraction()
        q_embs, q_mask = make_random_embs(5, 20, 64)
        k_embs, k_mask = make_random_embs(7, 20, 64)
        scores = mi(q_embs, k_embs, q_mask, k_mask)
        assert scores.max() <= 1.0 + 1e-5
        assert scores.min() >= -1.0 - 1e-5
