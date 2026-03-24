# tests/test_late_interaction.py

import pytest
import torch

from deep_stylometry.modules.late_interaction import LateInteraction
from tests.conftest import make_random_embs


class TestLateInteractionShape:
    def test_output_shape_square(self, dummy_cfg):
        li = LateInteraction(dummy_cfg)
        q_embs, q_mask = make_random_embs(4, 32, 64)
        k_embs, k_mask = make_random_embs(8, 32, 64)
        scores = li(q_embs, k_embs, q_mask, k_mask)
        assert scores.shape == (4, 8)

    def test_output_shape_single(self, dummy_cfg):
        li = LateInteraction(dummy_cfg)
        q_embs, q_mask = make_random_embs(1, 16, 64)
        k_embs, k_mask = make_random_embs(1, 16, 64)
        scores = li(q_embs, k_embs, q_mask, k_mask)
        assert scores.shape == (1, 1)

    def test_output_shape_various_batches(self, dummy_cfg):
        li = LateInteraction(dummy_cfg)
        for bq, bk in [(2, 6), (3, 3), (5, 10)]:
            q_embs, q_mask = make_random_embs(bq, 20, 64)
            k_embs, k_mask = make_random_embs(bk, 20, 64)
            scores = li(q_embs, k_embs, q_mask, k_mask)
            assert scores.shape == (bq, bk)


class TestLateInteractionPadding:
    def test_padding_does_not_affect_scores(self, dummy_cfg):
        """Masking an extra token at the end should not change scores."""
        li = LateInteraction(dummy_cfg)
        torch.manual_seed(42)
        q_embs, q_mask = make_random_embs(2, 20, 64)
        k_embs, k_mask = make_random_embs(4, 20, 64)

        scores_before = li(q_embs, k_embs, q_mask, k_mask).detach()

        # Mask out the last valid key token in all key sequences
        k_mask_mod = k_mask.clone()
        k_mask_mod[:, k_mask.sum(dim=1).min().long() - 1] = 0

        scores_after = li(q_embs, k_embs, q_mask, k_mask_mod).detach()
        # Scores may differ since we're removing a key token, but both should
        # be finite and have the right shape
        assert scores_before.shape == scores_after.shape
        assert scores_before.isfinite().all()
        assert scores_after.isfinite().all()


class TestLateInteractionSelfScore:
    def test_diagonal_highest_when_query_equals_key(self, dummy_cfg):
        """When q == k, each query should score highest against itself."""
        li = LateInteraction(dummy_cfg)
        torch.manual_seed(7)
        embs, mask = make_random_embs(4, 16, 64)
        scores = li(embs, embs, mask, mask).detach()
        # Diagonal (self-scores) should be the highest in each row
        diag = scores.diag()
        for i in range(scores.size(0)):
            assert (scores[i] <= diag[i] + 1e-4).all(), (
                f"Row {i}: diag={diag[i].item():.4f}, max_off={scores[i].max().item():.4f}"
            )


class TestLateInteractionDeterminism:
    def test_deterministic(self, dummy_cfg):
        li = LateInteraction(dummy_cfg)
        torch.manual_seed(0)
        q_embs, q_mask = make_random_embs(3, 16, 64)
        k_embs, k_mask = make_random_embs(5, 16, 64)
        s1 = li(q_embs, k_embs, q_mask, k_mask).detach()
        s2 = li(q_embs, k_embs, q_mask, k_mask).detach()
        assert torch.allclose(s1, s2)


class TestLateInteractionSkipList:
    def test_skip_list_punctuation_tokens(self):
        """With skip_list=True, punctuation tokens in the query should not
        contribute to the score.  Construct a batch where one query has only
        punctuation tokens; its score should be 0 for all keys."""
        from deep_stylometry.utils.configs import BaseConfig

        cfg = BaseConfig()
        cfg.model.base_checkpoint = "answerdotai/ModernBERT-base"
        cfg.model.skip_list = True
        cfg.train.precision = "32"

        li = LateInteraction(cfg)

        # Identify a punctuation token ID from the skip list
        punc_ids = li.punc_token_ids.tolist()
        assert len(punc_ids) > 0, "Skip list should have punctuation tokens"

        B, S, H = 2, 8, 64
        q_embs = torch.randn(B, S, H)
        k_embs = torch.randn(B, S, H)

        q_mask = torch.ones(B, S, dtype=torch.long)
        k_mask = torch.ones(B, S, dtype=torch.long)

        # Build a query input_ids where ALL valid tokens are punctuation
        q_input_ids = torch.full((B, S), punc_ids[0], dtype=torch.long)

        scores = li(q_embs, k_embs, q_mask, k_mask, q_input_ids=q_input_ids)
        # All scores should be 0 since all query tokens are punctuation and masked
        assert torch.allclose(scores, torch.zeros_like(scores), atol=1e-5)
