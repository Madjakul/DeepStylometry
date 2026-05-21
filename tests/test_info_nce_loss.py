# tests/test_info_nce_loss.py

import torch
import torch.nn.functional as F
import pytest

from deep_stylometry.modules.info_nce_loss import InfoNCELoss
from tests.conftest import make_random_embs


class TestInfoNCELossBasics:
    def test_loss_is_scalar_and_non_negative(self, dummy_cfg):
        loss_fn = InfoNCELoss(dummy_cfg)
        B, S, H = 4, 32, 64
        q, q_mask = make_random_embs(B, S, H)
        k, k_mask = make_random_embs(2 * B, S, H)
        targets = torch.arange(B)
        out = loss_fn(q, k, q_mask, k_mask, targets)
        assert out["loss"].ndim == 0
        assert out["loss"].item() >= 0.0

    def test_poss_negs_shape(self, dummy_cfg):
        loss_fn = InfoNCELoss(dummy_cfg)
        B, S, H = 4, 32, 64
        q, q_mask = make_random_embs(B, S, H)
        k, k_mask = make_random_embs(2 * B, S, H)
        targets = torch.arange(B)
        out = loss_fn(q, k, q_mask, k_mask, targets)
        assert out["poss"].shape == (B,)
        assert out["negs"].shape == (B,)

    def test_all_scores_shape(self, dummy_cfg):
        loss_fn = InfoNCELoss(dummy_cfg)
        B, S, H = 4, 32, 64
        q, q_mask = make_random_embs(B, S, H)
        k, k_mask = make_random_embs(2 * B, S, H)
        targets = torch.arange(B)
        out = loss_fn(q, k, q_mask, k_mask, targets)
        assert out["all_scores"].shape == (B, 2 * B)


class TestInfoNCELossPerfectSeparation:
    def test_perfect_separation(self, dummy_cfg):
        """With identical query=positive, orthogonal negatives, loss → 0,
        poss > negs."""
        dummy_cfg.model.pooling_method = "mean"
        loss_fn = InfoNCELoss(dummy_cfg)

        B, H = 8, 64
        # Unit vectors along standard basis axes (orthogonal)
        basis = F.normalize(torch.eye(2 * B, H), p=2, dim=-1)

        q = basis[:B].unsqueeze(1)      # (B, 1, H) — query
        pos = basis[:B].unsqueeze(1)    # (B, 1, H) — same as query
        neg = basis[B:].unsqueeze(1)    # (B, 1, H) — orthogonal

        q_mask = torch.ones(B, 1, dtype=torch.long)
        k_mask = torch.ones(2 * B, 1, dtype=torch.long)

        k = torch.cat([pos, neg], dim=0)
        targets = torch.arange(B)

        out = loss_fn(q, k, q_mask, k_mask, targets)
        assert (out["poss"] > out["negs"]).all()
        assert out["loss"].item() < 0.5


class TestInfoNCELossMeanPooling:
    def test_mean_pooling_variant(self, dummy_cfg):
        dummy_cfg.model.pooling_method = "mean"
        loss_fn = InfoNCELoss(dummy_cfg)
        B, S, H = 3, 16, 64
        q, q_mask = make_random_embs(B, S, H)
        k, k_mask = make_random_embs(2 * B, S, H)
        targets = torch.arange(B)
        out = loss_fn(q, k, q_mask, k_mask, targets)
        assert out["loss"].isfinite()


class TestInfoNCELossPLI:
    def test_pli_ngram(self, pli_cfg):
        """InfoNCELoss with pooling_method=pli runs without error."""
        loss_fn = InfoNCELoss(pli_cfg)
        B, S, H = 4, 32, 64
        q, q_mask = make_random_embs(B, S, H)
        k, k_mask = make_random_embs(2 * B, S, H)
        targets = torch.arange(B)
        out = loss_fn(q, k, q_mask, k_mask, targets)
        assert out["loss"].isfinite()
        assert out["all_scores"].shape == (B, 2 * B)

    def test_pli_learned_returns_patch_reg_loss(self, learned_pli_cfg):
        """InfoNCELoss with learned PLI should return patch_reg_loss."""
        loss_fn = InfoNCELoss(learned_pli_cfg)
        B, S, H = 4, 32, 64
        q, q_mask = make_random_embs(B, S, H)
        k, k_mask = make_random_embs(2 * B, S, H)
        targets = torch.arange(B)
        out = loss_fn(q, k, q_mask, k_mask, targets, step=0)
        assert "patch_reg_loss" in out
        assert out["patch_reg_loss"].isfinite()
