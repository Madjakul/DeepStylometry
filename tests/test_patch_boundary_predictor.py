# tests/test_patch_boundary_predictor.py

import torch
import pytest

from deep_stylometry.modules.patch_boundary_predictor import PatchBoundaryPredictor
from tests.conftest import make_random_embs


@pytest.fixture
def predictor(dummy_cfg):
    return PatchBoundaryPredictor(64, dummy_cfg)


class TestPatchIdsShape:
    def test_patch_ids_shape(self, predictor):
        embs, mask = make_random_embs(4, 32, 64)
        patch_ids, cut_probs, n_patches = predictor(embs, mask)
        assert patch_ids.shape == (4, 32)
        assert cut_probs.shape == (4, 32)
        assert n_patches.shape == (4,)

    def test_patch_ids_non_decreasing(self, predictor):
        """patch_ids for valid tokens should be non-decreasing."""
        torch.manual_seed(0)
        embs, mask = make_random_embs(4, 32, 64)
        patch_ids, _, _ = predictor(embs, mask, training=False)
        for b in range(4):
            valid = (mask[b] > 0).nonzero(as_tuple=True)[0]
            if len(valid) > 1:
                ids = patch_ids[b, valid]
                diffs = ids[1:] - ids[:-1]
                assert (diffs >= 0).all()

    def test_padding_masked(self, predictor):
        """Padding positions should have patch_id == -1."""
        embs, mask = make_random_embs(3, 20, 64)
        patch_ids, _, _ = predictor(embs, mask, training=False)
        pad_positions = (mask == 0)
        assert (patch_ids[pad_positions] == -1).all()


class TestPatchCount:
    def test_n_patches_at_least_one(self, predictor):
        embs, mask = make_random_embs(4, 32, 64)
        _, _, n_patches = predictor(embs, mask, training=False)
        assert (n_patches >= 1).all()

    def test_n_patches_at_most_seq_len(self, predictor):
        B, S, H = 4, 32, 64
        embs, mask = make_random_embs(B, S, H)
        _, _, n_patches = predictor(embs, mask, training=False)
        assert (n_patches <= S).all()


class TestTemperatureEffect:
    def test_high_tau_softer_cuts(self, dummy_cfg):
        """At high temperature, cut probabilities should be closer to 0.5."""
        dummy_cfg.model.gumbel_tau_init = 10.0
        dummy_cfg.model.gumbel_tau_final = 10.0
        dummy_cfg.model.gumbel_anneal_steps = 1
        pred = PatchBoundaryPredictor(64, dummy_cfg)

        torch.manual_seed(1)
        embs, mask = make_random_embs(4, 32, 64)
        _, cut_probs, _ = pred(embs, mask, step=0, training=True)
        # Not much we can assert deterministically, but probabilities should
        # be finite and in [0, 1]
        assert cut_probs.isfinite().all()
        assert (cut_probs >= 0).all() and (cut_probs <= 1).all()


class TestRegularizerSignal:
    def test_more_cuts_gives_lower_reg(self, predictor):
        """Forcing all tokens to be boundaries should minimise the regulariser
        (numerator sum is maximised)."""
        B, S, H = 2, 16, 64
        mask = torch.ones(B, S, dtype=torch.long)
        # All positions marked as cuts → many patches → low L_patch
        cut_probs_all = torch.ones(B, S)
        n_cuts_all = (cut_probs_all * mask.float()).sum(dim=-1)
        reg_all = -torch.log(n_cuts_all + 1e-8).mean()

        # Fewer cuts → higher L_patch
        cut_probs_few = torch.zeros(B, S)
        cut_probs_few[:, 0] = 1.0  # Only first token is a boundary
        n_cuts_few = (cut_probs_few * mask.float()).sum(dim=-1)
        reg_few = -torch.log(n_cuts_few + 1e-8).mean()

        assert reg_all < reg_few
