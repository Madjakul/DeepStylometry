# tests/test_mean_centerer.py
"""Unit tests for MeanCenterer module."""

import torch
import pytest

from deep_stylometry.modules.mean_centerer import MeanCenterer


class TestMeanCentererUpdate:
    def test_update_shifts_mu_toward_batch_mean(self):
        """After one update, mu equals the batch mean."""
        dim = 8
        mc = MeanCenterer(dim)
        embs = torch.randn(16, dim)
        mc.update(embs)
        expected_mean = embs.float().mean(dim=0)
        assert torch.allclose(mc.mu, expected_mean, atol=1e-5), (
            f"mu={mc.mu}, expected={expected_mean}"
        )

    def test_multiple_updates_converge_to_global_mean(self):
        """Multiple update calls converge to the true global mean."""
        dim = 4
        mc = MeanCenterer(dim)
        torch.manual_seed(0)
        # Generate 100 batches of size 32
        all_embs = torch.randn(3200, dim)
        batch_size = 32
        for i in range(0, 3200, batch_size):
            mc.update(all_embs[i: i + batch_size])

        true_mean = all_embs.float().mean(dim=0)
        assert torch.allclose(mc.mu, true_mean, atol=1e-4), (
            f"Running mean {mc.mu} does not match true mean {true_mean}"
        )

    def test_empty_batch_no_op(self):
        """update() with empty tensor leaves mu unchanged."""
        dim = 4
        mc = MeanCenterer(dim)
        # Prime the centerer with some data
        mc.update(torch.ones(10, dim))
        mu_before = mc.mu.clone()
        # Now update with empty tensor
        mc.update(torch.zeros(0, dim))
        assert torch.allclose(mc.mu, mu_before), "Empty batch should not change mu"

    def test_n_accumulates(self):
        """Buffer n tracks the total number of samples seen."""
        mc = MeanCenterer(4)
        mc.update(torch.randn(8, 4))
        mc.update(torch.randn(12, 4))
        assert mc.n.item() == 20


class TestMeanCentererApply:
    def test_apply_without_update_is_l2_norm(self):
        """Before any update, mu is zero, so apply() is pure L2 normalisation."""
        dim = 6
        mc = MeanCenterer(dim)
        embs = torch.randn(5, dim)
        out = mc.apply(embs)
        norms = out.norm(p=2, dim=-1)
        assert torch.allclose(norms, torch.ones(5), atol=1e-6)

    def test_apply_centers_and_normalizes(self):
        """After updating with known embeddings, apply produces unit-norm output."""
        dim = 4
        mc = MeanCenterer(dim)
        known = torch.ones(10, dim) * 5.0
        mc.update(known)
        # Apply to a fresh batch
        out = mc.apply(torch.randn(8, dim) + 5.0)
        norms = out.norm(p=2, dim=-1)
        assert torch.allclose(norms, torch.ones(8), atol=1e-5)

    def test_apply_does_not_modify_input(self):
        """apply() should not modify the input tensor in-place."""
        mc = MeanCenterer(4)
        embs = torch.randn(3, 4)
        original = embs.clone()
        mc.apply(embs)
        assert torch.allclose(embs, original), "Input was modified in-place"

    def test_apply_output_shape(self):
        """Output shape matches input shape."""
        mc = MeanCenterer(16)
        embs = torch.randn(7, 16)
        out = mc.apply(embs)
        assert out.shape == embs.shape


class TestMeanCentererBuffers:
    def test_mu_is_buffer_not_parameter(self):
        """mu should be a buffer (not a learnable parameter)."""
        mc = MeanCenterer(8)
        assert "mu" in dict(mc.named_buffers()), "mu should be a buffer"
        assert "mu" not in dict(mc.named_parameters()), "mu should not be a parameter"

    def test_n_is_buffer_not_parameter(self):
        """n should be a buffer (not a learnable parameter)."""
        mc = MeanCenterer(8)
        assert "n" in dict(mc.named_buffers()), "n should be a buffer"
        assert "n" not in dict(mc.named_parameters()), "n should not be a parameter"

    def test_state_dict_contains_buffers(self):
        """Buffers appear in the state dict (they will be saved in checkpoint)."""
        mc = MeanCenterer(4)
        mc.update(torch.randn(5, 4))
        sd = mc.state_dict()
        assert "mu" in sd
        assert "n" in sd

    def test_load_state_dict(self):
        """After save/load, the restored mu matches the original."""
        dim = 4
        mc1 = MeanCenterer(dim)
        mc1.update(torch.randn(20, dim))

        mc2 = MeanCenterer(dim)
        mc2.load_state_dict(mc1.state_dict())
        assert torch.allclose(mc1.mu, mc2.mu)
        assert mc1.n.item() == mc2.n.item()
