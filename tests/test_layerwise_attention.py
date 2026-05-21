# tests/test_layerwise_attention.py
"""Unit tests for LayerwiseAttention module."""

import pytest
import torch

from deep_stylometry.modules.layerwise_attention import LayerwiseAttention


class TestLayerwiseAttentionShape:
    def test_output_shape(self):
        """Output shape matches (batch, seq, hidden)."""
        B, S, H, L = 4, 32, 64, 5
        module = LayerwiseAttention(num_layers=L)
        hidden_states = [torch.randn(B, S, H) for _ in range(L)]
        out = module(hidden_states)
        assert out.shape == (B, S, H), f"Expected ({B}, {S}, {H}), got {out.shape}"

    def test_output_shape_single_layer(self):
        """Works with a single hidden state."""
        module = LayerwiseAttention(num_layers=1)
        out = module([torch.randn(2, 16, 32)])
        assert out.shape == (2, 16, 32)

    def test_output_dtype_preserved(self):
        """Output dtype matches input dtype."""
        module = LayerwiseAttention(num_layers=3)
        hidden_states = [torch.randn(2, 10, 8, dtype=torch.float32) for _ in range(3)]
        out = module(hidden_states)
        assert out.dtype == torch.float32


class TestLayerwiseAttentionWeights:
    def test_softmax_weights_sum_to_one(self):
        """Softmax-normalised scalar weights sum to 1."""
        L = 7
        module = LayerwiseAttention(num_layers=L)
        weights = torch.cat(list(module.scalar_parameters))
        normed = torch.softmax(weights, dim=0)
        assert torch.isclose(normed.sum(), torch.tensor(1.0), atol=1e-6), (
            f"Weights sum to {normed.sum().item()}, expected 1.0"
        )

    def test_initial_uniform_weights(self):
        """With zero scalar parameters the weights are uniform."""
        L = 4
        module = LayerwiseAttention(num_layers=L)
        # All scalar params are initialised to 0.0
        weights = torch.cat(list(module.scalar_parameters))
        normed = torch.softmax(weights, dim=0)
        expected = torch.full((L,), 1.0 / L)
        assert torch.allclose(normed, expected, atol=1e-6)

    def test_gamma_learnable(self):
        """gamma is a learnable Parameter."""
        module = LayerwiseAttention(num_layers=3)
        assert isinstance(module.gamma, torch.nn.Parameter)
        assert module.gamma.requires_grad


class TestLayerwiseAttentionGradients:
    def test_gradient_flows(self):
        """Scalar parameters and gamma receive gradients after backward."""
        L = 3
        B, S, H = 2, 8, 16
        module = LayerwiseAttention(num_layers=L)
        hidden_states = [torch.randn(B, S, H) for _ in range(L)]
        out = module(hidden_states)
        loss = out.sum()
        loss.backward()

        for i, param in enumerate(module.scalar_parameters):
            assert param.grad is not None, f"scalar_parameters[{i}] has no gradient"
            assert param.grad.abs().sum() > 0, f"scalar_parameters[{i}] gradient is zero"

        assert module.gamma.grad is not None, "gamma has no gradient"
        assert module.gamma.grad.abs().sum() > 0, "gamma gradient is zero"

    def test_hidden_states_do_not_need_grad(self):
        """Hidden states (detached inputs) still allow backward on parameters."""
        L = 2
        module = LayerwiseAttention(num_layers=L)
        hidden_states = [torch.randn(2, 4, 8, requires_grad=False) for _ in range(L)]
        out = module(hidden_states)
        out.sum().backward()
        for p in module.scalar_parameters:
            assert p.grad is not None


class TestLayerwiseAttentionErrors:
    def test_wrong_number_of_hidden_states_raises(self):
        """Passing wrong number of hidden states raises ValueError."""
        module = LayerwiseAttention(num_layers=5)
        wrong_states = [torch.randn(2, 8, 16) for _ in range(3)]  # 3 ≠ 5
        with pytest.raises(ValueError, match="5.*3|3.*5"):
            module(wrong_states)

    def test_empty_hidden_states_raises(self):
        """Zero hidden states (num_layers=0) raises ValueError."""
        module = LayerwiseAttention(num_layers=3)
        with pytest.raises(ValueError):
            module([])


class TestLayerwiseAttentionDeterminism:
    def test_deterministic_with_same_input(self):
        """Same input → same output (no stochastic components)."""
        L = 4
        module = LayerwiseAttention(num_layers=L)
        module.eval()
        hidden = [torch.randn(2, 6, 8) for _ in range(L)]
        out1 = module(hidden)
        out2 = module(hidden)
        assert torch.allclose(out1, out2)
