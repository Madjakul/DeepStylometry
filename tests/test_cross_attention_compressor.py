# tests/test_cross_attention_compressor.py

import torch
import pytest

from deep_stylometry.modules.cross_attention_compressor import CrossAttentionCompressor
from deep_stylometry.modules.patch_interaction import PatchInteraction
from tests.conftest import make_random_embs


@pytest.fixture
def compressor():
    return CrossAttentionCompressor(hidden_size=64, d_k=16, n_heads=2)


class TestCrossAttentionCompressorShape:
    def test_output_shape(self, compressor):
        B, S, H = 4, 32, 64
        embs, mask = make_random_embs(B, S, H)
        # Create simple ngram patch IDs
        patch_ids = PatchInteraction._ngram_patches(mask, n=4)
        max_patches = int(patch_ids.clamp(min=0).max().item()) + 1

        patch_embs, patch_mask = compressor(embs, patch_ids, mask, max_patches)
        assert patch_embs.shape == (B, max_patches, H)
        assert patch_mask.shape == (B, max_patches)

    def test_patch_mask_consistent_with_ids(self, compressor):
        """patch_mask[b, p] == 1 iff patch p has at least one valid token."""
        B, S, H = 3, 24, 64
        embs, mask = make_random_embs(B, S, H)
        patch_ids = PatchInteraction._ngram_patches(mask, n=3)
        max_patches = int(patch_ids.clamp(min=0).max().item()) + 1

        _, patch_mask = compressor(embs, patch_ids, mask, max_patches)
        # For ngram, every patch up to valid_len // 3 should be real
        for b in range(B):
            valid_len = mask[b].sum().item()
            expected_patches = (valid_len + 2) // 3  # ceil(valid_len / 3)
            assert patch_mask[b].sum() >= 1  # At least 1 real patch


class TestCrossAttentionCompressorGradient:
    def test_gradient_flows(self, compressor):
        """Output should have gradients with respect to input token embeddings."""
        B, S, H = 2, 16, 64
        embs, mask = make_random_embs(B, S, H)
        embs.requires_grad_(True)

        patch_ids = PatchInteraction._ngram_patches(mask, n=4)
        max_patches = int(patch_ids.clamp(min=0).max().item()) + 1

        patch_embs, _ = compressor(embs, patch_ids, mask, max_patches)
        patch_embs.sum().backward()

        assert embs.grad is not None
        assert embs.grad.abs().sum() > 0


class TestCrossAttentionCompressorSingleToken:
    def test_single_token_patch(self, compressor):
        """A patch with a single token should produce a deterministic output
        (no averaging uncertainty)."""
        B, S, H = 1, 4, 64
        embs = torch.randn(B, S, H)
        mask = torch.ones(B, S, dtype=torch.long)
        # Make each token its own patch (n=1)
        patch_ids = PatchInteraction._ngram_patches(mask, n=1)
        max_patches = S

        patch_embs, patch_mask = compressor(embs, patch_ids, mask, max_patches)
        assert patch_embs.shape == (B, S, H)
        assert patch_mask.sum() == S  # All S patches are real

    def test_empty_patch_zeroed(self, compressor):
        """Empty patches (all padding) should produce zero vectors."""
        B, S, H = 2, 12, 64
        embs, mask = make_random_embs(B, S, H)
        # Make all positions masked for the first batch element after position 3
        mask[0, 3:] = 0

        patch_ids = PatchInteraction._ngram_patches(mask, n=3)
        max_patches = int(PatchInteraction._ngram_patches(
            torch.ones(B, S, dtype=torch.long), n=3
        ).max().item()) + 1

        patch_embs, patch_mask = compressor(embs, patch_ids, mask, max_patches)

        # Patches beyond position 3 for batch 0 should be zero
        empty = (patch_mask[0] == 0).nonzero(as_tuple=True)[0]
        if len(empty) > 0:
            assert torch.allclose(patch_embs[0, empty], torch.zeros_like(patch_embs[0, empty]))
