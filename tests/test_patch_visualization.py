# tests/test_patch_visualization.py
"""Tests for the patch interaction visualisation helpers."""

import pytest
import torch

from deep_stylometry.experiments.patch_interactions import (
    compute_patch_alignments,
    compute_patch_stats,
    generate_pair_html,
    get_patch_spans,
)


class TestComputePatchAlignments:
    def test_output_shapes(self):
        Pq, Pk, H = 5, 7, 64
        q = torch.randn(1, Pq, H)
        k = torch.randn(1, Pk, H)
        q_mask = torch.ones(1, Pq, dtype=torch.long)
        k_mask = torch.ones(1, Pk, dtype=torch.long)

        scores, align, total = compute_patch_alignments(q, k, q_mask, k_mask)

        assert scores.shape == (Pq,)
        assert align.shape == (Pq,)
        assert total.ndim == 0  # scalar

    def test_align_indices_in_range(self):
        Pq, Pk, H = 4, 6, 32
        q = torch.randn(1, Pq, H)
        k = torch.randn(1, Pk, H)
        q_mask = torch.ones(1, Pq, dtype=torch.long)
        k_mask = torch.ones(1, Pk, dtype=torch.long)

        _, align, _ = compute_patch_alignments(q, k, q_mask, k_mask)
        assert (align >= 0).all()
        assert (align < Pk).all()

    def test_every_query_patch_has_alignment(self):
        Pq, Pk, H = 6, 4, 32
        q = torch.randn(1, Pq, H)
        k = torch.randn(1, Pk, H)
        q_mask = torch.ones(1, Pq, dtype=torch.long)
        k_mask = torch.ones(1, Pk, dtype=torch.long)

        _, align, _ = compute_patch_alignments(q, k, q_mask, k_mask)
        # Every query patch should be aligned to some doc patch
        assert len(align) == Pq

    def test_sim_stays_3d(self):
        """Masking must not inflate sim to 4D (old bug: unsqueeze(0).unsqueeze(2)
        on a (1,Pk) mask produced (1,1,1,Pk) which broadcast (1,Pq,Pk) to 4D).
        scores and align must remain 1D (Pq,), not 2D."""
        Pq, Pk, H = 5, 7, 64
        q = torch.randn(1, Pq, H)
        k = torch.randn(1, Pk, H)
        q_mask = torch.ones(1, Pq, dtype=torch.long)
        k_mask = torch.ones(1, Pk, dtype=torch.long)
        k_mask[:, 4:] = 0  # partial padding

        scores, align, total = compute_patch_alignments(q, k, q_mask, k_mask)
        assert scores.dim() == 1, f"scores should be 1D, got shape {scores.shape}"
        assert align.dim() == 1, f"align should be 1D, got shape {align.shape}"
        assert scores.shape[0] == Pq
        assert align.shape[0] == Pq

    def test_padded_key_patches_not_selected(self):
        """Padded key patches (mask=0) should never be the best match."""
        Pq, Pk, H = 3, 6, 32
        q = torch.randn(1, Pq, H)
        k = torch.randn(1, Pk, H)
        q_mask = torch.ones(1, Pq, dtype=torch.long)
        k_mask = torch.ones(1, Pk, dtype=torch.long)
        # Mark last 3 key patches as padding
        k_mask[:, 3:] = 0

        _, align, _ = compute_patch_alignments(q, k, q_mask, k_mask)
        # All alignments should be in [0, 2]
        assert (align < 3).all()


class TestGetPatchSpans:
    def test_basic_spans(self):
        tokens = ["[CLS]", "hello", "Ġworld", "Ġtest", "[SEP]", "[PAD]"]
        patch_ids = torch.tensor([0, 1, 2, 3, 4, -1])
        mask = torch.tensor([1, 1, 1, 1, 1, 0])

        spans = get_patch_spans(tokens, patch_ids, mask)
        assert len(spans) == 5  # 5 valid patches
        assert spans[1] == [1]  # "hello" is patch 1

    def test_multi_token_patch(self):
        tokens = ["[CLS]", "run", "##ning", "Ġtest", "[SEP]"]
        patch_ids = torch.tensor([0, 1, 1, 2, 3])
        mask = torch.ones(5, dtype=torch.long)

        spans = get_patch_spans(tokens, patch_ids, mask)
        assert 1 in [p for patches in spans for p in patches if len(
            [pp for pp in spans if 1 in pp or 2 in pp]
        )]  # Tokens 1 and 2 are in same patch
        # Patch 1 contains tokens 1 and 2
        assert 1 in spans[1] and 2 in spans[1]


class TestComputePatchStats:
    def test_empty(self):
        stats = compute_patch_stats([])
        assert stats == {}

    def test_basic_stats(self):
        lengths = [1, 2, 3, 3, 3, 4, 5]
        stats = compute_patch_stats(lengths)
        assert "mean" in stats
        assert "median" in stats
        assert "most_common" in stats
        assert stats["mean"] == pytest.approx(sum(lengths) / len(lengths))


class TestGeneratePairHTML:
    def test_returns_string(self):
        tokens = ["[CLS]", "hello", "Ġworld", "[SEP]"]
        spans = [[0], [1], [2], [3]]
        q_scores = torch.tensor([0.9, 0.5, 0.3, 0.1])
        d_scores = [0.4, 0.6, 0.5, 0.2]
        align = torch.tensor([0, 1, 2, 3])
        mask = torch.ones(4, dtype=torch.long)

        html = generate_pair_html(
            q_tokens=tokens,
            d_tokens=tokens,
            q_spans=spans,
            d_spans=spans,
            q_scores=q_scores,
            d_scores=d_scores,
            align_1d=align,
            q_mask=mask,
            d_mask=mask,
            pos_score=0.8,
            neg_score=0.3,
            pair_idx="test_0",
        )
        assert isinstance(html, str)
        assert len(html) > 100  # Non-trivial output
        assert "test_0" in html

    def test_empty_tokens_returns_empty(self):
        html = generate_pair_html(
            q_tokens=[],
            d_tokens=[],
            q_spans=[],
            d_spans=[],
            q_scores=torch.tensor([]),
            d_scores=[],
            align_1d=torch.tensor([], dtype=torch.long),
            q_mask=torch.tensor([], dtype=torch.long),
            d_mask=torch.tensor([], dtype=torch.long),
            pos_score=0.5,
            neg_score=0.5,
            pair_idx="empty",
        )
        assert html == ""
