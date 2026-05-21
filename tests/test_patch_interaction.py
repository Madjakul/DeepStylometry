# tests/test_patch_interaction.py

import pytest
import torch

from deep_stylometry.modules.patch_interaction import PatchInteraction
from tests.conftest import make_random_embs


def _make_pli(cfg):
    cfg.model.pooling_method = "pli"
    return PatchInteraction(cfg)


class TestPatchInteractionNgram:
    def test_output_shape(self, pli_cfg):
        pli = _make_pli(pli_cfg)
        q, q_mask = make_random_embs(4, 32, 64)
        k, k_mask = make_random_embs(8, 32, 64)
        scores = pli(q, k, q_mask, k_mask)
        assert scores.shape == (4, 8)

    def test_output_shape_various(self, pli_cfg):
        pli = _make_pli(pli_cfg)
        for bq, bk in [(2, 4), (3, 6), (1, 1)]:
            q, q_mask = make_random_embs(bq, 20, 64)
            k, k_mask = make_random_embs(bk, 20, 64)
            scores = pli(q, k, q_mask, k_mask)
            assert scores.shape == (bq, bk)

    def test_scores_finite(self, pli_cfg):
        pli = _make_pli(pli_cfg)
        q, q_mask = make_random_embs(4, 32, 64)
        k, k_mask = make_random_embs(8, 32, 64)
        scores = pli(q, k, q_mask, k_mask)
        assert scores.isfinite().all()

    def test_deterministic(self, pli_cfg):
        pli = _make_pli(pli_cfg)
        pli.eval()
        torch.manual_seed(42)
        q, q_mask = make_random_embs(3, 24, 64)
        k, k_mask = make_random_embs(6, 24, 64)
        s1 = pli(q, k, q_mask, k_mask).detach()
        s2 = pli(q, k, q_mask, k_mask).detach()
        assert torch.allclose(s1, s2)

    def test_ngram_1_equivalent_to_late_interaction(self, dummy_cfg):
        """NGram with n=1: each token is its own patch.  Scores should match
        LateInteraction scores (same MaxSim formulation)."""
        from deep_stylometry.modules.late_interaction import LateInteraction

        dummy_cfg.model.patch_size = 1
        pli_cfg = dummy_cfg
        pli_cfg.model.pooling_method = "pli"
        pli_cfg.model.patch_method = "ngram"
        pli_cfg.model.patch_compression = "mean"

        pli = PatchInteraction(pli_cfg)
        li = LateInteraction(dummy_cfg)

        torch.manual_seed(7)
        q, q_mask = make_random_embs(3, 16, 64)
        k, k_mask = make_random_embs(5, 16, 64)

        s_pli = pli(q, k, q_mask, k_mask).detach()
        s_li = li(q, k, q_mask, k_mask).detach()

        # Scores should be identical (same MaxSim with same normalised vectors)
        assert torch.allclose(s_pli, s_li, atol=1e-4), (
            f"Max diff: {(s_pli - s_li).abs().max()}"
        )


class TestPatchInteractionNgramPatchCount:
    def test_fewer_patches_than_tokens(self, pli_cfg):
        """With patch_size=4, the number of patches P < S for non-trivial S."""
        pli_cfg.model.patch_size = 4
        pli = _make_pli(pli_cfg)

        B, S, H = 2, 24, 64
        q, q_mask = make_random_embs(B, S, H)
        k, k_mask = make_random_embs(B, S, H)

        # Run forward to exercise compression
        scores = pli(q, k, q_mask, k_mask)
        assert scores.shape == (B, B)
        # Verify patches via _ngram_patches directly
        patch_ids = PatchInteraction._ngram_patches(q_mask, n=4)
        max_patches = patch_ids.clamp(min=0).max().item() + 1
        assert max_patches < S, f"Expected fewer patches than tokens; got {max_patches} for S={S}"


class TestPatchInteractionWhitespace:
    def test_whitespace_output_shape(self, dummy_cfg):
        dummy_cfg.model.pooling_method = "pli"
        dummy_cfg.model.patch_method = "whitespace"
        pli = PatchInteraction(dummy_cfg)

        # Use real tokenised input for a short sentence
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        text = ["Hello world test sentence.", "Another example text here."]
        enc = tok(text, return_tensors="pt", padding=True, truncation=True, max_length=32)

        B = enc["input_ids"].shape[0]
        S = enc["input_ids"].shape[1]
        H = 64

        q = torch.randn(B, S, H)
        k = torch.randn(B, S, H)

        scores = pli(q, k, enc["attention_mask"], enc["attention_mask"],
                     q_input_ids=enc["input_ids"], k_input_ids=enc["input_ids"])
        assert scores.shape == (B, B)
        assert scores.isfinite().all()

    def test_whitespace_fewer_patches_for_multiword(self, dummy_cfg):
        """Whitespace patching groups tokens into word-level patches.
        For a sentence with multiple words, P ≤ S."""
        dummy_cfg.model.pooling_method = "pli"
        dummy_cfg.model.patch_method = "whitespace"
        pli = PatchInteraction(dummy_cfg)

        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        enc = tok(["Hello world test sentence long enough."],
                  return_tensors="pt", truncation=True, max_length=32)
        ids = enc["input_ids"]
        mask = enc["attention_mask"]

        patch_ids = pli._whitespace_patches(ids, mask)
        n_patches = patch_ids.clamp(min=0).max().item() + 1
        n_tokens = mask.sum().item()
        assert n_patches <= n_tokens


    def test_whitespace_no_python_loop(self, dummy_cfg):
        """_whitespace_patches must use direct LUT buffers, not a stored tokenizer
        or isin-style lookup (LUT indexing is O(1) per token vs O(S log V))."""
        dummy_cfg.model.pooling_method = "pli"
        dummy_cfg.model.patch_method = "whitespace"
        pli = PatchInteraction(dummy_cfg)
        assert not hasattr(pli, "tokenizer") or pli.tokenizer is None, (
            "PatchInteraction should not keep a tokenizer instance after init"
        )
        assert hasattr(pli, "is_word_start_lut") and pli.is_word_start_lut is not None
        assert hasattr(pli, "is_special_lut") and pli.is_special_lut is not None
        # LUT should be 1-D boolean of size vocab_size
        assert pli.is_word_start_lut.dtype == torch.bool
        assert pli.is_word_start_lut.dim() == 1

    def test_whitespace_three_word_sentence(self, dummy_cfg):
        """'Hello world foo' should produce exactly 3 content patches
        (one per word) plus 2 special-token patches (CLS + SEP) = 5 patches."""
        dummy_cfg.model.pooling_method = "pli"
        dummy_cfg.model.patch_method = "whitespace"
        pli = PatchInteraction(dummy_cfg)

        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        enc = tok(["Hello world foo"], return_tensors="pt")
        patch_ids = pli._whitespace_patches(enc["input_ids"], enc["attention_mask"])
        n_patches = patch_ids.clamp(min=0).max().item() + 1
        # CLS(1) + Hello(1) + world(1) + foo(1) + SEP(1) = 5
        assert n_patches == 5, f"Expected 5 patches, got {n_patches}"

    def test_whitespace_padding_gives_minus_one(self, dummy_cfg):
        """Padding positions must have patch_id == -1."""
        dummy_cfg.model.pooling_method = "pli"
        dummy_cfg.model.patch_method = "whitespace"
        pli = PatchInteraction(dummy_cfg)

        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        texts = ["Hi", "Hello world test sentence long enough to force padding"]
        enc = tok(texts, return_tensors="pt", padding=True, max_length=32, truncation=True)
        patch_ids = pli._whitespace_patches(enc["input_ids"], enc["attention_mask"])
        pad_positions = enc["attention_mask"] == 0
        assert (patch_ids[pad_positions] == -1).all()


class TestPatchInteractionPaddingIndependence:
    def test_padding_does_not_affect_ngram_scores(self, pli_cfg):
        """Adding fully-masked padding positions should not change scores."""
        pli = _make_pli(pli_cfg)
        torch.manual_seed(3)

        B, S, H = 3, 20, 64
        q, q_mask = make_random_embs(B, S, H)
        k, k_mask = make_random_embs(5, S, H)

        s1 = pli(q, k, q_mask, k_mask).detach()

        # Extend with 4 padding positions
        extra_q = torch.zeros(B, 4, H)
        extra_k = torch.zeros(5, 4, H)
        q_ext = torch.cat([q, extra_q], dim=1)
        k_ext = torch.cat([k, extra_k], dim=1)
        q_mask_ext = torch.cat([q_mask, torch.zeros(B, 4, dtype=torch.long)], dim=1)
        k_mask_ext = torch.cat([k_mask, torch.zeros(5, 4, dtype=torch.long)], dim=1)

        s2 = pli(q_ext, k_ext, q_mask_ext, k_mask_ext).detach()
        assert torch.allclose(s1, s2, atol=1e-4)


class TestPatchInteractionInsideInfoNCE:
    def test_end_to_end_inside_loss(self, pli_cfg):
        """PatchInteraction should work inside InfoNCELoss."""
        from deep_stylometry.modules.info_nce_loss import InfoNCELoss

        loss_fn = InfoNCELoss(pli_cfg)
        B, S, H = 4, 32, 64
        q, q_mask = make_random_embs(B, S, H)
        k, k_mask = make_random_embs(2 * B, S, H)
        targets = torch.arange(B)
        out = loss_fn(q, k, q_mask, k_mask, targets)
        assert out["loss"].isfinite()
        assert out["all_scores"].shape == (B, 2 * B)


class TestPatchInteractionLearned:
    def test_learned_scores_finite(self, learned_pli_cfg):
        pli = _make_pli(learned_pli_cfg)
        q, q_mask = make_random_embs(4, 32, 64)
        k, k_mask = make_random_embs(8, 32, 64)
        scores = pli(q, k, q_mask, k_mask, step=0)
        assert scores.shape == (4, 8)
        assert scores.isfinite().all()

    def test_learned_patch_ids_non_decreasing(self, learned_pli_cfg):
        pli = _make_pli(learned_pli_cfg)
        q, q_mask = make_random_embs(4, 32, 64)
        patch_ids, _ = pli._compute_patches(q, q_mask, None, step=0, training=False)
        # patch_ids for valid tokens should be non-decreasing
        for b in range(4):
            valid = (q_mask[b] > 0).nonzero(as_tuple=True)[0]
            if len(valid) > 1:
                ids_valid = patch_ids[b, valid]
                diffs = ids_valid[1:] - ids_valid[:-1]
                assert (diffs >= 0).all(), f"patch_ids not non-decreasing at batch {b}"

    def test_learned_backward(self, learned_pli_cfg):
        """Gradients should flow to predictor parameters via the regulariser.

        Hard patch assignments (cumsum → one_hot) break the gradient path for
        the MaxSim score, but ``-log(sum(cut_probs))`` is differentiable and
        is the primary training signal for the boundary predictor.
        """
        pli = _make_pli(learned_pli_cfg)
        pli.train()
        q, q_mask = make_random_embs(2, 16, 64)
        k, k_mask = make_random_embs(4, 16, 64)
        pli(q, k, q_mask, k_mask, step=50)  # populates _last_cut_probs
        reg_loss = pli.get_patch_reg_loss()
        assert reg_loss is not None
        reg_loss.backward()
        pred_params = list(pli.predictor.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in pred_params)

    def test_learned_reg_loss_available(self, learned_pli_cfg):
        pli = _make_pli(learned_pli_cfg)
        pli.train()
        q, q_mask = make_random_embs(3, 24, 64)
        k, k_mask = make_random_embs(6, 24, 64)
        pli(q, k, q_mask, k_mask, step=10)
        reg = pli.get_patch_reg_loss()
        assert reg is not None
        assert reg.isfinite()
        assert reg.ndim == 0
