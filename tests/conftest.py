# tests/conftest.py

import torch
import pytest

from deep_stylometry.utils.configs import BaseConfig


@pytest.fixture
def dummy_cfg():
    """Minimal BaseConfig for unit tests.

    - Uses pooling_method="li" (LateInteraction) by default.
    - skip_list=False avoids tokeniser loading in LateInteraction.__init__.
    - expansion_ratio=1 keeps head small.
    - lm_hidden_size=64 so PatchBoundaryPredictor / CrossAttentionCompressor
      use a compact architecture.
    """
    cfg = BaseConfig()
    cfg.model.base_checkpoint = "answerdotai/ModernBERT-base"
    cfg.model.pooling_method = "li"
    cfg.model.skip_list = False
    cfg.model.dropout = 0.1
    cfg.model.expansion_ratio = 1
    cfg.model.lm_hidden_size = 64
    cfg.model.patch_method = "ngram"
    cfg.model.patch_size = 3
    cfg.model.patch_compression = "mean"
    cfg.model.patch_cross_attn_dim = 16
    cfg.model.patch_cross_attn_heads = 2
    cfg.model.patch_lambda = 0.1
    cfg.model.gumbel_tau_init = 1.0
    cfg.model.gumbel_tau_final = 0.1
    cfg.model.gumbel_anneal_steps = 100
    cfg.train.tau = 0.05
    cfg.train.loss = "info_nce"
    cfg.train.gather = False
    cfg.train.precision = "32"
    cfg.data.batch_size = 4
    return cfg


@pytest.fixture
def pli_cfg(dummy_cfg):
    """Config with PLI (ngram) enabled."""
    dummy_cfg.model.pooling_method = "pli"
    dummy_cfg.model.patch_method = "ngram"
    dummy_cfg.model.patch_size = 3
    return dummy_cfg


@pytest.fixture
def learned_pli_cfg(dummy_cfg):
    """Config with PLI (learned) enabled."""
    dummy_cfg.model.pooling_method = "pli"
    dummy_cfg.model.patch_method = "learned"
    dummy_cfg.model.patch_compression = "mean"
    return dummy_cfg


@pytest.fixture
def dummy_batch():
    """Random batch matching the training collator output format.

    Shapes:
        input_ids, pos_input_ids, neg_input_ids: (4, 32)
        attention_mask, pos_attention_mask, neg_attention_mask: (4, 32)
        index: (4,)
    """
    B, S = 4, 32
    valid_len = 20  # Number of real tokens (last 12 are padding)

    def _ids_and_mask():
        ids = torch.randint(0, 30000, (B, S))
        mask = torch.zeros(B, S, dtype=torch.long)
        mask[:, :valid_len] = 1
        # Ensure first token is CLS (50281) and last valid is SEP (50282)
        ids[:, 0] = 50281
        ids[:, valid_len - 1] = 50282
        ids[:, valid_len:] = 50283  # PAD
        return ids, mask

    q_ids, q_mask = _ids_and_mask()
    pos_ids, pos_mask = _ids_and_mask()
    neg_ids, neg_mask = _ids_and_mask()

    return {
        "input_ids": q_ids,
        "attention_mask": q_mask,
        "pos_input_ids": pos_ids,
        "pos_attention_mask": pos_mask,
        "neg_input_ids": neg_ids,
        "neg_attention_mask": neg_mask,
        "index": torch.arange(B),
    }


def make_random_embs(batch: int, seq: int, hidden: int):
    """Return random float32 embeddings and a binary mask."""
    embs = torch.randn(batch, seq, hidden)
    mask = torch.zeros(batch, seq, dtype=torch.long)
    valid = max(1, seq - 4)  # Last 4 positions are padding
    mask[:, :valid] = 1
    return embs, mask


@pytest.fixture
def random_embs():
    """Factory fixture: ``random_embs(B, S, H) → (embs, mask)``."""
    return make_random_embs
