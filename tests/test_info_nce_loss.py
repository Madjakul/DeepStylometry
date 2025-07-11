# tests/test_info_nce_loss.py

import pytest
import torch

from deep_stylometry.modules.info_nce_loss import InfoNCELoss
from deep_stylometry.utils.configs import BaseConfig


@pytest.fixture
def q_embs():
    return torch.randn(2, 128, 768)


@pytest.fixture
def k_embs():
    return torch.randn(2 * 3, 128, 768)


@pytest.fixture
def q_mask():
    return torch.ones(2, 128)


@pytest.fixture
def k_mask():
    mask = torch.ones(2 * 3, 128)
    mask[:, :96] = 0
    return mask


def test_li_forward(q_embs, k_embs, q_mask, k_mask):
    cfg = BaseConfig()
    cfg.model.pooling_method = "li"
    cfg.model.use_softmax = False
    loss_fn = InfoNCELoss(cfg)

    all_scores, targets, loss = loss_fn(q_embs, k_embs, q_mask, k_mask)

    assert not torch.isnan(loss), "InfoNCE loss is NaN"
    assert loss >= 0
    assert all_scores.shape == (2, 6), "All scores shape mismatch"
    assert targets.shape == (2,), "Targets shape mismatch"
