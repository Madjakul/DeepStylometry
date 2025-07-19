# tests/test_margin_loss.py

import pytest
import torch

from deep_stylometry.modules.margin_loss import MarginLoss
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
    cfg.model.distance_weightning = "none"
    cfg.train.margin = 0.5
    loss_fn = MarginLoss(cfg)

    all_scores, targets, loss = loss_fn(q_embs, k_embs, q_mask, k_mask)

    assert not torch.isnan(loss), "InfoNCE loss is NaN"
    assert loss >= 0
    assert all_scores.shape == (2, 6), "All scores shape mismatch"
    assert targets.shape == (2,), "Targets shape mismatch"


def test_li_exp_decay_forward(q_embs, k_embs, q_mask, k_mask):
    cfg = BaseConfig()
    cfg.model.pooling_method = "li"
    cfg.model.use_softmax = False
    cfg.model.distance_weightning = "exp"
    cfg.model.alpha = 0.1
    cfg.train.margin = 0.5
    cfg.data.max_length = 128
    loss_fn = MarginLoss(cfg)
    all_scores, targets, loss = loss_fn(q_embs, k_embs, q_mask, k_mask)
    assert not torch.isnan(loss), "InfoNCE loss is NaN"
    assert loss >= 0
    assert all_scores.shape == (2, 6), "All scores shape mismatch"
    assert targets.shape == (2,), "Targets shape mismatch"


def test_li_linear_decay_forward(q_embs, k_embs, q_mask, k_mask):
    cfg = BaseConfig()
    cfg.model.pooling_method = "li"
    cfg.model.use_softmax = False
    cfg.model.distance_weightning = "linear"
    cfg.model.alpha = 0.1
    cfg.train.margin = 0.5
    cfg.data.max_length = 128
    loss_fn = MarginLoss(cfg)
    all_scores, targets, loss = loss_fn(q_embs, k_embs, q_mask, k_mask)
    assert not torch.isnan(loss), "InfoNCE loss is NaN"
    assert loss >= 0
    assert all_scores.shape == (2, 6), "All scores shape mismatch"
    assert targets.shape == (2,), "Targets shape mismatch"


def test_li_softmax_forward(q_embs, k_embs, q_mask, k_mask):
    cfg = BaseConfig()
    cfg.model.pooling_method = "li"
    cfg.model.use_softmax = True
    cfg.model.distance_weightning = "none"
    cfg.train.margin = 0.5
    loss_fn = MarginLoss(cfg)
    all_scores, targets, loss = loss_fn(q_embs, k_embs, q_mask, k_mask)
    assert not torch.isnan(loss), "InfoNCE loss is NaN"
    assert loss >= 0
    assert all_scores.shape == (2, 6), "All scores shape mismatch"
    assert targets.shape == (2,), "Targets shape mismatch"


def test_li_gumbel_softmax_forward(q_embs, k_embs, q_mask, k_mask):
    cfg = BaseConfig()
    cfg.model.pooling_method = "li"
    cfg.model.use_softmax = True
    cfg.model.distance_weightning = "none"
    cfg.model.initial_gumbel_temp = 0.5
    cfg.train.margin = 0.5
    loss_fn = MarginLoss(cfg)
    all_scores, targets, loss = loss_fn(q_embs, k_embs, q_mask, k_mask, gumbel_temp=0.5)
    assert not torch.isnan(loss), "InfoNCE loss is NaN"
    assert loss >= 0
    assert all_scores.shape == (2, 6), "All scores shape mismatch"
    assert targets.shape == (2,), "Targets shape mismatch"


def test_mean_pooling_forward(q_embs, k_embs, q_mask, k_mask):
    cfg = BaseConfig()
    cfg.model.pooling_method = "mean"
    cfg.train.margin = 0.5
    loss_fn = MarginLoss(cfg)
    all_scores, targets, loss = loss_fn(q_embs, k_embs, q_mask, k_mask)
    assert not torch.isnan(loss), "InfoNCE loss is NaN"
    assert loss >= 0
    assert all_scores.shape == (2, 6), "All scores shape mismatch"
    assert targets.shape == (2,), "Targets shape mismatch"
