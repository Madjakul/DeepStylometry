# tests/test_triplet_loss.py

import pytest
import torch

from deep_stylometry.modules.triplet_loss import TripletLoss
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
    loss_fn = TripletLoss(cfg)

    loss_metrics = loss_fn(q_embs, k_embs, q_mask, k_mask)

    assert not torch.isnan(loss_metrics["loss"]), "InfoNCE loss is NaN"
    assert loss_metrics["loss"] >= 0
    assert loss_metrics["all_scores"].shape == (2, 6), "All scores shape mismatch"
    assert loss_metrics["targets"].shape == (2,), "Targets shape mismatch"
    assert loss_metrics["poss"].shape == (2,), "Positive scores shape mismatch"
    assert loss_metrics["negs"].shape == (2,), "Negative scores shape mismatch"


def test_li_exp_decay_forward(q_embs, k_embs, q_mask, k_mask):
    cfg = BaseConfig()
    cfg.model.pooling_method = "li"
    cfg.model.use_softmax = False
    cfg.model.distance_weightning = "exp"
    cfg.model.alpha = 0.1
    cfg.train.margin = 0.5
    cfg.data.max_length = 128
    loss_fn = TripletLoss(cfg)
    loss_metrics = loss_fn(q_embs, k_embs, q_mask, k_mask)
    assert not torch.isnan(loss_metrics["loss"]), "InfoNCE loss is NaN"
    assert loss_metrics["loss"] >= 0
    assert loss_metrics["all_scores"].shape == (2, 6), "All scores shape mismatch"
    assert loss_metrics["targets"].shape == (2,), "Targets shape mismatch"
    assert loss_metrics["poss"].shape == (2,), "Positive scores shape mismatch"
    assert loss_metrics["negs"].shape == (2,), "Negative scores shape mismatch"


def test_li_linear_decay_forward(q_embs, k_embs, q_mask, k_mask):
    cfg = BaseConfig()
    cfg.model.pooling_method = "li"
    cfg.model.use_softmax = False
    cfg.model.distance_weightning = "linear"
    cfg.model.alpha = 0.1
    cfg.train.margin = 0.5
    cfg.data.max_length = 128
    loss_fn = TripletLoss(cfg)
    loss_metrics = loss_fn(q_embs, k_embs, q_mask, k_mask)
    assert not torch.isnan(loss_metrics["loss"]), "InfoNCE loss is NaN"
    assert loss_metrics["loss"] >= 0
    assert loss_metrics["all_scores"].shape == (2, 6), "All scores shape mismatch"
    assert loss_metrics["targets"].shape == (2,), "Targets shape mismatch"
    assert loss_metrics["poss"].shape == (2,), "Positive scores shape mismatch"
    assert loss_metrics["negs"].shape == (2,), "Negative scores shape mismatch"


def test_li_softmax_forward(q_embs, k_embs, q_mask, k_mask):
    cfg = BaseConfig()
    cfg.model.pooling_method = "li"
    cfg.model.use_softmax = True
    cfg.model.distance_weightning = "none"
    cfg.train.margin = 0.5
    loss_fn = TripletLoss(cfg)
    loss_metrics = loss_fn(q_embs, k_embs, q_mask, k_mask)
    assert not torch.isnan(loss_metrics["loss"]), "InfoNCE loss is NaN"
    assert loss_metrics["loss"] >= 0
    assert loss_metrics["all_scores"].shape == (2, 6), "All scores shape mismatch"
    assert loss_metrics["targets"].shape == (2,), "Targets shape mismatch"
    assert loss_metrics["poss"].shape == (2,), "Positive scores shape mismatch"
    assert loss_metrics["negs"].shape == (2,), "Negative scores shape mismatch"


def test_li_gumbel_softmax_forward(q_embs, k_embs, q_mask, k_mask):
    cfg = BaseConfig()
    cfg.model.pooling_method = "li"
    cfg.model.use_softmax = True
    cfg.model.distance_weightning = "none"
    cfg.model.initial_gumbel_temp = 0.5
    cfg.train.margin = 0.5
    loss_fn = TripletLoss(cfg)
    loss_metrics = loss_fn(q_embs, k_embs, q_mask, k_mask)
    assert not torch.isnan(loss_metrics["loss"]), "InfoNCE loss is NaN"
    assert loss_metrics["loss"] >= 0
    assert loss_metrics["all_scores"].shape == (2, 6), "All scores shape mismatch"
    assert loss_metrics["targets"].shape == (2,), "Targets shape mismatch"
    assert loss_metrics["poss"].shape == (2,), "Positive scores shape mismatch"
    assert loss_metrics["negs"].shape == (2,), "Negative scores shape mismatch"


def test_mean_pooling_forward(q_embs, k_embs, q_mask, k_mask):
    cfg = BaseConfig()
    cfg.model.pooling_method = "mean"
    cfg.train.margin = 0.5
    loss_fn = TripletLoss(cfg)
    loss_metrics = loss_fn(q_embs, k_embs, q_mask, k_mask)
    assert not torch.isnan(loss_metrics["loss"]), "InfoNCE loss is NaN"
    assert loss_metrics["loss"] >= 0
    assert loss_metrics["all_scores"].shape == (2, 6), "All scores shape mismatch"
    assert loss_metrics["targets"].shape == (2,), "Targets shape mismatch"
    assert loss_metrics["poss"].shape == (2,), "Positive scores shape mismatch"
    assert loss_metrics["negs"].shape == (2,), "Negative scores shape mismatch"
