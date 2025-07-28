# tests/test_deep_stylometry.py

from unittest.mock import MagicMock, patch

import lightning as L
import pytest
import torch

from deep_stylometry.modules.info_nce_loss import InfoNCELoss
from deep_stylometry.modules.modeling_deep_stylometry import DeepStylometry
from deep_stylometry.modules.triplet_loss import TripletLoss
from deep_stylometry.utils.configs import BaseConfig


@pytest.fixture
def cfg():
    """Provides a default configuration."""
    config = BaseConfig()
    config.data.batch_size = 2
    config.data.max_length = 64
    config.model.base_model_name = "prajjwal1/bert-tiny"
    config.model.add_linear_layers = False
    config.train.loss = "info_nce"
    config.train.lr = 1e-4
    config.train.weight_decay = 1e-2
    config.train.betas = (0.9, 0.999)
    config.train.eps = 1e-8
    config.train.num_cycles = 1.0
    config.model.auto_anneal_gumbel = False
    config.model.initial_gumbel_temp = None
    config.train.lm_loss_weight = 0.1
    return config


@pytest.fixture
def batch():
    """Provides a sample batch of data."""
    batch_size = 2
    seq_len = 64
    return {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "pos_input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
        "pos_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "neg_input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
        "neg_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "labels": torch.randint(0, 1000, (batch_size, seq_len)),
    }


def test_init(cfg):
    """Tests the initialization of the DeepStylometry model."""
    model = DeepStylometry(cfg)
    assert isinstance(model, L.LightningModule)
    assert model.cfg == cfg
    assert isinstance(model.contrastive_loss, InfoNCELoss)

    # Test with TripletLoss
    cfg.train.loss = "triplet"
    cfg.train.margin = 0.5
    model = DeepStylometry(cfg)
    assert isinstance(model.contrastive_loss, TripletLoss)

    # Test with linear layers
    cfg.model.add_linear_layers = True
    model = DeepStylometry(cfg)
    assert hasattr(model, "fc1")
    assert hasattr(model, "fc2")


def test_forward(cfg, batch):
    """Tests the forward pass of the model."""
    model = DeepStylometry(cfg)
    lm_loss, last_hidden_states, projected_embs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    assert isinstance(lm_loss, torch.Tensor)
    assert lm_loss > 0
    assert last_hidden_states.shape == (
        cfg.data.batch_size,
        cfg.data.max_length,
        model.lm.hidden_size,
    )
    assert projected_embs.shape == last_hidden_states.shape


def test_forward_with_linear_layers(cfg, batch):
    """Tests the forward pass with additional linear layers."""
    cfg.model.add_linear_layers = True
    model = DeepStylometry(cfg)
    lm_loss, last_hidden_states, projected_embs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    assert projected_embs.shape == last_hidden_states.shape
    # Check that the residual connection is there
    assert not torch.allclose(projected_embs, last_hidden_states)


def test_compute_losses(cfg, batch):
    """Tests the loss computation logic."""
    model = DeepStylometry(cfg)
    metrics = model._compute_losses(batch)

    assert "all_scores" in metrics
    assert "targets" in metrics
    assert "poss" in metrics
    assert "negs" in metrics
    assert "lm_loss" in metrics
    assert "contrastive_loss" in metrics
    assert "total_loss" in metrics

    assert metrics["all_scores"].shape == (cfg.data.batch_size, 2 * cfg.data.batch_size)
    assert metrics["targets"].shape == (cfg.data.batch_size,)
    assert not torch.isnan(metrics["total_loss"])
    assert metrics["total_loss"] >= 0


@patch("deep_stylometry.modules.modeling_deep_stylometry.DeepStylometry.log_dict")
def test_training_step(mock_log_dict, cfg, batch):
    """Tests a single training step."""
    model = DeepStylometry(cfg)
    loss = model.training_step(batch, 0)

    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
    mock_log_dict.assert_called()
    logged_data = mock_log_dict.call_args[0][0]
    assert "train_total_loss" in logged_data
    assert "train_lm_loss" in logged_data
    assert "train_contrastive_loss" in logged_data


@patch("deep_stylometry.modules.modeling_deep_stylometry.DeepStylometry.log_dict")
def test_validation_step(mock_log_dict, cfg, batch):
    """Tests a single validation step."""
    model = DeepStylometry(cfg)
    model.val_auroc = MagicMock()
    model.val_hr1 = MagicMock()
    model.val_hr5 = MagicMock()
    model.val_hr10 = MagicMock()
    model.val_rr = MagicMock()

    model.validation_step(batch, 0)

    model.val_auroc.update.assert_called_once()
    model.val_hr1.update.assert_called_once()
    model.val_hr5.update.assert_called_once()
    model.val_hr10.update.assert_called_once()
    model.val_rr.update.assert_called_once()
    mock_log_dict.assert_called()


@patch("deep_stylometry.modules.modeling_deep_stylometry.DeepStylometry.log_dict")
def test_test_step(mock_log_dict, cfg, batch):
    """Tests a single test step."""
    model = DeepStylometry(cfg)
    model.test_auroc = MagicMock()
    model.test_hr1 = MagicMock()
    model.test_hr5 = MagicMock()
    model.test_hr10 = MagicMock()
    model.test_rr = MagicMock()

    model.test_step(batch, 0)

    model.test_auroc.update.assert_called_once()
    model.test_hr1.update.assert_called_once()
    model.test_hr5.update.assert_called_once()
    model.test_hr10.update.assert_called_once()
    model.test_rr.update.assert_called_once()
    mock_log_dict.assert_called()


def test_configure_optimizers(cfg):
    """Tests the optimizer and scheduler configuration."""
    model = DeepStylometry(cfg)
    # Mock the trainer attribute that is accessed in configure_optimizers
    model.trainer = MagicMock()
    model.trainer.estimated_stepping_batches = 1000

    optimizers = model.configure_optimizers()

    assert "optimizer" in optimizers
    assert "lr_scheduler" in optimizers
    assert isinstance(optimizers["optimizer"], torch.optim.AdamW)
    assert optimizers["optimizer"].defaults["lr"] == cfg.train.lr
    assert optimizers["lr_scheduler"]["interval"] == "step"
