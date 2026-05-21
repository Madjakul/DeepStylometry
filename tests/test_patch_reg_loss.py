# tests/test_patch_reg_loss.py

import pytest
import torch

from deep_stylometry.modules.patch_interaction import PatchInteraction
from deep_stylometry.utils.configs import BaseConfig


def _make_learned_pli(cfg=None):
    if cfg is None:
        cfg = BaseConfig()
        cfg.model.pooling_method = "pli"
        cfg.model.patch_method = "learned"
        cfg.model.patch_compression = "mean"
        cfg.model.skip_list = False
        cfg.model.dropout = 0.1
        cfg.model.expansion_ratio = 1
        cfg.model.lm_hidden_size = 64
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
    return PatchInteraction(cfg)


def _make_ngram_pli():
    cfg = BaseConfig()
    cfg.model.pooling_method = "pli"
    cfg.model.patch_method = "ngram"
    cfg.model.patch_size = 3
    cfg.model.patch_compression = "mean"
    cfg.model.skip_list = False
    cfg.model.lm_hidden_size = 64
    cfg.model.patch_lambda = 0.1
    cfg.train.tau = 0.05
    cfg.train.loss = "info_nce"
    cfg.train.gather = False
    cfg.train.precision = "32"
    cfg.data.batch_size = 4
    return PatchInteraction(cfg)


def _p_target():
    cfg = BaseConfig()
    return cfg.model.patch_target_rate  # 1/3


def test_interior_minimum():
    """Loss is ~0 when mean cut rate equals p_target exactly."""
    pli = _make_learned_pli()
    p = _p_target()
    B, S = 4, 16
    pli._last_cut_probs = torch.full((B, S), p)
    pli._last_q_mask = torch.ones(B, S, dtype=torch.long)
    loss = pli.get_patch_reg_loss()
    assert loss is not None
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_pulls_back_from_saturation():
    """Loss is (0.999 - 1/3)^2 when cut_probs are at near-1 (failure mode)."""
    pli = _make_learned_pli()
    p = _p_target()
    B, S = 4, 16
    q = 0.999
    pli._last_cut_probs = torch.full((B, S), q)
    pli._last_q_mask = torch.ones(B, S, dtype=torch.long)
    loss = pli.get_patch_reg_loss()
    expected = (q - p) ** 2
    assert loss.item() == pytest.approx(expected, rel=1e-4)


def test_pulls_back_from_collapse():
    """Loss is (0.001 - 1/3)^2 when cut_probs are near-0 (collapse mode)."""
    pli = _make_learned_pli()
    p = _p_target()
    B, S = 4, 16
    q = 0.001
    pli._last_cut_probs = torch.full((B, S), q)
    pli._last_q_mask = torch.ones(B, S, dtype=torch.long)
    loss = pli.get_patch_reg_loss()
    expected = (q - p) ** 2
    assert loss.item() == pytest.approx(expected, rel=1e-4)


def test_symmetry():
    """Loss at p_target+δ equals loss at p_target-δ (symmetric around minimum)."""
    pli = _make_learned_pli()
    p = _p_target()
    delta = 0.1
    B, S = 4, 16
    mask = torch.ones(B, S, dtype=torch.long)

    pli._last_cut_probs = torch.full((B, S), p + delta)
    pli._last_q_mask = mask
    loss_high = pli.get_patch_reg_loss().item()

    pli._last_cut_probs = torch.full((B, S), p - delta)
    pli._last_q_mask = mask
    loss_low = pli.get_patch_reg_loss().item()

    assert loss_high == pytest.approx(loss_low, rel=1e-5)


def test_mask_handling():
    """Masked positions do not contribute; result is ~0 for unmasked q=p_target."""
    pli = _make_learned_pli()
    p = _p_target()
    B, S = 4, 16
    half = S // 2

    # Unmasked half has q = p_target; masked half has q = 0.999
    q = torch.full((B, S), 0.999)
    q[:, :half] = p
    mask = torch.zeros(B, S, dtype=torch.long)
    mask[:, :half] = 1  # Only first half is valid

    pli._last_cut_probs = q
    pli._last_q_mask = mask
    loss = pli.get_patch_reg_loss()
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_gradient_flows_correctly():
    """Gradient at each sequence points toward p_target.

    Seq 0: all q > p_target  →  mean > p_target  →  grad > 0 (descent lowers q)
    Seq 1: all q < p_target  →  mean < p_target  →  grad < 0 (descent raises q)
    """
    pli = _make_learned_pli()
    p = _p_target()
    S = 8

    q_vals = torch.tensor(
        [
            [p + 0.3] * S,
            [p - 0.2] * S,
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    mask = torch.ones(2, S, dtype=torch.long)

    pli._last_cut_probs = q_vals
    pli._last_q_mask = mask
    loss = pli.get_patch_reg_loss()
    loss.backward()

    assert q_vals.grad is not None
    assert (q_vals.grad[0] > 0).all(), "grad should be positive for q > p_target"
    assert (q_vals.grad[1] < 0).all(), "grad should be negative for q < p_target"


def test_returns_none_for_ngram():
    """Non-learned patching never sets cut_probs; regulariser returns None."""
    pli = _make_ngram_pli()
    # No forward pass — _last_cut_probs stays None
    assert pli.get_patch_reg_loss() is None
