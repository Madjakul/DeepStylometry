# tests/test_modeling.py

from unittest.mock import MagicMock, patch

import pytest
import torch

from deep_stylometry.modules.modeling_deep_stylometry import DeepStylometry


@pytest.fixture
def mock_model(dummy_cfg):
    """Create a DeepStylometry with a mocked LanguageModel so that
    ModernBERT weights do not have to be loaded into memory."""
    H = 64  # Must match dummy_cfg.model.lm_hidden_size

    mock_hf_config = MagicMock()
    mock_hf_config.hidden_size = H
    mock_hf_config.vocab_size = 30522

    def _fake_hf_forward(input_ids, attention_mask, return_dict=True):
        B, S = input_ids.shape
        out = MagicMock()
        out.last_hidden_state = torch.randn(B, S, H)
        return out

    mock_hf_model = MagicMock(side_effect=_fake_hf_forward)
    mock_hf_model.config = mock_hf_config

    import deep_stylometry.modules.language_model as lm_mod

    with patch.object(lm_mod, "AutoConfig") as mock_auto_config, \
         patch.object(lm_mod, "AutoModel") as mock_auto_model:
        mock_auto_config.from_pretrained.return_value = mock_hf_config
        mock_auto_model.from_pretrained.return_value = mock_hf_model

        model = DeepStylometry(dummy_cfg)

    return model


class TestDeepStylometryForward:
    def test_forward_shape(self, mock_model, dummy_batch):
        """forward(input_ids, attention_mask) → (B, S, H)."""
        out = mock_model(
            input_ids=dummy_batch["input_ids"],
            attention_mask=dummy_batch["attention_mask"],
        )
        B, S = dummy_batch["input_ids"].shape
        assert out.shape == (B, S, 64)

    def test_forward_finite(self, mock_model, dummy_batch):
        out = mock_model(
            input_ids=dummy_batch["input_ids"],
            attention_mask=dummy_batch["attention_mask"],
        )
        assert out.isfinite().all()


class TestDeepStylometryTestStep:
    def test_test_step_returns_expected_keys(self, mock_model, dummy_batch):
        out = mock_model.test_step(dummy_batch, 0)
        for key in ("q_embs", "q_mask", "q_input_ids",
                    "pos_embs", "pos_mask", "pos_input_ids",
                    "neg_embs", "neg_mask", "neg_input_ids"):
            assert key in out, f"Missing key: {key}"

    def test_test_step_emb_shape(self, mock_model, dummy_batch):
        out = mock_model.test_step(dummy_batch, 0)
        B, S = dummy_batch["input_ids"].shape
        assert out["q_embs"].shape == (B, S, 64)
        assert out["pos_embs"].shape == (B, S, 64)
        assert out["neg_embs"].shape == (B, S, 64)
