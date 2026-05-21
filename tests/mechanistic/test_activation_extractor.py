# deep_stylometry/experiments/mechanistic/tests/test_activation_extractor.py
"""Tests for the activation extractor."""

import numpy as np
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Toy model
# ---------------------------------------------------------------------------

class _ToyEncoder(nn.Module):
    """Minimal two-layer encoder that returns output_hidden_states."""

    def __init__(self, hidden: int = 10):
        super().__init__()
        self.hidden = hidden
        self.layers = nn.ModuleList([
            nn.Linear(hidden, hidden),
            nn.Linear(hidden, hidden),
        ])

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, return_dict=False):
        B, S = input_ids.shape
        h = torch.zeros(B, S, self.hidden)
        for i, lid in enumerate(input_ids.tolist()):
            for j, tok_id in enumerate(lid):
                h[i, j] = float(tok_id % 100) / 100.0

        all_hs = [h.clone()]
        for layer in self.layers:
            h = torch.relu(layer(h))
            all_hs.append(h.clone())

        class _Out:
            pass
        out = _Out()
        out.last_hidden_state = h
        out.hidden_states = tuple(all_hs)
        return out


class _ToyModel:
    """Tiny wrapper that mimics DeepStylometry's interface."""
    def __init__(self, hidden: int = 10):
        self.lm = type("lm", (), {"model": _ToyEncoder(hidden)})()
        head = nn.Sequential(nn.Linear(hidden, hidden))
        self.head = head

    def eval(self):
        self.head.eval()
        return self

    def to(self, device):
        return self


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestActivationExtractor:
    def _make_tokenizer(self):
        """Minimal mock tokenizer returning a BatchEncoding-like object."""
        class _TokOut(dict):
            def to(self, device):
                return {k: v.to(device) for k, v in self.items()}

        class _Tok:
            def __call__(self, texts, truncation=True, max_length=512,
                         padding=True, return_tensors="pt", add_special_tokens=True):
                B = len(texts)
                S = 8
                return _TokOut({
                    "input_ids": torch.randint(1, 100, (B, S)),
                    "attention_mask": torch.ones(B, S, dtype=torch.long),
                })
        return _Tok()

    def test_output_shape(self):
        from deep_stylometry.experiments.mechanistic.activation_extractor import (
            extract_hidden_states_ds,
        )
        model = _ToyModel(hidden=10)
        tok = self._make_tokenizer()
        texts = ["hello world"] * 4
        device = torch.device("cpu")

        hidden, post_head = extract_hidden_states_ds(
            model, texts, tok, device, batch_size=4
        )

        # 3 patch points: embedding + 2 layers
        assert hidden.shape == (4, 3, 10), f"Expected (4, 3, 10), got {hidden.shape}"
        assert post_head.shape == (4, 10)

    def test_deterministic(self):
        from deep_stylometry.experiments.mechanistic.activation_extractor import (
            extract_hidden_states_ds,
        )

        class _FixedTok:
            """Returns the same fixed token ids for every call."""
            def __call__(self, texts, **kwargs):
                B = len(texts)
                S = 8
                ids = torch.arange(1, S + 1).unsqueeze(0).expand(B, -1).clone()
                mask = torch.ones(B, S, dtype=torch.long)
                class _Out(dict):
                    def to(self, device): return {k: v.to(device) for k, v in self.items()}
                return _Out({"input_ids": ids, "attention_mask": mask})

        torch.manual_seed(0)
        model = _ToyModel(hidden=10)
        tok = _FixedTok()
        texts = ["same text"] * 4
        device = torch.device("cpu")

        h1, _ = extract_hidden_states_ds(model, texts, tok, device)
        h2, _ = extract_hidden_states_ds(model, texts, tok, device)

        np.testing.assert_array_almost_equal(h1, h2, decimal=5)

    def test_masking(self):
        """Masked positions should not contribute to mean pool."""
        from deep_stylometry.experiments.mechanistic.activation_extractor import _mean_pool

        hidden = torch.ones(1, 5, 4)
        hidden[0, 3, :] = 999.0  # padding position
        hidden[0, 4, :] = 999.0
        mask = torch.tensor([[1, 1, 1, 0, 0]])

        pooled = _mean_pool(hidden, mask)
        # Should be mean of first 3 positions = 1.0
        np.testing.assert_array_almost_equal(pooled.numpy(), np.ones((1, 4)), decimal=5)
