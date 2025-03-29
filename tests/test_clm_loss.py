# tests/test_clm_loss.py

import torch

from deep_stylometry.modules.clm_loss import CLMLoss


def test_clm_loss():
    clm_loss = CLMLoss()
    logits = torch.randn(2, 5, 10)  # batch_size=2, seq_len=5, vocab_size=10
    input_ids = torch.randint(0, 10, (2, 5))
    attention_mask = torch.ones(2, 5)
    loss = clm_loss(logits, input_ids, attention_mask)

    assert not torch.isnan(loss)
    assert loss > 0
