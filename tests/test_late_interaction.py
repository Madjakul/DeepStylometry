# tests/test_late_interaction.py

import torch

from deep_stylometry.modules.late_interaction import LateInteraction


def test_late_interaction_distance_matrix():
    seq_len = 128
    module = LateInteraction(do_distance=True, exp_decay=True, seq_len=seq_len)
    assert module.distance.shape == (seq_len, seq_len)
    # Check device after moving
    if torch.cuda.is_available():
        module.to("cuda")
        assert module.distance.device.type == "cuda"


def test_late_interaction_forward():
    batch_size, seq_len, hidden = 2, 128, 768
    module = LateInteraction(do_distance=False, exp_decay=False, seq_len=seq_len)
    q = torch.randn(batch_size, seq_len, hidden)
    k = torch.randn(batch_size, seq_len, hidden)
    mask = torch.ones(batch_size, seq_len)

    scores = module(q, k, mask, mask, gumbel_temp=0.5)
    assert scores.shape == (batch_size, batch_size)
    assert not torch.any(torch.isnan(scores))
