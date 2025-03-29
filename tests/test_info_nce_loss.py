# tets/test_info_nce_loss.py

import torch

from deep_stylometry.modules.info_nce_loss import InfoNCELoss


def test_info_nce_loss():
    batch_size, hidden = 2, 768
    loss_fn = InfoNCELoss(
        do_late_interaction=False,
        do_distance=False,
        exp_decay=False,
        seq_len=128,
    )
    q_embs = torch.randn(batch_size, 128, hidden)
    k_embs = torch.randn(batch_size, 128, hidden)
    labels = torch.tensor([1, 0])
    mask = torch.ones(batch_size, 128)

    loss = loss_fn(q_embs, k_embs, labels, mask, mask)
    assert not torch.isnan(loss), "InfoNCE loss is NaN"
    assert loss >= 0
