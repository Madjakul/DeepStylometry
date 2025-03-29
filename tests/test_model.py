# tests/test_model.py

import torch

from deep_stylometry.modules.modeling_deep_stylometry import DeepStylometry


def test_model_forward():
    model = DeepStylometry(
        optim_name="adamw",
        base_model_name="gpt2",
        batch_size=2,
        seq_len=128,
    )
    input_ids = torch.randint(0, 50257, (2, 128))
    attention_mask = torch.ones(2, 128)

    embs, logits = model(input_ids, attention_mask)
    assert embs.shape == (2, 128, model.lm.hidden_size)
    assert logits.shape == (2, 128, model.lm.vocab_size)


def test_model_training_step():
    model = DeepStylometry(
        optim_name="adamw",
        base_model_name="gpt2",
        batch_size=2,
        seq_len=128,
    )
    batch = {
        "q_input_ids": torch.randint(0, 50257, (2, 128)),
        "k_input_ids": torch.randint(0, 50257, (2, 128)),
        "q_attention_mask": torch.ones(2, 128),
        "k_attention_mask": torch.ones(2, 128),
        "pair_labels": torch.tensor([1, 0]),
    }

    loss = model.training_step(batch, 0)
    assert not torch.isnan(loss)
