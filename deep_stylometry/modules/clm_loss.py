# deep_stylometry/modules/clm_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLMLoss(nn.Module):
    """Cross-entropy loss for causal language modeling."""

    def forward(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """Compute the cross-entropy loss for causal language modeling.

        Args:
            logits: torch.Tensor
                (batch_size, seq_len, vocab_size)
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)

        Returns:
            torch.Tensor: The cross-entropy loss.
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()
        shift_labels = shift_labels.masked_fill(~shift_mask.bool(), -100)
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
