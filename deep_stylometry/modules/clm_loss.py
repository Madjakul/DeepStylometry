# deep_stylometry/modules/clm_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLMLoss(nn.Module):
    """Causal language modeling loss."""

    def __init__(self, vocab_size, ignore_index=-100):
        """Initialize the CLM loss.

        Args:
            vocab_size: Size of vocabulary
            ignore_index: Index to ignore in loss computation
        """
        super(CLMLoss, self).__init__()
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits, labels, attention_mask=None):
        """Compute causal language modeling loss.

        Args:
            logits: Prediction logits [batch_size, seq_len, vocab_size]
            labels: Target labels [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            CLM loss value
        """
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        if attention_mask is not None:
            # Create a mask to only include tokens where we have attention
            shift_mask = attention_mask[..., 1:].contiguous()
            active_loss = shift_mask.view(-1) == 1
            active_logits = shift_logits.view(-1, shift_logits.size(-1))[active_loss]
            active_labels = shift_labels.view(-1)[active_loss]
            loss = self.loss_fn(active_logits, active_labels)
        else:
            loss = self.loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        return loss
