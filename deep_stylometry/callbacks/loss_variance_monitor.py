# deep_stylometry/callbacks/variance_monitor.py

import logging
from collections import deque
from typing import Any, Deque, Optional

import lightning as L
import torch


class LossVarianceMonitor(L.Callback):
    """Monitor training stability via rolling loss variance and rolling query-
    length variance, logged at reduced frequency so wandb plots remain
    readable.

    Because both metrics are logged at the same global_step, wandb lets you build
    a custom scatter chart of loss_variance_rolling vs query_length_variance_rolling.

    Args:
        window_size:      Number of steps in the rolling window for both metrics.
        log_every_n_steps: Emit to the logger only every N steps (default: window_size).
                           Set independently of trainer.log_every_n_steps.
    """

    def __init__(self, window_size: int = 100, log_every_n_steps: Optional[int] = None):
        super().__init__()
        if window_size < 2:
            raise ValueError("window_size must be at least 2 to compute variance.")
        self.window_size = window_size
        self.log_every_n_steps = (
            log_every_n_steps if log_every_n_steps is not None else window_size
        )
        self.losses: Deque[torch.Tensor] = deque(maxlen=self.window_size)
        self.query_lengths: Deque[torch.Tensor] = deque(maxlen=self.window_size)

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        # --- accumulate loss ---
        if isinstance(outputs, dict):
            loss = outputs.get("loss")
            if loss is None:
                logging.warning(
                    "VarianceMonitor failed to find 'loss' in training_step outputs."
                )
                return
        else:
            loss = outputs
        self.losses.append(loss.detach().cpu())

        # --- accumulate per-sample query lengths for this batch ---
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            # store mean query length for the batch as a scalar representative
            batch_length_var = torch.var(attention_mask.sum(dim=1).float())
            self.query_lengths.append(batch_length_var.cpu())

        # --- gate logging ---
        if trainer.global_step % self.log_every_n_steps != 0:
            return
        if len(self.losses) < 2:
            return

        metrics: dict[str, torch.Tensor] = {}

        loss_tensor = torch.stack(list(self.losses))
        metrics["train/loss_variance_rolling"] = torch.var(loss_tensor)

        if len(self.query_lengths) >= 2:
            length_tensor = torch.stack(list(self.query_lengths))
            metrics["train/query_length_variance_rolling"] = length_tensor.mean()
            metrics["train/query_length_variance_rolling"] = torch.var(length_tensor)

        pl_module.log_dict(
            metrics,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )
