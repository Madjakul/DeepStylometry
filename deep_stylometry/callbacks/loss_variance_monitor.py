# deep_stylometry/callbacks/loss_variance_monitor.py

import logging
from collections import deque
from typing import Any, Deque, Optional

import lightning as L
import torch

logger = logging.getLogger(__name__)


class LossVarianceMonitor(L.Callback):
    """Tracks rolling variance of the training loss and query-length variance.

    Maintains a fixed-size deque of recent loss values and logs their variance
    every ``log_every_n_steps`` steps. Useful for detecting training instability
    or overfitting to short sequences.

    Parameters
    ----------
    window_size : int, optional
        Number of recent batches to include in the rolling window (default: 100).
    log_every_n_steps : int, optional
        How often to log; defaults to ``window_size`` if not set.
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
                logger.warning(
                    "VarianceMonitor failed to find 'loss' in training_step outputs."
                )
                return
        else:
            loss = outputs
        self.losses.append(loss.detach().cpu())

        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            # store mean query length for the batch as a scalar representative
            batch_length_var = torch.var(attention_mask.sum(dim=1).float())
            self.query_lengths.append(batch_length_var.cpu())

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
