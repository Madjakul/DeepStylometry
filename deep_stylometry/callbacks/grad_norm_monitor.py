# deep_stylometry/callbacks/grad_norm_monitor.py

from collections import deque
from typing import Deque

import torch
import torch.nn as nn
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only


class GradNormMonitor(Callback):
    """Log a rolling mean of the gradient norm at each optimizer step.

    Avoids per-step spikes collapsing the wandb visualization while still
    surfacing genuine instability trends.

    Args:
        window_size:       Number of optimizer steps in the rolling window.
        log_every_n_steps: Emit to the logger every N optimizer steps.
                           Defaults to window_size (non-overlapping windows).
    """

    def __init__(self, window_size: int = 100, log_every_n_steps: int | None = None):
        super().__init__()
        if window_size < 2:
            raise ValueError("window_size must be at least 2.")
        self.window_size = window_size
        self.log_every_n_steps = (
            log_every_n_steps if log_every_n_steps is not None else window_size
        )
        self._norms: Deque[float] = deque(maxlen=window_size)
        self._optimizer_step_count = 0
        self._pending_norm: float | None = None

    @rank_zero_only
    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        # Compute total grad norm without touching the gradients.
        # Avoids double-clipping when trainer.gradient_clip_val is also set.
        total_norm = self._compute_grad_norm(pl_module)
        self._pending_norm = total_norm

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._pending_norm is None:
            return  # no optimizer step happened this batch (gradient accumulation)

        self._norms.append(self._pending_norm)
        self._pending_norm = None
        self._optimizer_step_count += 1

        if self._optimizer_step_count % self.log_every_n_steps != 0:
            return
        if len(self._norms) < 2:
            return

        rolling_mean = sum(self._norms) / len(self._norms)
        pl_module.log(
            "train/grad_norm_rolling",
            rolling_mean,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=False,
        )

    @staticmethod
    def _compute_grad_norm(pl_module: nn.Module) -> float:
        total_norm_sq = sum(
            p.grad.data.norm(2).item() ** 2
            for p in pl_module.parameters()
            if p.grad is not None
        )
        return total_norm_sq**0.5
