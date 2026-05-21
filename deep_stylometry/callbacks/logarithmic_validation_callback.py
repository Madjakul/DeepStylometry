# deep_stylometry/callbacks/logarithmic_validation_callback.py

import logging
import math
from typing import Optional

import lightning as L
import torch

logger = logging.getLogger(__name__)


class LogarithmicValidationCallback(L.Callback):
    """Exponentially increases the validation interval after each validation run.

    Starts validating every ``start_step`` batches, then multiplies the
    interval by ``growth`` after each validation. Clamped to
    ``[min_interval, max_interval]``. This amortises the cost of retrieval
    evaluation over long training runs.

    Parameters
    ----------
    start_step : int, optional
        Initial validation interval in training batches (default: 50).
    growth : float, optional
        Multiplicative growth factor applied after each validation (default: 2.0).
    min_interval : int, optional
        Minimum interval floor (default: 1).
    max_interval : int, optional
        Interval ceiling; ``None`` means no cap (default: ``None``).
    """

    def __init__(
        self,
        start_step: int = 50,
        growth: float = 2.0,
        min_interval: int = 1,
        max_interval: Optional[int] = None,
    ):
        assert start_step >= 1 and growth >= 1.0
        self.current_interval = int(start_step)
        self.growth = float(growth)
        self.min_interval = int(min_interval)
        self.max_interval = int(max_interval) if max_interval is not None else None
        # used to avoid double-applying on resume during same boundary
        self._just_updated = False

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # Ensure epoch-based validation doesn't interfere
        trainer.check_val_every_n_epoch = None
        trainer.val_check_interval = int(self.current_interval)
        trainer.val_check_batch = int(self.current_interval)
        if trainer.is_global_zero:
            logger.info(
                f"[log-val] initial val_check_interval = {self.current_interval}"
                " batches"
            )

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        return

    def on_validation_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        new_interval = max(
            self.min_interval, int(math.ceil(self.current_interval * self.growth))
        )
        if self.max_interval is not None:
            new_interval = min(new_interval, self.max_interval)

        self.current_interval = new_interval

        # Set both attributes so Lightning's loop uses the new batch-based threshold.
        trainer.val_check_interval = int(self.current_interval)
        trainer.val_check_batch = int(self.current_interval)

        # synchronize all ranks to prevent races / mismatched internal state
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        if trainer.is_global_zero:
            logger.info(
                "[log-val] validation finished. Increasing interval -> every"
                f" {self.current_interval} batches"
            )
