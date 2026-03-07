# deep_stylometry/callbacks/logarithmic_validation_callback.py

import logging
import math
from typing import Optional

import lightning as L
import torch


class LogarithmicValidationCallback(L.Callback):

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
            logging.info(
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
            logging.info(
                "[log-val] validation finished. Increasing interval -> every"
                f" {self.current_interval} batches"
            )
