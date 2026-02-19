# deep_stylometry/callbacks/logarithmic_increase_val_interval.py

import logging
import math
from typing import Optional

import lightning as L
import torch


class LogarithmicValidationCallback(L.Callback):
    """Increase validation interval (in training batches) multiplicatively.

    - Starts with `start_step` (int) meaning "validate every start_step batches".
    - After each validation completion, multiply the integer interval by `growth` (>=1).
    - Works with DDP: updates trainer.val_check_interval AND trainer.val_check_batch on ALL ranks,
      and uses a barrier to synchronize.
    Requirements:
      - Use integer intervals (this implementation is integer-based).
      - Trainer should use batch-based validation (set check_val_every_n_epoch=None or ensure semantics).
      - num_sanity_val_steps=0 recommended for predictable startup behavior.
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
        # Initialize trainer internal interval (do it on all ranks)
        trainer.val_check_interval = int(self.current_interval)
        trainer.val_check_batch = int(self.current_interval)
        if trainer.is_global_zero:
            logging.info(
                f"[log-val] initial val_check_interval = {self.current_interval} batches"
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
        # Ensure we update values only once when hitting the boundary.
        # When Lightning decides to run validation (val_check_batch), it calls validation after this hook,
        # so we use _just_updated to avoid toggling immediately after setting values below.
        # Nothing to do here for this strategy: we only change interval after validation completes.
        return

    def on_validation_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Called on ALL ranks after Lightning finishes a validation run.

        We now increase the integer interval multiplicatively and update
        Trainer state on all ranks.
        """
        # Increase interval multiplicatively (ceil to integer)
        new_interval = max(
            self.min_interval, int(math.ceil(self.current_interval * self.growth))
        )
        if self.max_interval is not None:
            new_interval = min(new_interval, self.max_interval)

        # If training was resumed and trainer.global_step already passed multiple targets,
        # advance until new_interval would schedule in the future (best-effort)
        # but we keep it simple: set to new_interval and rely on Lightning's loop going forward.
        self.current_interval = new_interval

        # Set both attributes so Lightning's loop uses the new batch-based threshold.
        trainer.val_check_interval = int(self.current_interval)
        trainer.val_check_batch = int(self.current_interval)

        # synchronize all ranks to prevent races / mismatched internal state
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        if trainer.is_global_zero:
            logging.info(
                f"[log-val] validation finished. Increasing interval -> every {self.current_interval} batches"
            )
