# deep_stylometry/callbacks/eval_runtime_monitor.py

import time

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only


class EvalRuntimeMonitor(Callback):

    def __init__(self):
        super().__init__()
        self.validation_start_time = None
        self.test_start_time = None

    @rank_zero_only
    def on_validation_epoch_start(self, trainer, pl_module):
        """Record the start time of the validation epoch."""
        self.validation_start_time = time.monotonic()

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        """Compute and log the validation epoch's total runtime."""
        if self.validation_start_time is not None:
            runtime = time.monotonic() - self.validation_start_time
            pl_module.log(
                "val/runtime", runtime, on_step=False, on_epoch=True, sync_dist=False
            )
            self.validation_start_time = None

    @rank_zero_only
    def on_test_epoch_start(self, trainer, pl_module):
        """Record the start time of the test epoch."""
        self.test_start_time = time.monotonic()

    @rank_zero_only
    def on_test_epoch_end(self, trainer, pl_module):
        """Compute and log the test epoch's total runtime."""
        if self.test_start_time is not None:
            runtime = time.monotonic() - self.test_start_time
            pl_module.log(
                "test/runtime", runtime, on_step=False, on_epoch=True, sync_dist=False
            )
            self.test_start_time = None
