# deep_stylometry/utils/configs/test_config.py

from dataclasses import dataclass
from typing import Literal, Optional

from deep_stylometry.utils.helpers import DictAccessMixin


@dataclass
class TestConfig(DictAccessMixin):
    # --- trainer ---
    device: str = "gpu"
    num_devices_per_trial: int = 3
    num_cpus_per_trial: int = 10
    max_steps: int = -1
    max_epochs: int = 3
    log_every_n_steps: int = 1
    accumulate_grad_batches: int = 4
    gradient_clip_val: Optional[float] = None
    precision: Literal["16-mixed", "32"] = "32"
    # --- tuner ---
    metric: Literal["val_auroc", "val_mrr", "val_total_loss"] = "val_auroc"
    mode: Literal["min", "max"] = "max"
    num_samples: int = 300
    max_concurrent_trials: int = 3
    time_budget_s: int = 151200
    max_t: int = 3
    grace_period = 1
    # --- misc ---
    use_wandb: bool = True
