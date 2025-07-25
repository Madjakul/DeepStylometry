# deep_stylometry/utils/configs/tune_config.py

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

from deep_stylometry.utils.helpers import DictAccessMixin


@dataclass
class TuneConfig(DictAccessMixin):

    loss: Literal["info_nce", "triplet", "hybrid", "hard_margin", "margin"] = "info_nce"
    tau: Dict = field(default_factory=dict)  # Only with info_nce or hybrid loss
    lambda_: Dict = field(default_factory=dict)  # Only with hybrid loss
    margin: Optional[float] = None  # Only used for triplet loss
    lm_loss_weight: float = 0.0
    # --- optimizer ---
    lr: Dict = field(default_factory=dict)
    betas: Dict = field(default_factory=dict)
    eps: Dict = field(default_factory=dict)
    weight_decay: Dict = field(default_factory=dict)
    num_cycles: float = 0.5
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
