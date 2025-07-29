# deep_stylometry/utils/configs/train_config.py

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

from deep_stylometry.utils.helpers import DictAccessMixin


@dataclass
class TrainConfig(DictAccessMixin):
    loss: Literal["info_nce", "triplet", "hybrid", "hard_margin", "margin"] = "info_nce"
    tau: float = 0.07
    lambda_: float = 0.5
    margin: Optional[float] = None  # Only used for triplet or hybrid loss
    lm_loss_weight: float = 0.0
    # --- optimizer ---
    lr: float = 4.73e-5
    betas: Tuple[float, float] = (0.7, 0.999)
    eps: float = 1e-9
    weight_decay: float = 0.09
    num_cycles: float = 0.5
    # --- checkpointing ---
    checkpoint_metric: Literal["val_total_loss", "val_auroc", "val_hr1", "val_mrr"] = (
        "val_total_loss"
    )
    checkpoint_mode: Literal["min", "max"] = "min"
    save_top_k: int = 2
    # --- trainer ---
    device: str = "gpu"
    num_devices: int = 3
    strategy: str = "ddp_find_unused_parameters_true"
    process_group_backend: Literal["nccl", "gloo", "mpi"] = "gloo"
    max_steps: int = -1
    max_epochs: int = 4
    val_check_interval: Optional[float] = None
    check_val_every_n_epoch: Optional[int] = None
    log_every_n_steps: int = 1
    accumulate_grad_batches: int = 4
    gradient_clip_val: Optional[float] = None
    precision: Literal["16-mixed", "32"] = "32"
    # --- misc ---
    use_wandb: bool = True
    log_model: bool = True
    watch: Literal["gradients", "parameters", "all", "none"] = "gradients"
