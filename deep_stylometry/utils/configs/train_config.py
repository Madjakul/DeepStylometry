# deep_stylometry/utils/configs/train_config.py

from dataclasses import dataclass
from typing import Literal, Optional

from deep_stylometry.utils.helpers import DictAccessMixin


@dataclass
class TrainConfig(DictAccessMixin):
    loss: Literal["info_nce", "triplet"] = "info_nce"
    tau: float = 0.07
    margin: Optional[float] = None  # Only used for triplet or hybrid loss
    gather: bool = True  # Whether to gather embeddings across GPUs for loss computation
    # --- optimizer ---
    lr: float = 4.73e-5
    weight_decay: float = 0.09
    # --- trainer ---
    device: str = "gpu"
    num_devices: int = 3
    strategy: str = "ddp_find_unused_parameters_true"
    process_group_backend: Literal["nccl", "gloo", "mpi"] = "gloo"
    max_steps: int = -1
    max_epochs: int = 1
    val_check_interval: Optional[float] = None
    check_val_every_n_epoch: Optional[int] = None
    log_every_n_steps: int = 1
    accumulate_grad_batches: int = 1
    gradient_clip_val: Optional[float] = None
    precision: Literal["16-mixed", "32"] = "16-mixed"
    # --- misc ---
    use_wandb: bool = True
    log_model: bool = True
    watch: Literal["gradients", "parameters", "all", "none"] = "gradients"
