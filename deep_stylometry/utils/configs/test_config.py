# deep_stylometry/utils/configs/test_config.py

from dataclasses import dataclass
from typing import Literal

from deep_stylometry.utils.helpers import DictAccessMixin


@dataclass
class TestConfig(DictAccessMixin):
    # --- trainer ---
    device: str = "gpu"
    num_devices: int = 1
    log_every_n_steps: int = 1
    precision: Literal["16-mixed", "32"] = "16-mixed"
    # --- misc ---
    use_wandb: bool = True
