# deep_stylometry/utils/configs/test_config.py

from dataclasses import dataclass
from typing import Literal, Optional

from deep_stylometry.utils.helpers import DictAccessMixin


@dataclass
class TestConfig(DictAccessMixin):
    # --- trainer ---
    device: str = "gpu"
    log_every_n_steps: int = 1
    precision: Literal["16-mixed", "32"] = "32"
    # --- misc ---
    use_wandb: bool = True
