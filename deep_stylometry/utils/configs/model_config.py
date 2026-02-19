# deep_stylometry/utils/configs/model_config.py

from dataclasses import dataclass
from typing import Literal

from deep_stylometry.utils.helpers import DictAccessMixin


@dataclass
class ModelConfig(DictAccessMixin):

    base_checkpoint: str = "answerdotai/ModernBERT-base"
    dropout: float = 0.1
    expansion_ratio: int = 4
    pooling_method: Literal["mean", "li"] = "li"
    skip_list: bool = False
