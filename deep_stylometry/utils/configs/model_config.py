# deep_stylometry/utils/configs/model_config.py

from dataclasses import dataclass
from typing import Literal

from deep_stylometry.utils.helpers import DictAccessMixin


@dataclass
class ModelConfig(DictAccessMixin):

    base_checkpoint: str = "FacebookAI/roberta-base"
    add_linear_layers: bool = True
    dropout: float = 0.1
    pooling_method: Literal["mean", "li"] = "li"
    skip_list: bool = False
