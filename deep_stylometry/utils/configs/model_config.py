# deep_stylometry/utils/configs/model_config.py

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Union

from deep_stylometry.utils.helpers import DictAccessMixin


@dataclass
class ModelConfig(DictAccessMixin):

    base_model_name: str = "FacebookAI/roberta-base"
    is_decoder_model: bool = False
    add_linear_layers: bool = True
    dropout: float = 0.1
    lm_weight: float = 0.0
    contrastive_weight: float = 1.0
    contrastive_temp: Union[float, Dict] = 0.98
    pooling_method: Literal["mean", "li"] = "li"
    # --- If the pooling method is late interaction ---
    distance_weightning: Union[Literal["none", "exp", "linear"], Dict] = "none"
    # If you use distance weightning
    alpha: Union[float, Dict] = 0.22
    use_softmax: bool = True
    initial_gumbel_temp: Optional[Union[float, Dict]] = None
    auto_anneal_gumbel: bool = True
    min_gumbel_temp: float = 0.5
