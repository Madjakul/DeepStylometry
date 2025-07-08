# deep_stylometr/utils/configs/data_config.py

from dataclasses import dataclass
from typing import Literal, Optional

from deep_stylometry.utils.helpers import DictAccessMixin


@dataclass
class DataConfig(DictAccessMixin):
    ds_name: Literal["se", "halvest"] = "se"
    batch_size: int = 32
    tokenizer_name: str = "FacebookAI/roberta-base"
    max_length: int = 512
    map_batch_size: int = 1000
    load_from_cache_file: bool = True
    config_name: Optional[str] = None
    mlm_collator: bool = False
