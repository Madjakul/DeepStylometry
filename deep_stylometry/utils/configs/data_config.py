# deep_stylometr/utils/configs/data_config.py

from dataclasses import dataclass, field
from typing import List, Literal

from deep_stylometry.utils.helpers import DictAccessMixin


@dataclass
class DataConfig(DictAccessMixin):
    ds_name: Literal["se", "halvest"] = "halvest"
    batch_size: int = 32
    tokenizer_name: str = "answerdotai/ModernBERT-base"
    max_length: int = 512
    padding: Literal["max_length", "longest", "do_not_pad"] = "do_not_pad"
    truncation: Literal[
        "longest_first", "only_first", "only_second", "do_not_truncate"
    ] = "longest_first"
    add_special_tokens: bool = True
    map_batch_size: int = 1000
    load_from_cache_file: bool = True
    subsets: List[str] = field(default_factory=list)
    shuffle: bool = True
    test_subset: str = "base-2"
