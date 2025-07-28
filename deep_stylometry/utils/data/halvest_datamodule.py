# deep_stylometry/utils/data/halvest_data.py

from typing import Any, Dict, List, Optional

import lightning as L
from datasets import load_dataset
from torch.utils.data import DataLoader

from deep_stylometry.utils.data.custom_data_collator import (
    CustomDataCollatorForLanguageModeling,
)
from deep_stylometry.utils.helpers import get_tokenizer


class HALvestDataModule(L.LightningDataModule):

    pass
