# deep_stylometry/utils/data/triplet_collator.py

from dataclasses import dataclass
from typing import Any, Dict, List
from transformers import PreTrainedTokenizerBase
import torch


@dataclass
class TripletDataCollator:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        def pad_field(ids_key, mask_key):
            return self.tokenizer.pad(
                {
                    "input_ids": [f[ids_key] for f in features],
                    "attention_mask": [f[mask_key] for f in features],
                },
                padding=True,
                return_tensors="pt",
            )

        q = pad_field("input_ids", "attention_mask")
        pos = pad_field("pos_input_ids", "pos_attention_mask")
        neg = pad_field("neg_input_ids", "neg_attention_mask")

        return {
            "input_ids": q["input_ids"],
            "attention_mask": q["attention_mask"],
            "pos_input_ids": pos["input_ids"],
            "pos_attention_mask": pos["attention_mask"],
            "neg_input_ids": neg["input_ids"],
            "neg_attention_mask": neg["attention_mask"],
            "index": torch.stack([f["index"] for f in features]),
        }
