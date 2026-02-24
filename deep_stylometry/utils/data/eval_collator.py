# deep_stylometry/utils/data/eval_collator.py

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class EvalCollator:
    """Pads triplet fields to longest in batch, and pads target_indices with
    -1."""

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

        batch = {
            "input_ids": q["input_ids"],
            "attention_mask": q["attention_mask"],
            "pos_input_ids": pos["input_ids"],
            "pos_attention_mask": pos["attention_mask"],
            "neg_input_ids": neg["input_ids"],
            "neg_attention_mask": neg["attention_mask"],
            "index": torch.stack([f["index"] for f in features]),
        }

        # Pad variable-length target_indices with -1
        if (
            "target_indices" in features[0]
            and features[0]["target_indices"] is not None
        ):
            targets = [f["target_indices"] for f in features]
            max_len = max(len(t) for t in targets)
            padded = []
            for t in targets:
                if isinstance(t, torch.Tensor):
                    t = t.tolist()
                pad_len = max_len - len(t)
                padded.append(torch.tensor(t + [-1] * pad_len, dtype=torch.long))
            batch["target_indices"] = torch.stack(padded)

        return batch
