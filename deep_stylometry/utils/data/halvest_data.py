# deep_stylometry/utils/data/halvest_data.py

from typing import Any, Dict, List

import lightning as L
from datasets import load_dataset
from deep_stylometry.utils.helpers import get_tokenizer
from torch.utils.data import DataLoader


class SEDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_proc: int,
        tokenizer_name: str,
        max_length: int,
        ds_name: str = "AnnaWegmann/StyleEmbeddingData",
    ):
        super().__init__()
        self.ds_name = ds_name
        self.batch_size = batch_size
        self.num_proc = num_proc
        self.max_length = max_length
        self.tokenizer = get_tokenizer(tokenizer_name)

    def prepare_data(self):
        load_dataset(self.ds_name)

    def tokenize_function(self, examples: Dict[str, List[Any]]):
        anchors = [
            str(text) if text is not None else "" for text in examples["Anchor (A)"]
        ]
        u1s = [
            str(text) if text is not None else ""
            for text in examples["Utterance 1 (U1)"]
        ]
        u2s = [
            str(text) if text is not None else ""
            for text in examples["Utterance 2 (U2)"]
        ]

        # Tokenize with empty string handling
        tokenized_anchor = self.tokenizer(
            anchors,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokenized_u1 = self.tokenizer(
            u1s,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokenized_u2 = self.tokenizer(
            u2s,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "anchor_input_ids": tokenized_anchor["input_ids"],
            "anchor_attention_mask": tokenized_anchor["attention_mask"],
            "u1_input_ids": tokenized_u1["input_ids"],
            "u1_attention_mask": tokenized_u1["attention_mask"],
            "u2_input_ids": tokenized_u2["input_ids"],
            "u2_attention_mask": tokenized_u2["attention_mask"],
            "label": examples["Same Author Label"],
        }

    def setup(self, stage: str = None):
        ds = load_dataset(self.ds_name)
        columns_to_remove = ds["train"].column_names

        if stage == "fit" or stage is None:
            train_dataset = ds["train"].map(
                self.tokenize_function,
                batched=True,
                num_proc=self.num_proc,
                remove_columns=columns_to_remove,
            )
            val_dataset = ds["validation"].map(
                self.tokenize_function,
                batched=True,
                num_proc=self.num_proc,
                remove_columns=columns_to_remove,
            )
            self.train_dataset = train_dataset.with_format("torch")
            self.val_dataset = val_dataset.with_format("torch")

        if stage == "test" or stage is None:
            test_dataset = ds["test"].map(
                self.tokenize_function,
                batched=True,
                num_proc=self.num_proc,
                remove_columns=columns_to_remove,
            )
            self.test_dataset = test_dataset.with_format("torch")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_proc
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_proc
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_proc
        )
