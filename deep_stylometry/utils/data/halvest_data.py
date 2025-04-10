# deep_stylometry/utils/data/halvest_data.py

from typing import Any, Dict, List, Optional

import lightning as L
from datasets import load_dataset
from torch.utils.data import DataLoader

from deep_stylometry.utils.helpers import get_tokenizer


class HALvestDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_proc: int,
        tokenizer_name: str,
        max_length: int,
        ds_name: str = "almanach/HALvest-Contrastive",
        config_name: Optional[str] = None,
    ):
        super().__init__()
        self.ds_name = ds_name
        self.config_name = config_name if config_name is not None else "base-2"
        self.batch_size = batch_size
        self.num_proc = num_proc
        self.max_length = max_length
        self.tokenizer = get_tokenizer(tokenizer_name)
        if tokenizer_name == "openai-community/gpt2":
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def prepare_data(self):
        load_dataset(self.ds_name, self.config_name)

    def tokenize_function(self, batch: Dict[str, List[Any]]):
        qs = [
            text if isinstance(text, str) or text is None else str(text)
            for text in batch["query_text"]
        ]
        ks = [
            text if isinstance(text, str) or text is None else str(text)
            for text in batch["key_text"]
        ]

        tokenized_q = self.tokenizer(
            qs,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        tokenized_k = self.tokenizer(
            ks,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        return {
            "q_input_ids": tokenized_q["input_ids"],
            "q_attention_mask": tokenized_q["attention_mask"],
            "k_input_ids": tokenized_k["input_ids"],
            "k_attention_mask": tokenized_k["attention_mask"],
            "domain_label": batch["domain_label"],
            "affiliation_label": batch["affiliation_label"],
            "author_label": batch["author_label"],
        }

    def setup(self, stage: str):
        ds = load_dataset(self.ds_name, self.config_name)
        columns_to_remove = ds["train"].column_names  # type: ignore

        if stage == "fit" or stage is None:
            train_dataset = (
                ds["train"]
                .select(range(8))
                .map(  # type: ignore
                    self.tokenize_function,
                    batched=True,
                    num_proc=self.num_proc,
                    remove_columns=columns_to_remove,
                )
            )
            # val_dataset = ds["valid"].map(  # type: ignore
            #     self.tokenize_function,
            #     batched=True,
            #     num_proc=self.num_proc,
            #     remove_columns=columns_to_remove,
            # )
            self.train_dataset = train_dataset.with_format("torch")
            # self.val_dataset = val_dataset.with_format("torch")

        if stage == "test" or stage is None:
            test_dataset = ds["test"].map(  # type: ignore
                self.tokenize_function,
                batched=True,
                num_proc=self.num_proc,
                remove_columns=columns_to_remove,
            )
            self.test_dataset = test_dataset.with_format("torch")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_proc, drop_last=True  # type: ignore
        )

    #
    # def val_dataloader(self):
    #     return DataLoader(
    #         self.val_dataset, batch_size=self.batch_size, num_workers=self.num_proc, drop_last=True  # type: ignore
    #     )
    #
    # def test_dataloader(self):
    #     return DataLoader(
    #         self.test_dataset, batch_size=self.batch_size, num_workers=self.num_proc, drop_last=True  # type: ignore
    #     )
