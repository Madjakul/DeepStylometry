# deep_strylometry/utils/data/se_data.py

from typing import Any, Dict, List

import lightning as L
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

from deep_stylometry.utils.helpers import get_tokenizer


class SEDataModule(L.LightningDataModule):

    def __init__(
        self,
        batch_size: int,
        num_proc: int,
        tokenizer_name: str,
        max_length: int,
        map_batch_size: int,
        load_from_cache_file: bool,
        cache_dir: str,
        ds_name: str = "Madjakul/StyleEmbeddingPairwiseData",
        mlm_collator: bool = False,
    ):
        super().__init__()
        self.ds_name = ds_name
        self.batch_size = batch_size
        self.num_proc = num_proc
        self.max_length = max_length
        self.load_from_cache_file = load_from_cache_file
        self.cache_dir = cache_dir
        self.map_batch_size = map_batch_size
        self.tokenizer = get_tokenizer(tokenizer_name)
        if mlm_collator:
            self.mlm_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm_probability=0.15,
            )
        else:
            self.mlm_collator = None

    def prepare_data(self):
        load_dataset(self.ds_name, cache_dir=self.cache_dir)

    def tokenize_function(self, batch: Dict[str, List[Any]]):
        # qs = [
        #     text if isinstance(text, str) or text is None else str(text)
        #     for text in batch["query_text"]
        # ]
        # ks = [
        #     text if isinstance(text, str) or text is None else str(text)
        #     for text in batch["key_text"]
        # ]

        tokenized_q = self.tokenizer(
            # qs,
            batch["query_text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        tokenized_k = self.tokenizer(
            # ks,
            batch["key_text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        return {
            "input_ids": tokenized_q["input_ids"],
            "attention_mask": tokenized_q["attention_mask"],
            "k_input_ids": tokenized_k["input_ids"],
            "k_attention_mask": tokenized_k["attention_mask"],
            "author_label": batch["author_label"],
        }

    def setup(self, stage: str):
        ds = load_dataset(self.ds_name, cache_dir=self.cache_dir)
        columns_to_remove = ds["train"].column_names  # type: ignore

        if stage == "fit" or stage is None:
            train_dataset = (
                ds["train"]
                .select(range(4))
                .map(  # type: ignore
                    self.tokenize_function,
                    batched=True,
                    batch_size=self.map_batch_size,
                    num_proc=self.num_proc,
                    load_from_cache_file=self.load_from_cache_file,
                    remove_columns=columns_to_remove,
                )
            )
            # val_dataset = ds["validation"].map(  # type: ignore
            #     self.tokenize_function,
            #     batched=True,
            #     batch_size=self.map_batch_size,
            #     num_proc=self.num_proc,
            #     load_from_cache_file=self.load_from_cache_file,
            #     remove_columns=columns_to_remove,
            # )
            self.train_dataset = train_dataset.with_format("torch")
            # self.val_dataset = val_dataset.with_format("torch")

        if stage == "test" or stage is None:
            test_dataset = ds["test"].map(  # type: ignore
                self.tokenize_function,
                batch_size=self.map_batch_size,
                batched=True,
                num_proc=self.num_proc,
                load_from_cache_file=self.load_from_cache_file,
                remove_columns=columns_to_remove,
            )
            self.test_dataset = test_dataset.with_format("torch")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_proc,
            collate_fn=self.mlm_collator,
        )

    # def val_dataloader(self):
    #     return DataLoader(
    #         self.val_dataset, batch_size=self.batch_size, num_workers=self.num_proc  # type: ignore
    #     )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_proc  # type: ignore
        )
