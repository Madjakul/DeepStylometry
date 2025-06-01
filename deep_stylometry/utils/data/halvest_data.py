# deep_stylometry/utils/data/halvest_data.py

from typing import Any, Dict, List, Optional

import lightning as L
from datasets import load_dataset
from torch.utils.data import DataLoader

from deep_stylometry.utils.data.custom_data_collator import \
    CustomDataCollatorForLanguageModeling
from deep_stylometry.utils.helpers import get_tokenizer


class HALvestDataModule(L.LightningDataModule):

    test_dataset = None

    def __init__(
        self,
        batch_size: int,
        num_proc: int,
        tokenizer_name: str,
        max_length: int,
        map_batch_size: int,
        load_from_cache_file: bool,
        cache_dir: str,
        ds_name: str = "almanach/HALvest-Contrastive",
        config_name: Optional[str] = None,
        mlm_collator: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        self.ds_name = ds_name
        self.config_name = config_name if config_name is not None else "base-2"
        self.batch_size = batch_size
        self.num_proc = num_proc
        self.max_length = max_length
        self.load_from_cache_file = load_from_cache_file
        self.cache_dir = cache_dir
        self.map_batch_size = map_batch_size
        self.tokenizer = get_tokenizer(tokenizer_name)
        if mlm_collator:
            self.mlm_collator = CustomDataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm_probability=0.15,
            )
        else:
            self.mlm_collator = None
        # Get tuning_mode from kwargs, default to False
        self.tuning_mode = kwargs.get("tuning_mode", False)

    def prepare_data(self):
        load_dataset(self.ds_name, self.config_name, cache_dir=self.cache_dir)

    def tokenize_function(self, batch: Dict[str, List[Any]]):
        tokenized_q = self.tokenizer(
            batch["query_text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        tokenized_k = self.tokenizer(
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
            "domain_label": batch["domain_label"],
            "affiliation_label": batch["affiliation_label"],
            "author_label": batch["author_label"],
        }

    def setup(self, stage: str):
        ds = load_dataset(self.ds_name, self.config_name, cache_dir=self.cache_dir)
        columns_to_remove = ds["train"].column_names  # type: ignore

        if stage == "fit" or stage is None:
            if self.tuning_mode:
                # train -> valid, valid -> test
                train_dataset = ds["valid"].map(  # type: ignore
                    self.tokenize_function,
                    batched=True,
                    batch_size=self.map_batch_size,
                    num_proc=self.num_proc,
                    load_from_cache_file=self.load_from_cache_file,
                    remove_columns=columns_to_remove,
                )
                val_dataset = ds["test"].map(  # type: ignore
                    self.tokenize_function,
                    batched=True,
                    batch_size=self.map_batch_size,
                    num_proc=self.num_proc,
                    load_from_cache_file=self.load_from_cache_file,
                    remove_columns=columns_to_remove,
                )
                self.train_dataset = train_dataset.with_format("torch")
                self.val_dataset = val_dataset.with_format("torch")
            else:
                train_dataset = ds["train"].map(  # type: ignore
                    self.tokenize_function,
                    batched=True,
                    batch_size=self.map_batch_size,
                    num_proc=self.num_proc,
                    load_from_cache_file=self.load_from_cache_file,
                    remove_columns=columns_to_remove,
                )
                val_dataset = ds["valid"].map(  # type: ignore
                    self.tokenize_function,
                    batched=True,
                    batch_size=self.map_batch_size,
                    num_proc=self.num_proc,
                    load_from_cache_file=self.load_from_cache_file,
                    remove_columns=columns_to_remove,
                )
                self.train_dataset = train_dataset.with_format("torch")
                self.val_dataset = val_dataset.with_format("torch")

        if (stage == "test" or stage is None) and self.test_dataset is None:
            test_dataset = ds["test"].map(  # type: ignore
                self.tokenize_function,
                batched=True,
                batch_size=self.map_batch_size,
                num_proc=self.num_proc,
                load_from_cache_file=self.load_from_cache_file,
                remove_columns=columns_to_remove,
            )
            self.test_dataset = test_dataset.with_format("torch")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,  # type: ignore
            batch_size=self.batch_size,
            num_workers=self.num_proc,
            collate_fn=self.mlm_collator,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,  # type: ignore
            batch_size=self.batch_size,
            num_workers=self.num_proc,
            collate_fn=self.mlm_collator,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_proc  # type: ignore
        )
