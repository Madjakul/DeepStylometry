# deep_stylometry/utils/data/se_datamodule.py

from typing import Any, Dict, List

import lightning as L
from datasets import load_dataset
from torch.utils.data import DataLoader

from deep_stylometry.utils.data.custom_data_collator import \
    CustomDataCollatorForLanguageModeling
from deep_stylometry.utils.helpers import get_tokenizer


class StyleEmbeddingDataModule(L.LightningDataModule):
    """Data module for the [Style Embedding dataset](https://huggingface.co/AnnaWegmann/Style-Embedding).
    This dataset contains text triplets with an Anchor A, and two utterances U1 and U2.
    The label indicates whether U1 or U2 are from the same author as A.
    This data module turns the triplet into a query and two keys, positive and hard
    negative.

    Parameters
    ----------
    batch_size: int
        The batch size for training and validation.
    num_proc: int
        The number of processes to use for data loading.
    tokenizer_name: str
        The name of the tokenizer to use for tokenizing the text.
    max_length: int
        The maximum length of the tokenized sequences.
    map_batch_size: int
        The batch size to use when mapping the dataset.
    load_from_cache_file: bool
        Whether to load the dataset from cache if available.
    cache_dir: str
        The directory where the dataset cache is stored.
    ds_name: str, optional
        The name of the dataset to load. Default is "AnnaWegmann/StyleEmbeddingData".
    mlm_collator: bool, optional
        Whether to use a data collator for masked language modeling. Default is False.
    """

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
        ds_name: str = "AnnaWegmann/StyleEmbeddingData",
        mlm_collator: bool = False,
        **kwargs: Any,
    ) -> None:
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
            self.mlm_collator = CustomDataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm_probability=0.15,
            )
        else:
            self.mlm_collator = None
        self.tuning_mode = kwargs.get("tuning_mode", False)

    def prepare_data(self) -> None:
        load_dataset(self.ds_name, cache_dir=self.cache_dir)

    def tokenize_function(self, batch: Dict[str, List[Any]]):
        anchors = [
            str(anchor) if not isinstance(anchor, str) else anchor
            for anchor in batch["Anchor (A)"]
        ]
        pos_texts = []
        neg_texts = []
        for label, u1, u2 in zip(
            batch["Same Author Label"],
            batch["Utterance 1 (U1)"],
            batch["Utterance 2 (U2)"],
        ):
            if label == 1:
                pos_texts.append(u1)
                neg_texts.append(u2)
            else:
                pos_texts.append(u2)
                neg_texts.append(u1)

        tokenized_q = self.tokenizer(
            anchors,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        tokenized_pos = self.tokenizer(
            pos_texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        tokenized_neg = self.tokenizer(
            neg_texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        return {
            "input_ids": tokenized_q["input_ids"],
            "attention_mask": tokenized_q["attention_mask"],
            "pos_input_ids": tokenized_pos["input_ids"],
            "pos_attention_mask": tokenized_pos["attention_mask"],
            "neg_input_ids": tokenized_neg["input_ids"],
            "neg_attention_mask": tokenized_neg["attention_mask"],
        }

    def setup(self, stage: str) -> None:
        ds = load_dataset(self.ds_name, cache_dir=self.cache_dir)
        columns_to_remove = ds["train"].column_names  # type: ignore

        if stage == "fit" or stage is None:
            if self.tuning_mode:
                # train -> val, val -> test
                train_dataset = ds["validation"].map(  # type: ignore
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
                val_dataset = ds["validation"].map(  # type: ignore
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

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,  # type: ignore
            batch_size=self.batch_size,
            num_workers=self.num_proc,
            collate_fn=self.mlm_collator,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,  # type: ignore
            batch_size=self.batch_size,
            num_workers=self.num_proc,
            collate_fn=self.mlm_collator if self.tuning_mode else None,
            shuffle=True if self.tuning_mode else False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,  # type: ignore
            batch_size=self.batch_size,
            num_workers=self.num_proc,
            shuffle=False,
        )
