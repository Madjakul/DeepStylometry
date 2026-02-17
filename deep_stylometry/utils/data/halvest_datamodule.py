# halvesting_contrastive/utils/data/halvest_contrastive_datamodule.py

import logging
import os
import os.path as osp
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import datasets
import lightning as L
from torch.utils.data import DataLoader
from tqdm import tqdm

from deep_stylometry.utils.helpers import get_tokenizer
from deep_stylometry.utils.data.triplet_collator import TripletDataCollator

if TYPE_CHECKING:
    from deep_stylometry.utils.configs.base_config import BaseConfig


class HALvestContrastiveDatamodule(L.LightningDataModule):

    val_targets: List[List]
    test_targets: List[List]

    def __init__(
        self,
        cfg: "BaseConfig",
        processed_ds_dir: str,
        num_proc: int,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.processed_ds_dir = processed_ds_dir
        self.num_proc = num_proc
        self.cache_dir = cache_dir
        self.tokenizer = get_tokenizer(cfg.data.tokenizer_name)

    def tokenize(
        self, batch: Dict[str, List[str]], indices: List[int]
    ) -> Dict[str, Any]:
        tokenized_q = self.tokenizer(
            batch["query"],
            truncation=self.cfg.data.truncation,
            padding=self.cfg.data.padding,
            max_length=self.cfg.data.max_length,
            add_special_tokens=self.cfg.data.add_special_tokens,
        )
        tokenized_pos = self.tokenizer(
            batch["positive"],
            truncation=self.cfg.data.truncation,
            padding=self.cfg.data.padding,
            max_length=self.cfg.data.max_length,
            add_special_tokens=self.cfg.data.add_special_tokens,
        )
        tokenized_neg = self.tokenizer(
            batch["negative"],
            truncation=self.cfg.data.truncation,
            padding=self.cfg.data.padding,
            max_length=self.cfg.data.max_length,
            add_special_tokens=self.cfg.data.add_special_tokens,
        )
        return {
            "input_ids": tokenized_q["input_ids"],
            "attention_mask": tokenized_q["attention_mask"],
            "pos_input_ids": tokenized_pos["input_ids"],
            "pos_attention_mask": tokenized_pos["attention_mask"],
            "neg_input_ids": tokenized_neg["input_ids"],
            "neg_attention_mask": tokenized_neg["attention_mask"],
            "index": indices,
        }

    def prepare_data(self) -> None:
        for subset in self.cfg.data.subsets:
            path = osp.join(self.processed_ds_dir, subset)
            if osp.exists(path) and osp.getsize(path) > 0:
                continue
            datasets.load_dataset("almanach/halvest-contrastive", name=subset)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.fit_setup()
        if stage == "test" or stage is None:
            self.test_setup()

    def fit_setup(self) -> None:
        train_dss = []
        for subset in self.cfg.data.subsets:
            subset_path = osp.join(self.processed_ds_dir, subset)
            train_path = osp.join(subset_path, "train")

            if osp.exists(train_path):
                logging.info(
                    f"Loading processed data for train from disk: {train_path}"
                )
                train_dss.append(datasets.load_from_disk(train_path))
                continue

            logging.info(
                "Processed data not found for train. Running full preprocessing pipeline..."
            )
            ds = datasets.load_dataset(
                path="almanach/halvest-contrastive",
                name=subset,
                split="train",
                cache_dir=self.cache_dir,
            )
            columns = ds.column_names  # type: ignore
            logging.info("Tokenizing train triplets...")
            ds = ds.map(
                self.tokenize,
                batched=True,
                with_indices=True,
                num_proc=self.num_proc,
                remove_columns=columns,
                load_from_cache_file=self.cfg.data.load_from_cache_file,
            )

            train_dss.append(ds)

            logging.info(f"Saving processed data to disk: {train_path}")
            ds.set_format("torch")
            os.makedirs(self.processed_ds_dir, exist_ok=True)
            ds.save_to_disk(train_path)

        self.train_ds = datasets.concatenate_datasets(train_dss)
        self.train_ds.set_format("torch")

        subset = "base-4"
        subset_path = osp.join(self.processed_ds_dir, subset)
        val_path = osp.join(subset_path, "val")
        if osp.exists(val_path):
            logging.info(f"Loading processed data for validation from disk: {val_path}")
            self.val_ds = datasets.load_from_disk(val_path)
            return

        logging.info(
            "Processed data not found for val. Running full preprocessing pipeline..."
        )
        ds = datasets.load_dataset(
            path="almanach/halvest-contrastive",
            name=subset,  # We only take the first subset for validation
            split="valid",
            cache_dir=self.cache_dir,
        )

        columns = ds.column_names
        self.val_ds = ds.map(
            self.tokenize,
            batched=True,
            with_indices=True,
            num_proc=self.num_proc,
            remove_columns=columns,
            load_from_cache_file=self.cfg.data.load_from_cache_file,
        )

        val_targets = self._get_targets(self.val_ds)
        self.val_ds = self.val_ds.add_column("target_indices", val_targets)
        self.val_ds.set_format("torch")

        logging.info(f"Saving processed data to disk: {val_path}")
        ds.set_format("torch")
        os.makedirs(self.processed_ds_dir, exist_ok=True)
        ds.save_to_disk(val_path)

    def test_setup(self) -> None:
        subset = self.cfg.data.test_subset
        subset_path = osp.join(self.processed_ds_dir, subset)
        test_path = osp.join(subset_path, "test")
        if osp.exists(test_path):
            logging.info(f"Loading processed data for test from disk: {test_path}")
            self.test_ds = datasets.load_from_disk(test_path)
            return

        logging.info(
            "Processed data not found for test. Running full preprocessing pipeline..."
        )
        ds = datasets.load_dataset(
            path="almanach/halvest-contrastive",
            name=subset,  # We only take the first subset for validation
            split="test",
            cache_dir=self.cache_dir,
        )

        test_targets = self._get_targets(ds)
        ds = ds.add_column("target_indices", test_targets)
        logging.info(f"Saving processed data to disk: {test_path}")
        ds.set_format("torch")
        os.makedirs(self.processed_ds_dir, exist_ok=True)
        ds.save_to_disk(test_path)

        columns = ds.column_names  # type: ignore
        self.test_ds = ds.map(
            self.tokenize,
            batched=True,
            with_indices=True,
            num_proc=self.num_proc,
            remove_columns=columns,
            load_from_cache_file=self.cfg.data.load_from_cache_file,
        )
        self.test_ds.set_format("torch")

        logging.info(f"Saving processed data to disk: {test_path}")
        self.test_ds.save_to_disk(test_path)

    def train_dataloader(self) -> DataLoader:
        collate_fn = None
        if self.cfg.train.gather:
            # Use a custom collate_fn to gather positive and negative samples across the batch
            collate_fn = TripletDataCollator(
                tokenizer=self.tokenizer, padding="longest", return_tensors="pt"
            )
        return DataLoader(
            self.train_ds,  # type: ignore
            batch_size=self.cfg.data.batch_size,
            num_workers=self.num_proc,
            shuffle=self.cfg.data.shuffle,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,  # type: ignore
            batch_size=self.cfg.data.batch_size,
            num_workers=self.num_proc,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,  # type: ignore
            batch_size=self.cfg.data.batch_size,
            num_workers=self.num_proc,
            shuffle=False,
        )

    @staticmethod
    def _get_targets(ds: datasets.Dataset) -> List[List]:
        auth_to_idx = defaultdict(list)
        idx_to_auth = dict()
        for idx, row in enumerate(tqdm(ds, desc="Processing targets")):
            pos_auths = frozenset(set(row["pos_authorids"]))
            auth_to_idx[pos_auths].append(idx)
            idx_to_auth[idx] = pos_auths

            neg_auths = frozenset(set(row["neg_authorids"]))
            auth_to_idx[neg_auths].append(idx + len(ds))
            idx_to_auth[idx + len(ds)] = neg_auths
        targets = [auth_to_idx[idx_to_auth[i]] for i in range(len(ds))]
        return targets
