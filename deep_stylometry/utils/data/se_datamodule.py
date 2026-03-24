# deep_stylometry/utils/data/se_datamodule.py
"""StyleEmbedding (Wegmann et al.) datamodule for zero-shot authorship
attribution evaluation.

The dataset consists of Reddit triplets:
  - Anchor (A)  : text from author X
  - Utterance 1 : same or different author compared to A
  - Utterance 2 : the complementary utterance

When ``Same Author Label == 1`` → U1 is the positive (same author),
U2 is the negative.  When ``Same Author Label == 0`` → U2 is the positive,
U1 is the negative.

Only ``test_dataloader`` is implemented; this dataset is used exclusively for
zero-shot out-of-domain evaluation of models trained on HALvest-Contrastive.
"""

import logging
import os
import os.path as osp
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import datasets
import lightning as L
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from deep_stylometry.utils.data.eval_collator import EvalCollator
from deep_stylometry.utils.helpers import get_tokenizer

if TYPE_CHECKING:
    from deep_stylometry.utils.configs.base_config import BaseConfig

_HF_DATASET = "AnnaWegmann/StyleEmbeddingData"


class StyleEmbeddingDatamodule(L.LightningDataModule):
    """Zero-shot test datamodule for the Wegmann StyleEmbedding dataset.

    Parameters
    ----------
    cfg:
        Global configuration.
    processed_ds_dir:
        Directory for caching the tokenised dataset.
    num_proc:
        Number of worker processes for tokenisation.
    cache_dir:
        Optional HuggingFace cache directory.
    split:
        Dataset split to use (default ``"test"``).
    """

    def __init__(
        self,
        cfg: "BaseConfig",
        processed_ds_dir: str,
        num_proc: int = 1,
        cache_dir: Optional[str] = None,
        split: str = "test",
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.processed_ds_dir = processed_ds_dir
        self.num_proc = num_proc
        self.cache_dir = cache_dir
        self.split = split
        self.tokenizer = get_tokenizer(cfg.data.tokenizer_name)

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def tokenize(
        self, batch: Dict[str, List[str]], indices: List[int]
    ) -> Dict[str, Any]:
        """Tokenise a batch of (anchor, positive, negative) triplets."""
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
            "target_indices": [[-1]] * len(batch["query"]),
            "index": indices,
        }

    # ------------------------------------------------------------------
    # Dataset conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_to_triplets(raw_ds: datasets.Dataset) -> datasets.Dataset:
        """Convert the pairwise verification format into (anchor, pos, neg)
        triplets.

        Dataset columns:
          - Anchor (A), Utterance 1 (U1), Utterance 2 (U2)
          - Same Author Label : 1 → author(A) == author(U1); 0 → author(A) != author(U1)

        When label == 1: A=query, U1=positive (same author), U2=negative.
        When label == 0: A=query, U2=positive (same author), U1=negative.
        """
        queries, positives, negatives = [], [], []
        logging.info("Converting StyleEmbedding to triplet format...")
        for row in tqdm(raw_ds, desc="StyleEmbedding triplet conversion"):
            anchor = row["Anchor (A)"]
            u1 = row["Utterance 1 (U1)"]
            u2 = row["Utterance 2 (U2)"]
            label = int(row["Same Author Label"])
            if label == 1:
                queries.append(anchor)
                positives.append(u1)
                negatives.append(u2)
            else:
                queries.append(anchor)
                positives.append(u2)
                negatives.append(u1)

        return datasets.Dataset.from_dict(
            {"query": queries, "positive": positives, "negative": negatives}
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("test", None):
            self._test_setup()

    def _test_setup(self) -> None:
        cache_path = osp.join(self.processed_ds_dir, "se_test")
        if osp.exists(cache_path):
            logging.info(f"Loading cached SE test data from {cache_path}")
            self.test_ds = datasets.load_from_disk(cache_path)
            self.test_ds.set_format("torch")
            return

        logging.info(f"Loading {_HF_DATASET} split={self.split}…")
        raw = datasets.load_dataset(
            _HF_DATASET,
            split=self.split,
            cache_dir=self.cache_dir,
        )
        triplet_ds = self._convert_to_triplets(raw)

        columns = triplet_ds.column_names
        logging.info("Tokenising StyleEmbedding test triplets…")
        self.test_ds = triplet_ds.map(
            self.tokenize,
            batched=True,
            with_indices=True,
            num_proc=self.num_proc,
            remove_columns=columns,
            load_from_cache_file=self.cfg.data.load_from_cache_file,
        )
        self.test_ds.set_format("torch")
        os.makedirs(self.processed_ds_dir, exist_ok=True)
        self.test_ds.save_to_disk(cache_path)
        logging.info(f"Saved SE test data to {cache_path}")

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------

    def test_dataloader(self) -> DataLoader:
        collate_fn = EvalCollator(tokenizer=self.tokenizer)
        return DataLoader(
            self.test_ds,
            batch_size=self.cfg.data.batch_size,
            num_workers=min(self.num_proc, 2),
            shuffle=False,
            collate_fn=collate_fn,
        )
