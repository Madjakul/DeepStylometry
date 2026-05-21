# deep_stylometry/utils/data/pan19_datamodule.py
"""Zero-shot retrieval eval on PAN 2019 English authorship attribution.

Each unknown document is the query; the corpus is the concatenated known texts
per candidate author (9 per problem). A deterministic 80/20 split reserves the
last 20 % of unknowns (sorted by filename) per problem as the *test* split.
"""

import json
import logging
import os
import os.path as osp
import random
import zipfile
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import datasets
import torch
import lightning as L
from torch.utils.data import DataLoader

from deep_stylometry.utils.data.eval_collator import EvalCollator
from deep_stylometry.utils.helpers import get_tokenizer

if TYPE_CHECKING:
    from deep_stylometry.utils.configs.base_config import BaseConfig

logger = logging.getLogger(__name__)


class PAN19Datamodule(L.LightningDataModule):
    """Zero-shot test datamodule for PAN 2019 Cross-Domain Authorship Attribution.

    Only implements ``test_dataloader``; this dataset is used exclusively for
    zero-shot out-of-domain evaluation.

    Parameters
    ----------
    cfg : BaseConfig
        Global configuration.
    processed_ds_dir : str
        Directory for caching the tokenised dataset.
    num_proc : int
        Number of worker processes for tokenisation.
    cache_dir : Optional[str]
        Unused; kept for API consistency with other datamodules.
    pan19_zip : Optional[str]
        Path to the PAN 2019 ZIP archive. Falls back to the
        ``PAN19_ZIP`` environment variable.
    language : str
        ISO language code to filter problems (default ``"en"``).
    split : Optional[str]
        Which split to load — ``"dev"`` (first 80 %) or ``"test"``
        (last 20 %), or ``None`` for all valid queries.
    test_fraction : float
        Fraction of queries per problem reserved for the test
        split (default 0.2).
    seed : int
        Random seed for negative candidate selection (default 42).
    """

    def __init__(
        self,
        cfg: "BaseConfig",
        processed_ds_dir: str,
        num_proc: int = 1,
        cache_dir: Optional[str] = None,
        pan19_zip: Optional[str] = None,
        language: str = "en",
        split: Optional[str] = "test",
        test_fraction: float = 0.2,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.processed_ds_dir = processed_ds_dir
        self.num_proc = num_proc
        self.pan19_zip = pan19_zip or os.environ.get("PAN19_ZIP")
        self.language = language
        self.split = split
        self.test_fraction = test_fraction
        self.seed = seed
        self.tokenizer = get_tokenizer(cfg.data.tokenizer_name)

    # ZIP-based parser (primary — reads directly from the archive)

    @staticmethod
    def _read_zip_text(zf: zipfile.ZipFile, name: str) -> str:
        """Read and decode a text file from an open ZipFile."""
        try:
            return zf.read(name).decode("utf-8").strip()
        except (KeyError, UnicodeDecodeError) as exc:
            logger.warning("Could not read ZIP entry %s: %s", name, exc)
            return ""

    @staticmethod
    def _parse_problems_from_zip(
        zip_path: str,
        language: str = "en",
    ) -> List[Dict[str, Any]]:
        """Parse PAN 2019 problems from the official ZIP archive.

        Returns one dict per valid ``(problem_id, unknown_file)`` pair, i.e.
        multiple dicts per problem. Entries whose ``true-author`` is
        ``"<UNK>"`` (author not in the candidate set) are skipped.

        Returns
        -------
        List[Dict[str, Any]]
            List of dicts with keys:

            - ``problem_id`` (str)
            - ``unknown_file`` (str)  — e.g. ``"unknown00003.txt"``
            - ``unknown_text`` (str)
            - ``candidates`` (dict[str, str])  — candidate_id → full text
            - ``true_author`` (str)
        """
        prefix = (
            "pan19-cross-domain-authorship-attribution-training-dataset-2019-01-23/"
        )
        results: List[Dict[str, Any]] = []

        with zipfile.ZipFile(zip_path, "r") as zf:
            # 1. Identify target problems via collection-info.json
            collection_info: List[Dict] = json.loads(
                zf.read(f"{prefix}collection-info.json")
            )
            target_problems = {
                entry["problem-name"]
                for entry in collection_info
                if entry.get("language") == language
            }
            if not target_problems:
                logger.warning(
                    "No problems found for language '%s' in collection-info.json.",
                    language,
                )
                return results

            logger.info(
                "Found %d '%s' problems: %s",
                len(target_problems),
                language,
                sorted(target_problems),
            )

            # Build an index of all zip entries for fast lookup
            all_names = set(zf.namelist())

            for prob_id in sorted(target_problems):
                prob_prefix = f"{prefix}{prob_id}/"

                # 2. Ground truth
                gt_path = f"{prob_prefix}ground-truth.json"
                if gt_path not in all_names:
                    logger.warning("%s: ground-truth.json missing.", prob_id)
                    continue
                gt_data = json.loads(zf.read(gt_path))
                # Format: {"ground_truth": [{"unknown-text": "...", "true-author": "..."}]}
                gt_list: List[Dict] = gt_data.get("ground_truth", [])
                # Map unknown filename → true author (skip <UNK>)
                gt_map: Dict[str, str] = {
                    entry["unknown-text"]: entry["true-author"]
                    for entry in gt_list
                    if entry.get("true-author") not in (None, "<UNK>")
                }
                if not gt_map:
                    logger.warning(
                        "%s: no valid (non-UNK) ground-truth entries.", prob_id
                    )
                    continue

                # 3. Candidate texts (concatenate all known*.txt in sorted order)
                cand_files: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
                for name in all_names:
                    if not name.startswith(prob_prefix):
                        continue
                    rel = name[len(prob_prefix):]  # e.g. "candidate00001/known00002.txt"
                    parts = rel.split("/")
                    if (
                        len(parts) == 2
                        and parts[0].startswith("candidate")
                        and parts[1].startswith("known")
                        and parts[1].endswith(".txt")
                    ):
                        text = PAN19Datamodule._read_zip_text(zf, name)
                        if text:
                            cand_files[parts[0]].append((parts[1], text))

                candidates: Dict[str, str] = {}
                for cand_id, file_list in cand_files.items():
                    sorted_texts = [t for _, t in sorted(file_list)]
                    combined = " ".join(sorted_texts)
                    if combined:
                        candidates[cand_id] = combined

                if not candidates:
                    logger.warning("%s: no candidate texts found.", prob_id)
                    continue

                # 4. Unknown texts
                # The archive stores unknowns under an "unknown/" subdirectory:
                #   problem00001/unknown/unknown00001.txt
                # Detect this by scanning all_names (ZIP archives may not include
                # explicit directory entries, so we check for any .txt file under
                # the subdirectory rather than relying on the directory marker).
                unk_subdir_prefix = f"{prob_prefix}unknown/"
                unk_in_subdir = any(
                    n.startswith(unk_subdir_prefix) and n.endswith(".txt")
                    for n in all_names
                )

                for unk_file in sorted(gt_map.keys()):
                    true_author = gt_map[unk_file]
                    if true_author not in candidates:
                        logger.warning(
                            "%s/%s: true_author '%s' not in candidates.",
                            prob_id, unk_file, true_author,
                        )
                        continue
                    unk_path = (
                        f"{unk_subdir_prefix}{unk_file}"
                        if unk_in_subdir
                        else f"{prob_prefix}{unk_file}"
                    )
                    unknown_text = PAN19Datamodule._read_zip_text(zf, unk_path)
                    if not unknown_text:
                        logger.warning(
                            "%s/%s: unknown text is empty.", prob_id, unk_file
                        )
                        continue
                    results.append(
                        {
                            "problem_id": prob_id,
                            "unknown_file": unk_file,
                            "unknown_text": unknown_text,
                            "candidates": candidates,
                            "true_author": true_author,
                        }
                    )

        logger.info(
            "Parsed %d valid (query, true-author) pairs from %s.",
            len(results), zip_path,
        )
        return results

    # Triplet conversion

    @staticmethod
    def _apply_split(
        problems: List[Dict[str, Any]],
        split: Optional[str],
        test_fraction: float,
    ) -> List[Dict[str, Any]]:
        """Filter problems to the requested dev/test split."""
        if split is None:
            return problems

        by_prob: Dict[str, List[Dict]] = defaultdict(list)
        for p in problems:
            by_prob[p["problem_id"]].append(p)

        selected: List[Dict] = []
        for prob_id in sorted(by_prob):
            items = sorted(
                by_prob[prob_id],
                key=lambda x: x.get("unknown_file", x["problem_id"]),
            )
            n = len(items)
            n_test = max(1, round(n * test_fraction))
            n_dev = n - n_test
            if split == "dev":
                selected.extend(items[:n_dev])
            elif split == "test":
                selected.extend(items[n_dev:])

        logger.info(
            "Split '%s': %d / %d queries selected.", split, len(selected), len(problems)
        )
        return selected

    @staticmethod
    def _convert_to_triplets(
        problems: List[Dict[str, Any]],
        seed: int = 42,
        split: Optional[str] = None,
        test_fraction: float = 0.2,
    ) -> datasets.Dataset:
        """Convert parsed problem dicts into (query, positive, negative) triplets."""
        problems = PAN19Datamodule._apply_split(problems, split, test_fraction)

        rng = random.Random(seed)
        queries: List[str] = []
        positives: List[str] = []
        negatives: List[str] = []

        for prob in problems:
            true_author = prob["true_author"]
            candidates = prob["candidates"]

            if true_author is None:
                continue
            if true_author not in candidates:
                logger.warning(
                    "%s: true_author '%s' not in candidates — skipped.",
                    prob["problem_id"], true_author,
                )
                continue

            other_candidates = [c for c in candidates if c != true_author]
            if not other_candidates:
                logger.warning(
                    "%s: no other candidates for negative — skipped.",
                    prob["problem_id"],
                )
                continue

            neg_candidate = rng.choice(other_candidates)
            queries.append(prob["unknown_text"])
            positives.append(candidates[true_author])
            negatives.append(candidates[neg_candidate])

        logger.info("Built %d triplets.", len(queries))
        return datasets.Dataset.from_dict(
            {"query": queries, "positive": positives, "negative": negatives}
        )

    # Tokenisation

    def tokenize(
        self, batch: Dict[str, List[str]], indices: List[int]
    ) -> Dict[str, Any]:
        """Tokenise a batch of (query, positive, negative) triplets."""
        tok_q = self.tokenizer(
            batch["query"],
            truncation=self.cfg.data.truncation,
            padding=self.cfg.data.padding,
            max_length=self.cfg.data.max_length,
            add_special_tokens=self.cfg.data.add_special_tokens,
        )
        tok_pos = self.tokenizer(
            batch["positive"],
            truncation=self.cfg.data.truncation,
            padding=self.cfg.data.padding,
            max_length=self.cfg.data.max_length,
            add_special_tokens=self.cfg.data.add_special_tokens,
        )
        tok_neg = self.tokenizer(
            batch["negative"],
            truncation=self.cfg.data.truncation,
            padding=self.cfg.data.padding,
            max_length=self.cfg.data.max_length,
            add_special_tokens=self.cfg.data.add_special_tokens,
        )
        return {
            "input_ids": tok_q["input_ids"],
            "attention_mask": tok_q["attention_mask"],
            "pos_input_ids": tok_pos["input_ids"],
            "pos_attention_mask": tok_pos["attention_mask"],
            "neg_input_ids": tok_neg["input_ids"],
            "neg_attention_mask": tok_neg["attention_mask"],
            # target_indices = [[-1]] signals binary qrels:
            # build_qrels creates {q_i: {d_i: 1}} (each query's true positive
            # is its corresponding positive document in the corpus).
            "target_indices": [[-1]] * len(batch["query"]),
            "index": indices,
        }

    # Setup

    def setup(self, stage: Optional[str] = None) -> None:
        """Prepare the PAN 2019 test dataset."""
        if stage in ("test", None):
            self._test_setup()

    def _test_setup(self) -> None:
        split_tag = self.split or "all"
        cache_path = osp.join(
            self.processed_ds_dir, f"pan19_{self.language}_{split_tag}_test"
        )
        if osp.exists(cache_path):
            logger.info("Loading cached PAN 2019 data from %s", cache_path)
            self.test_ds = datasets.load_from_disk(cache_path)
            self.test_ds.set_format("torch")
            return

        # Parse
        if not self.pan19_zip:
            raise ValueError(
                "No PAN 2019 data source configured. Set pan19_zip= or the "
                "PAN19_ZIP environment variable."
            )
        if not osp.isfile(self.pan19_zip):
            raise FileNotFoundError(f"PAN19 ZIP archive not found: {self.pan19_zip}")
        problems = self._parse_problems_from_zip(self.pan19_zip, language=self.language)

        triplet_ds = self._convert_to_triplets(
            problems,
            seed=self.seed,
            split=self.split,
            test_fraction=self.test_fraction,
        )

        columns = triplet_ds.column_names
        logger.info("Tokenising %d PAN 2019 triplets …", len(triplet_ds))
        # HF datasets forks workers even at num_proc=1; pass None to run inline
        # when CUDA is already initialised (matches halvest_datamodule.test_setup).
        num_proc = None if torch.cuda.is_initialized() else self.num_proc
        self.test_ds = triplet_ds.map(
            self.tokenize,
            batched=True,
            with_indices=True,
            num_proc=num_proc,
            remove_columns=columns,
            load_from_cache_file=self.cfg.data.load_from_cache_file,
        )
        self.test_ds.set_format("torch")
        os.makedirs(self.processed_ds_dir, exist_ok=True)
        self.test_ds.save_to_disk(cache_path)
        logger.info("Saved PAN 2019 test data to %s", cache_path)

    # Dataloaders

    def test_dataloader(self) -> DataLoader:
        """Return the test DataLoader with dynamic padding via EvalCollator."""
        collate_fn = EvalCollator(tokenizer=self.tokenizer)
        return DataLoader(
            self.test_ds,
            batch_size=self.cfg.data.batch_size,
            num_workers=min(self.num_proc, 2),
            shuffle=False,
            collate_fn=collate_fn,
        )
