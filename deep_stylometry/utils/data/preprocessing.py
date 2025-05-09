# deep_stylometry/utils/data/preprocessing.py

import logging
from typing import Any, Dict, List

from datasets import DatasetDict, load_dataset


def _transform_triplet_to_pairwise(batch: Dict[str, List[Any]]):
    """Transforms a batch of triplet data into pairwise data.

    Each triplet (A, U1, U2, label) yields two pairs:
    - One positive pair (A, Positive Utterance) with label 1
    - One negative pair (A, Negative Utterance) with label 0

    Parameters
    ----------
    batch: Dict[str, List[Any]]
        A batch of triplet data from StyleEMBeddingData.

    Returns
    -------
    Dict[str, List[Any]]
        A dictionary containing:
        - query_text: List of query texts (A)
        - query_id: List of IDs for query texts
        - key_text: List of key texts (U1 or U2)
        - key_id: List of IDs for key texts
        - author_label: List of labels (1 for positive, 0 for negative)
    """
    new_sentence1 = []
    s1_ids = []
    new_sentence2 = []
    s2_ids = []
    new_labels = []

    anchors = batch["Anchor (A)"]
    anchor_ids = batch["Utterance ID A"]
    u1s = batch["Utterance 1 (U1)"]
    u1_ids = batch["ID U1"]
    u2s = batch["Utterance 2 (U2)"]
    u2_ids = batch["ID U2"]
    labels = batch["Same Author Label"]

    for anchor, anchor_id, u1, u1_id, u2, u2_id, label in zip(
        anchors, anchor_ids, u1s, u1_ids, u2s, u2_ids, labels
    ):
        anchor_str = str(anchor) if anchor is not None else ""
        u1_str = str(u1) if u1 is not None else ""
        u2_str = str(u2) if u2 is not None else ""

        if label == 1:
            # Positive Pair: A and U1
            new_sentence1.append(anchor_str)
            s1_ids.append(anchor_id)
            new_sentence2.append(u1_str)
            s2_ids.append(u1_id)
            new_labels.append(1)
            # Negative Pair: A and U2
            new_sentence1.append(anchor_str)
            s1_ids.append(anchor_id)
            new_sentence2.append(u2_str)
            s2_ids.append(u2_id)
            new_labels.append(0)
        elif label == 0:
            # Positive Pair: A and U2
            new_sentence1.append(anchor_str)
            s1_ids.append(anchor_id)
            new_sentence2.append(u2_str)
            s2_ids.append(u2_id)
            new_labels.append(1)
            # Negative Pair: A and U1
            new_sentence1.append(anchor_str)
            s1_ids.append(anchor_id)
            new_sentence2.append(u1_str)
            s2_ids.append(u1_id)
            new_labels.append(0)
        else:
            logging.warning(f"Unexpected label '{label}' found. Skipping entry.")
            continue

    return {
        "query_text": new_sentence1,
        "query_id": s1_ids,
        "key_text": new_sentence2,
        "key_id": s2_ids,
        "author_label": new_labels,
    }


def _push_to_hub(ds: DatasetDict, ds_name: str):
    """Pushes the transformed dataset to Hugging Face Hub.

    Parameters
    ----------
    ds: DatasetDict
        The transformed dataset to be pushed.
    ds_name: str
        The name of the dataset on Hugging Face Hub.
    """
    logging.info(f"\nAttempting to push transformed dataset to Hub: {ds_name}")
    ds.push_to_hub(ds_name)
    logging.info(
        f"Successfully pushed dataset to: https://huggingface.co/datasets/{ds_name}"
    )


def run(
    target_ds_name: str,
    batch_size: int,
    num_proc: int,
    load_from_cache_file: bool,
    cache_dir: str,
    original_ds_name: str = "AnnaWegmann/StyleEmbeddingData",
):
    """Load and reprocesses the StyleEmbeddingData dataset to create a pairwise
    dataset.

    Parameters
    ----------
    target_ds_name: str
        The name of the dataset to be created on Hugging Face Hub.
    batch_size: int
        The batch size for processing the dataset.
    num_proc: int
        The number of processes to use for parallel processing.
    load_from_cache_file: bool
        Whether to load from cache or not.
    cache_dir: str
        The directory where the dataset is cached.
    original_ds_name: str
        The name of the original dataset to load from Hugging Face Hub.
    """
    logging.info(f"Loading original dataset: {original_ds_name}...")
    original_ds = load_dataset(original_ds_name, cache_dir=cache_dir)
    logging.info("Original dataset loaded.")
    logging.info(original_ds)

    logging.info("Transforming dataset splits...")
    transformed_ds = DatasetDict()
    original_columns = original_ds["train"].column_names  # type: ignore

    for split in original_ds.keys():  # type: ignore
        logging.info(f"Processing split: {split}...")
        transformed_ds[split] = original_ds[split].map(  # type: ignore
            _transform_triplet_to_pairwise,
            batched=True,
            batch_size=batch_size,
            load_from_cache_file=load_from_cache_file,
            num_proc=num_proc,
            remove_columns=original_columns,
            desc=f"Transforming {split} split",
        )

    logging.info("Transformed dataset structure:")
    logging.info(transformed_ds)
    logging.info("Example entry (train split):")
    logging.info(transformed_ds["train"][0])
    logging.info(transformed_ds["train"][1])

    _push_to_hub(transformed_ds, target_ds_name)
