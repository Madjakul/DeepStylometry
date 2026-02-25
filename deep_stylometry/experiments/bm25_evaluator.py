# deep_stylometry/experiments/bm25_evaluator.py

import logging
from typing import Any, Dict, List

import bm25s
from datasets import Dataset
from ranx import Qrels, Run, evaluate
from tqdm import tqdm
from transformers import AutoTokenizer

from deep_stylometry.utils.helpers import get_tokenizer


def _tokenize_to_strings(
    batch: Dict[str, Any], tokenizer: AutoTokenizer
) -> Dict[str, List[str]]:
    """
    Map function: convert token IDs to space-joined token strings.
    Uses tokenizer.convert_ids_to_tokens (C-backed, fast) instead of
    Python str() loops.
    """
    q_tokens, p_tokens, n_tokens = [], [], []

    for ids, mask in zip(batch["input_ids"], batch["attention_mask"]):
        length = sum(mask)
        q_tokens.append(" ".join(tokenizer.convert_ids_to_tokens(ids[:length])))

    for ids, mask in zip(batch["pos_input_ids"], batch["pos_attention_mask"]):
        length = sum(mask)
        p_tokens.append(" ".join(tokenizer.convert_ids_to_tokens(ids[:length])))

    for ids, mask in zip(batch["neg_input_ids"], batch["neg_attention_mask"]):
        length = sum(mask)
        n_tokens.append(" ".join(tokenizer.convert_ids_to_tokens(ids[:length])))

    return {"q_tokens": q_tokens, "p_tokens": p_tokens, "n_tokens": n_tokens}


def evaluate_bm25(
    ds: Dataset,
    tokenizer_name: str,
    k: int = 100,
    num_proc: int = 4,
) -> Dict[str, float]:
    """Run BM25 retrieval on a triplet dataset.

    Corpus = positives [0, N) + negatives [N, 2N).
    Token IDs are converted to subword strings via the tokenizer.

    Args:
        ds: Dataset with input_ids, attention_mask, pos_*, neg_*, target_indices.
        tokenizer_name: HF tokenizer name for convert_ids_to_tokens.
        k: Cutoff for retrieval metrics.
        num_proc: Workers for ds.map().
    """
    tokenizer = get_tokenizer(tokenizer_name)
    n_queries = len(ds)

    # Batch convert IDs -> token strings via ds.map + multiprocessing
    logging.info("Converting token IDs to strings...")
    tokenized = ds.map(
        _tokenize_to_strings,
        batched=True,
        num_proc=num_proc,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=ds.column_names,
    )

    # Split back into word lists
    query_tokens = [row.split(" ") for row in tokenized["q_tokens"]]
    corpus_tokens = [row.split(" ") for row in tokenized["p_tokens"]] + [
        row.split(" ") for row in tokenized["n_tokens"]
    ]
    n_corpus = len(corpus_tokens)

    # Build qrels
    has_targets = "target_indices" in ds.column_names
    hard = {
        f"q{i}": {f"d{i}": 1}
        for i in tqdm(range(n_queries), desc="Building hard qrels")
    }
    if has_targets:
        soft = {}
        for i in tqdm(range(n_queries), desc="Building soft qrels"):
            targets = ds[i]["target_indices"]
            if isinstance(targets, list):
                soft[f"q{i}"] = {f"d{t}": 1 for t in targets if 0 <= t < n_corpus}
            else:
                soft[f"q{i}"] = {f"d{i}": 1}
    else:
        soft = hard

    hard_qrels = Qrels(hard)
    soft_qrels = Qrels(soft)

    # BM25 — index and retrieve in batch
    logging.info(f"Building BM25 index over {n_corpus} documents...")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    logging.info(f"Retrieving top-{k} for {n_queries} queries...")
    results, scores = retriever.retrieve(query_tokens, k=min(k, n_corpus))
    # results: (n_queries, k) doc indices
    # scores:  (n_queries, k) BM25 scores

    # Build run from batched results
    run_dict: Dict[str, Dict[str, float]] = {}
    for i in range(n_queries):
        run_dict[f"q{i}"] = {
            f"d{int(results[i, j])}": float(scores[i, j])
            for j in range(results.shape[1])
        }

    run = Run(run_dict)

    metrics = {
        f"mrr@{k}": evaluate(hard_qrels, run, f"mrr@{k}"),
        f"ndcg@{k}": evaluate(soft_qrels, run, f"ndcg@{k}"),
        f"recall@{k}": evaluate(soft_qrels, run, f"recall@{k}"),
        "accuracy": evaluate(hard_qrels, run, "precision@1"),
    }
    logging.info(f"BM25: {metrics}")
    return metrics
